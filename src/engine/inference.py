"""High-Resolution Inference Engine with 2D Hann Spatial Overlap-Tile for Static Bursts & Video."""

import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_ops import rearrange


class VideoInferenceEngine:
    """
    Production inference engine supporting both:
      1. `mode="static_burst"` (Default for LucidNet-Static):
         Processes an entire static burst `[1, N, 3, H, W]` in a single spatial
         2D Hann Overlap-Tile pass (aligned to training `tile_size=256`, `tile_overlap=64`),
         computing global `Cn^2` once on the full-resolution burst before tiling and
         returning a single clamped image `[1, 3, H, W]`.
      2. Legacy video modes (`mode="video"` / `mode="forensic_snapshot"`):
         Temporal sliding window with triangular weights + spatial Overlap-Tile.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        window_size: int = 12,
        temporal_stride: int = 6,
        tile_size: int = 256,
        tile_overlap: int = 64,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.window_size = window_size
        self.temporal_stride = temporal_stride
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.use_amp = use_amp and (device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _build_temporal_weights(self, length: int, device: torch.device) -> torch.Tensor:
        if length == 1:
            return torch.ones(1, 1, 1, 1, device=device)
        tau = torch.arange(length, device=device, dtype=torch.float32)
        center = (length - 1) / 2.0
        half = max(1.0, length / 2.0)
        w = (1.0 - torch.abs(tau - center) / half).clamp(min=0.05)
        return w.view(length, 1, 1, 1)

    def _build_hann_window_2d(
        self, ph: int, pw: int, device: torch.device
    ) -> torch.Tensor:
        """2D separable raised-cosine (Hann) window for seamless spatial tile stitching."""
        wy = torch.sin(
            math.pi * (torch.arange(ph, device=device, dtype=torch.float32) + 0.5) / ph
        ).square()
        wx = torch.sin(
            math.pi * (torch.arange(pw, device=device, dtype=torch.float32) + 0.5) / pw
        ).square()
        w2d = torch.outer(wy, wx).clamp(min=1e-4)
        return w2d.view(1, 1, ph, pw)

    @staticmethod
    def _pad_to_multiple(
        video: torch.Tensor, mult: int = 16
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, t, _, h, w = video.shape
        pad_h = (mult - (h % mult)) % mult
        pad_w = (mult - (w % mult)) % mult
        if pad_h == 0 and pad_w == 0:
            return video, (h, w)
        flat = rearrange(video, "b t c h w -> (b t) c h w")
        padded = F.pad(flat, (0, pad_w, 0, pad_h), mode="reflect")
        return rearrange(padded, "(b t) c h w -> b t c h w", b=b, t=t), (h, w)

    @torch.no_grad()
    def _infer_clip_spatial_tiled(
        self,
        clip: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
        return_rectified: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Runs model on a single temporal burst/clip `[B, T, 3, H, W]`, using 2D Hann
        Overlap-Tile whenever `H > tile_size` or `W > tile_size` (or if `tile_size`
        matches training crop statistics). Supports both 4D static (`[B, C, H, W]`)
        and 5D video (`[B, T, C, H, W]`) model outputs.
        """
        b, tw, c, h, w = clip.shape
        ts = self.tile_size
        ov = self.tile_overlap
        stride = max(1, ts - ov)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        collect_rect = return_rectified and bool(getattr(raw_model.decoder, "has_tilt_branch", True))

        if h <= ts and w <= ts:
            clip_pad, (orig_h, orig_w) = self._pad_to_multiple(clip, mult=16)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                model_out = self.model(clip_pad, precomputed_cn2=precomputed_cn2)
            out = model_out.restored.float()
            restored_crop = out[:, :, :orig_h, :orig_w] if out.ndim == 4 else out[:, :, :, :orig_h, :orig_w]
            rect_crop = model_out.rectified_rgb.float()[:, :, :, :orig_h, :orig_w] if collect_rect else None
            return restored_crop, rect_crop

        global_cn2 = precomputed_cn2
        if global_cn2 is None and hasattr(raw_model, "estimate_global_cn2"):
            global_cn2 = raw_model.estimate_global_cn2(clip)

        y_starts = list(range(0, max(1, h - ts + 1), stride))
        if y_starts[-1] + ts < h:
            y_starts.append(max(0, h - ts))

        x_starts = list(range(0, max(1, w - ts + 1), stride))
        if x_starts[-1] + ts < w:
            x_starts.append(max(0, w - ts))

        hann_2d = self._build_hann_window_2d(ts, ts, self.device)
        acc_canvas: Optional[torch.Tensor] = None
        rect_canvas: Optional[torch.Tensor] = None
        weight_canvas: Optional[torch.Tensor] = None

        for y0 in y_starts:
            for x0 in x_starts:
                ph = min(ts, h - y0)
                pw = min(ts, w - x0)
                tile = clip[:, :, :, y0 : y0 + ph, x0 : x0 + pw]
                tile_pad, (th, tw_px) = self._pad_to_multiple(tile, mult=16)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    model_out = self.model(tile_pad, precomputed_cn2=global_cn2)

                pred_raw = model_out.restored.float()
                is_static = pred_raw.ndim == 4
                pred_tile = (
                    pred_raw[:, :, :th, :tw_px]
                    if is_static
                    else pred_raw[:, :, :, :th, :tw_px]
                )

                if acc_canvas is None:
                    if is_static:
                        acc_canvas = torch.zeros(b, c, h, w, device=self.device, dtype=torch.float32)
                        weight_canvas = torch.zeros(1, 1, h, w, device=self.device, dtype=torch.float32)
                    else:
                        acc_canvas = torch.zeros(b, tw, c, h, w, device=self.device, dtype=torch.float32)
                        weight_canvas = torch.zeros(1, 1, 1, h, w, device=self.device, dtype=torch.float32)
                    if collect_rect:
                        rect_canvas = torch.zeros(b, tw, c, h, w, device=self.device, dtype=torch.float32)

                w_patch = (
                    hann_2d
                    if (ph == ts and pw == ts)
                    else self._build_hann_window_2d(ph, pw, self.device)
                )
                if not is_static:
                    w_patch_5d = w_patch.unsqueeze(0)
                    acc_canvas[:, :, :, y0 : y0 + ph, x0 : x0 + pw] += pred_tile * w_patch_5d
                    weight_canvas[:, :, :, y0 : y0 + ph, x0 : x0 + pw] += w_patch_5d  # type: ignore[index]
                else:
                    acc_canvas[:, :, y0 : y0 + ph, x0 : x0 + pw] += pred_tile * w_patch
                    weight_canvas[:, :, y0 : y0 + ph, x0 : x0 + pw] += w_patch  # type: ignore[index]

                if collect_rect and rect_canvas is not None:
                    rect_tile = model_out.rectified_rgb.float()[:, :, :, :th, :tw_px]
                    w_rect = w_patch.unsqueeze(0)
                    rect_canvas[:, :, :, y0 : y0 + ph, x0 : x0 + pw] += rect_tile * w_rect

        assert acc_canvas is not None and weight_canvas is not None
        denom = weight_canvas.clamp(min=1e-6)
        restored_out = acc_canvas / denom
        rect_out = None
        if collect_rect and rect_canvas is not None:
            rect_denom = denom if denom.ndim == 5 else denom.unsqueeze(1)
            rect_out = rect_canvas / rect_denom
        return restored_out, rect_out

    @torch.no_grad()
    def infer_burst_with_aux(
        self, burst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Restores a static burst `[B, N, 3, H, W]` via single-pass Overlap-Tile and returns
        `(restored_clamped, rectified_rgb_clamped, frf_reference_clamped)` without running
        un-tiled full-resolution network forward passes.
        """
        if burst.ndim == 4:
            burst = burst.unsqueeze(0)
        burst_dev = burst.to(self.device, non_blocking=True)
        restored, rect = self._infer_clip_spatial_tiled(burst_dev, return_rectified=True)
        if restored.ndim == 5:
            restored = restored.mean(dim=1)
        rectified = rect if rect is not None else burst_dev

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        prior = getattr(raw_model, "prior_extractor", None)
        if prior is not None and hasattr(prior, "ref_builder"):
            stab = prior.stabilizer(burst_dev) if hasattr(prior, "stabilizer") else burst_dev
            frf_ref = prior.ref_builder(stab)
        else:
            frf_ref = burst_dev.mean(dim=1)

        return (
            restored.clamp(0.0, 1.0),
            rectified.clamp(0.0, 1.0),
            frf_ref.clamp(0.0, 1.0),
        )

    @torch.no_grad()
    def infer_burst(self, burst: torch.Tensor) -> torch.Tensor:
        """
        Restores a static burst `[N, 3, H, W]` or `[B, N, 3, H, W]` into a single
        clamped image `[B, 3, H, W]` using 2D Hann Overlap-Tile without temporal windowing.
        """
        if burst.ndim == 4:
            burst = burst.unsqueeze(0)
        burst_dev = burst.to(self.device, non_blocking=True)
        restored, _ = self._infer_clip_spatial_tiled(burst_dev, return_rectified=False)
        if restored.ndim == 5:
            restored = restored.mean(dim=1)
        return restored.clamp(0.0, 1.0)

    @torch.no_grad()
    def infer_video(
        self,
        video: torch.Tensor,
        mode: Literal["static_burst", "video", "forensic_snapshot"] = "static_burst",
    ) -> torch.Tensor:
        """
        Args:
            video: Input degraded sequence `[T, 3, H, W]` or `[1, T, 3, H, W]` in `[0, 1]`
            mode: `"static_burst"` returns `[1, 3, H, W]`; `"video"` returns `[T, 3, H, W]`;
                  `"forensic_snapshot"` returns `[1, 3, H, W]`.
        """
        if mode == "static_burst":
            return self.infer_burst(video).cpu()

        if video.ndim == 4:
            video = video.unsqueeze(0)
        _, total_t, c, h, w = video.shape

        win = min(self.window_size, total_t)
        stride = max(1, min(self.temporal_stride, win))

        out_acc = torch.zeros(total_t, c, h, w, dtype=torch.float32)
        weight_acc = torch.zeros(total_t, 1, 1, 1, dtype=torch.float32)

        t_starts = list(range(0, max(1, total_t - win + 1), stride))
        if t_starts[-1] + win < total_t:
            t_starts.append(max(0, total_t - win))

        w_temp_cpu = self._build_temporal_weights(win, self.device).cpu()

        for t0 in t_starts:
            clip = video[:, t0 : t0 + win].to(self.device, non_blocking=True)
            restored_raw, _ = self._infer_clip_spatial_tiled(clip)
            restored_clip = restored_raw.squeeze(0).cpu()
            if restored_clip.ndim == 3:
                restored_clip = restored_clip.unsqueeze(0).expand(win, -1, -1, -1)
            out_acc[t0 : t0 + win] += restored_clip * w_temp_cpu
            weight_acc[t0 : t0 + win] += w_temp_cpu

        restored_video = (out_acc / weight_acc.clamp(min=1e-6)).clamp(0.0, 1.0)

        if mode == "forensic_snapshot":
            gx = restored_video[:, :, :, 1:] - restored_video[:, :, :, :-1]
            gy = restored_video[:, :, 1:, :] - restored_video[:, :, :-1, :]
            sharp_scores = gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))
            best_idx = int(torch.argmax(sharp_scores).item())
            return restored_video[best_idx : best_idx + 1]

        return restored_video
