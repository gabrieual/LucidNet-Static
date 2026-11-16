"""Block 4: Twin Decoder (Tilt Rectifier Branch A with CAB + NAFNet Restoration Branch B)."""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.interfaces import BaseTwinDecoder
from src.models.registry import DECODER_REGISTRY
from src.models.blocks.nafblock import ChannelAttentionBlock, NAFBlock
from src.models.blocks.daab import flow_warp
from src.utils.tensor_ops import rearrange


@DECODER_REGISTRY.register("twin_naf_decoder")
class TwinNAFDecoder(BaseTwinDecoder):
    """
    Implements Section 2.5 of `arquitetura.md`:
      - Branch A (Geometric Tilt Rectifier T_hat_t^{-1}):
        Receives deep fused feature F_{lucky, t} (scale 1/4, c3=128) and shallow encoder
        features f_t^2 (scale 1/2, c2=96) and f_t^1 (scale 1/1, c1=64).
        Uses 2 stages of `Upsample + ChannelAttentionBlock (CAB)` to predict the dense
        inverse tilt field T_hat_t^{-1} at scale 1/1 and its downsampled version at scale 1/2.
        Rectifies f_t^2, f_t^1, and the RGB frame I_t BEFORE skip-connection fusion.
      - Branch B (Residual NAFNet Restoration):
        * Scale 1/2: Concatenates [F_{deconv, t} (64 ch), f_tilde_t^2 (96 ch)], reduces to
          96 channels via Conv1x1, and passes through 4 NAFBlocks.
        * Scale 1/1: Upsamples to HxW via PixelShuffle, concatenates with f_tilde_t^1 (64 ch),
          reduces to 64 channels via Conv1x1, and passes through 6 NAFBlocks.
        * Strict Residual Output: J_hat_t = I_{rect, t} + Conv3x3(NAFBlocks_out).
    """

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.deconv_channels = deconv_channels
        self.max_tilt_displacement = max_tilt_displacement

        # --- Branch A: 2 Stages of Upsample + ChannelAttentionBlock (CAB) (§2.5.1) ---
        self.tilt_up_s2 = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1),
            ChannelAttentionBlock(c2),
            ChannelAttentionBlock(c2),
        )
        self.tilt_head_s2 = nn.Conv2d(c2, 2, kernel_size=3, padding=1)

        self.tilt_up_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1),
            ChannelAttentionBlock(c1),
            ChannelAttentionBlock(c1),
        )
        self.tilt_head_s1 = nn.Conv2d(c1, 2, kernel_size=3, padding=1)

        # Initialize final tilt layers to zero for identity start
        nn.init.zeros_(self.tilt_head_s2.weight)
        nn.init.zeros_(self.tilt_head_s2.bias)
        nn.init.zeros_(self.tilt_head_s1.weight)
        nn.init.zeros_(self.tilt_head_s1.bias)

        # --- Branch B Stage 1 (Scale 1/2: 4 NAFBlocks on 96 channels) (§2.5.2) ---
        self.fuse_s2 = nn.Conv2d(deconv_channels + c2, c2, kernel_size=1)
        self.decoder_s2 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks_s2)])

        # Upsample 1/2 -> 1/1
        self.up_s2_to_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

        # --- Branch B Stage 2 (Scale 1/1: 6 NAFBlocks on 64 channels) (§2.5.2) ---
        self.fuse_s1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.decoder_s1 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks_s1)])
        self.rgb_head = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    has_tilt_branch: bool = True

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky_seq: torch.Tensor,
        f_deconv_seq: torch.Tensor,
        sharpness_prior: Optional[torch.Tensor] = None,
        frf_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sharpness_prior, frf_static
        b, t, _, _, _ = frames.shape

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        f1_flat = rearrange(f1_seq, "b t c h w -> (b t) c h w")
        f2_flat = rearrange(f2_seq, "b t c h w -> (b t) c h w")
        f_lucky_flat = rearrange(f_lucky_seq, "b t c h w -> (b t) c h w")
        f_deconv_flat = rearrange(f_deconv_seq, "b t c h w -> (b t) c h w")

        # --- Branch A: 2-stage Upsample + CAB Inverse Tilt Predictor ---
        t_feat_s2 = self.tilt_cab_s2(
            torch.cat([self.tilt_up_s2(f_lucky_flat), f2_flat], dim=1)
        )
        tilt_s2 = torch.tanh(self.tilt_head_s2(t_feat_s2)) * (
            self.max_tilt_displacement * 0.5
        )

        t_feat_s1 = self.tilt_cab_s1(
            torch.cat([self.tilt_up_s1(t_feat_s2), f1_flat], dim=1)
        )
        tilt_s1 = (
            torch.tanh(self.tilt_head_s1(t_feat_s1)) * self.max_tilt_displacement
        )

        # Rectify shallow skip connections and RGB frame using inverse tilt field
        f2_rect = flow_warp(f2_flat, tilt_s2)
        f1_rect = flow_warp(f1_flat, tilt_s1)
        rectified_rgb_flat = flow_warp(frames_flat, tilt_s1)

        # --- Branch B: Multi-scale NAFNet restoration (4 blocks at 1/2 + 6 blocks at 1/1) ---
        x_s2 = self.fuse_s2(torch.cat([f_deconv_flat, f2_rect], dim=1))
        x_s2 = self.decoder_s2(x_s2)

        x_s1_up = self.up_s2_to_s1(x_s2)
        x_s1 = self.fuse_s1(torch.cat([x_s1_up, f1_rect], dim=1))
        x_s1 = self.decoder_s1(x_s1)

        residual_rgb = self.rgb_head(x_s1)
        restored_flat = torch.clamp(rectified_rgb_flat + residual_rgb, 0.0, 1.0)

        restored = rearrange(restored_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        rectified_rgb = rearrange(
            rectified_rgb_flat, "(b t) c h w -> b t c h w", b=b, t=t
        )
        inv_tilt_field = rearrange(tilt_s1, "(b t) c h w -> b t c h w", b=b, t=t)
        return restored, rectified_rgb, inv_tilt_field


@DECODER_REGISTRY.register("single_naf_decoder")
class SingleStageNAFDecoder(BaseTwinDecoder):
    """
    Ablation counterpart for Block 4 (Variants A, B, C without Twin Tilt Branch):
    Directly fuses unrectified shallow skip connections f1 and f2 with f_deconv
    using the same NAFBlock backbone (4 blocks at 1/2 + 6 blocks at 1/1).
    """

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.deconv_channels = deconv_channels

        self.fuse_s2 = nn.Conv2d(deconv_channels + c2, c2, kernel_size=1)
        self.decoder_s2 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks_s2)])
        self.up_s2_to_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse_s1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.decoder_s1 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks_s1)])
        self.rgb_head = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    has_tilt_branch: bool = False

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky_seq: torch.Tensor,
        f_deconv_seq: torch.Tensor,
        sharpness_prior: Optional[torch.Tensor] = None,
        frf_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del f_lucky_seq, sharpness_prior, frf_static
        b, t, _, h, w = frames.shape

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        f1_flat = rearrange(f1_seq, "b t c h w -> (b t) c h w")
        f2_flat = rearrange(f2_seq, "b t c h w -> (b t) c h w")
        f_deconv_flat = rearrange(f_deconv_seq, "b t c h w -> (b t) c h w")

        x_s2 = self.fuse_s2(torch.cat([f_deconv_flat, f2_flat], dim=1))
        x_s2 = self.decoder_s2(x_s2)

        x_s1_up = self.up_s2_to_s1(x_s2)
        x_s1 = self.fuse_s1(torch.cat([x_s1_up, f1_flat], dim=1))
        x_s1 = self.decoder_s1(x_s1)

        residual_rgb = self.rgb_head(x_s1)
        restored_flat = torch.clamp(frames_flat + residual_rgb, 0.0, 1.0)

        restored = rearrange(restored_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        zero_tilt = torch.zeros(
            b, t, 2, h, w, device=frames.device, dtype=frames.dtype
        )
        return restored, frames, zero_tilt


class _StaticLuckySkipFusion(nn.Module):
    """
    Collapses multi-frame skip tensors `[B, T, C, H, W] -> [B, C, H, W]` using
    sharpness priority P_t upsampled bilinearly, keeping weights convex (sum=1).
    """

    def __init__(self, init_temp: float = 5.0) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(init_temp))

    def _compute_weights(
        self,
        sharpness_prior: Optional[torch.Tensor],
        b: int,
        t: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if sharpness_prior is None:
            return torch.full((b, t, 1, h, w), 1.0 / max(t, 1), device=device, dtype=dtype)
        p_flat = rearrange(sharpness_prior.float(), "b t c h w -> (b t) c h w")
        if p_flat.shape[-2:] != (h, w):
            p_flat = F.interpolate(
                p_flat, size=(h, w), mode="bilinear", align_corners=False
            )
        p_seq = rearrange(p_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        return torch.softmax(self.temp * p_seq, dim=1).to(dtype)

    def forward(
        self,
        seq: torch.Tensor,
        sharpness_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, t, _, h, w = seq.shape
        weights = self._compute_weights(
            sharpness_prior, b, t, h, w, seq.device, seq.dtype
        )
        return (seq * weights).sum(dim=1)


class _StaticNAFBackbone(nn.Module):
    """Shared Branch B NAFNet restoration backbone (runs 1x after T -> 1 fusion)."""

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
    ) -> None:
        super().__init__()
        self.fuse_s2 = nn.Conv2d(deconv_channels + c2, c2, kernel_size=1)
        self.decoder_s2 = nn.Sequential(*[NAFBlock(c2) for _ in range(num_blocks_s2)])
        self.up_s2_to_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse_s1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.decoder_s1 = nn.Sequential(*[NAFBlock(c1) for _ in range(num_blocks_s1)])
        self.rgb_head = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def forward(
        self,
        anchor_rgb: torch.Tensor,
        f_deconv: torch.Tensor,
        f2_fused: torch.Tensor,
        f1_fused: torch.Tensor,
    ) -> torch.Tensor:
        x_s2 = self.fuse_s2(torch.cat([f_deconv, f2_fused], dim=1))
        x_s2 = self.decoder_s2(x_s2)

        x_s1_up = self.up_s2_to_s1(x_s2)
        x_s1 = self.fuse_s1(torch.cat([x_s1_up, f1_fused], dim=1))
        x_s1 = self.decoder_s1(x_s1)

        # Unclamped residual output during training (clamped only in eval/inference)
        return anchor_rgb + self.rgb_head(x_s1)


def _select_static_anchor(
    anchor_mode: str,
    default_mode: str,
    rgb_fused: torch.Tensor,
    frf_static: Optional[torch.Tensor],
) -> torch.Tensor:
    mode = default_mode if anchor_mode == "auto" else anchor_mode
    if mode == "frf" and frf_static is not None:
        return frf_static
    return rgb_fused


@DECODER_REGISTRY.register("static_twin_naf_decoder")
class StaticTwinNAFDecoder(BaseTwinDecoder):
    """
    Static Hybrid Twin Decoder (S2 & S3):
      - Branch A (per-frame in B*T): predicts inverse tilt field T_hat_t^{-1} at 1/2 and 1/1
        using the single fused `f_lucky` (upsampled 1x) + per-frame shallow skips `f2, f1`,
        and rectifies `f2, f1, frames` before skip fusion. Supervised via L_tilt + L_TV.
      - Lucky Skip Fusion (T -> 1): collapses rectified `f2_rect, f1_rect, rectified_rgb`
        via P_t-weighted convex combination.
      - Branch B (1x): runs 4+6 NAFBlocks once on the fused representation.
    """

    has_tilt_branch: bool = True

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
        anchor_mode: Literal["auto", "fused_rgb", "frf"] = "auto",
    ) -> None:
        super().__init__()
        self.max_tilt_displacement = max_tilt_displacement
        self.anchor_mode = anchor_mode

        self.tilt_up_s2 = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1),
            ChannelAttentionBlock(c2),
            ChannelAttentionBlock(c2),
        )
        self.tilt_head_s2 = nn.Conv2d(c2, 2, kernel_size=3, padding=1)

        self.tilt_up_s1 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tilt_cab_s1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1),
            ChannelAttentionBlock(c1),
            ChannelAttentionBlock(c1),
        )
        self.tilt_head_s1 = nn.Conv2d(c1, 2, kernel_size=3, padding=1)

        nn.init.zeros_(self.tilt_head_s2.weight)
        nn.init.zeros_(self.tilt_head_s2.bias)
        nn.init.zeros_(self.tilt_head_s1.weight)
        nn.init.zeros_(self.tilt_head_s1.bias)

        self.skip_fusion_s2 = _StaticLuckySkipFusion()
        self.skip_fusion_s1 = _StaticLuckySkipFusion()
        self.skip_fusion_rgb = _StaticLuckySkipFusion()
        self.backbone = _StaticNAFBackbone(
            c1=c1,
            c2=c2,
            deconv_channels=deconv_channels,
            num_blocks_s2=num_blocks_s2,
            num_blocks_s1=num_blocks_s1,
        )

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky: torch.Tensor,
        f_deconv: torch.Tensor,
        sharpness_prior: Optional[torch.Tensor] = None,
        frf_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, _, _ = frames.shape
        if f_lucky.ndim == 5:
            f_lucky = f_lucky.mean(dim=1)
        if f_deconv.ndim == 5:
            f_deconv = f_deconv.mean(dim=1)

        # Upsample fused latent 1x, then expand across T for per-frame tilt prediction
        lucky_up_s2 = self.tilt_up_s2(f_lucky)
        lucky_up_s2_flat = rearrange(
            lucky_up_s2.unsqueeze(1).expand(-1, t, -1, -1, -1),
            "b t c h w -> (b t) c h w",
        )

        frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
        f1_flat = rearrange(f1_seq, "b t c h w -> (b t) c h w")
        f2_flat = rearrange(f2_seq, "b t c h w -> (b t) c h w")

        t_feat_s2 = self.tilt_cab_s2(torch.cat([lucky_up_s2_flat, f2_flat], dim=1))
        tilt_s2 = torch.tanh(self.tilt_head_s2(t_feat_s2)) * (
            self.max_tilt_displacement * 0.5
        )

        t_feat_s1 = self.tilt_cab_s1(
            torch.cat([self.tilt_up_s1(t_feat_s2), f1_flat], dim=1)
        )
        tilt_s1 = torch.tanh(self.tilt_head_s1(t_feat_s1)) * self.max_tilt_displacement

        f2_rect = rearrange(
            flow_warp(f2_flat, tilt_s2), "(b t) c h w -> b t c h w", b=b, t=t
        )
        f1_rect = rearrange(
            flow_warp(f1_flat, tilt_s1), "(b t) c h w -> b t c h w", b=b, t=t
        )
        rectified_rgb = rearrange(
            flow_warp(frames_flat, tilt_s1), "(b t) c h w -> b t c h w", b=b, t=t
        )
        inv_tilt_field = rearrange(tilt_s1, "(b t) c h w -> b t c h w", b=b, t=t)

        f2_fused = self.skip_fusion_s2(f2_rect, sharpness_prior)
        f1_fused = self.skip_fusion_s1(f1_rect, sharpness_prior)
        rgb_fused = self.skip_fusion_rgb(rectified_rgb, sharpness_prior)

        anchor_rgb = _select_static_anchor(
            self.anchor_mode, "fused_rgb", rgb_fused, frf_static
        )
        restored = self.backbone(anchor_rgb, f_deconv, f2_fused, f1_fused)
        return restored, rectified_rgb, inv_tilt_field


@DECODER_REGISTRY.register("static_single_naf_decoder")
class StaticSingleNAFDecoder(BaseTwinDecoder):
    """
    Static Single-Stage Decoder (S0-flow & S1, without Branch A / L_tilt):
    Collapses shallow skips via P_t-weighted fusion and runs NAFNet Branch B 1x
    anchored on `frf_static` (default) or `rgb_fused`.
    """

    has_tilt_branch: bool = False

    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        c3: int = 128,
        deconv_channels: int = 64,
        num_blocks_s2: int = 4,
        num_blocks_s1: int = 6,
        max_tilt_displacement: float = 16.0,
        anchor_mode: Literal["auto", "fused_rgb", "frf"] = "auto",
    ) -> None:
        super().__init__()
        del c3, max_tilt_displacement
        self.anchor_mode = anchor_mode
        self.skip_fusion_s2 = _StaticLuckySkipFusion()
        self.skip_fusion_s1 = _StaticLuckySkipFusion()
        # Only instantiate rgb skip fusion when fused_rgb is explicitly requested;
        # in "auto"/"frf" mode frf_static is used directly, avoiding an unused parameter.
        self.skip_fusion_rgb: Optional[_StaticLuckySkipFusion] = (
            _StaticLuckySkipFusion() if anchor_mode == "fused_rgb" else None
        )
        self.backbone = _StaticNAFBackbone(
            c1=c1,
            c2=c2,
            deconv_channels=deconv_channels,
            num_blocks_s2=num_blocks_s2,
            num_blocks_s1=num_blocks_s1,
        )

    def forward(
        self,
        frames: torch.Tensor,
        f1_seq: torch.Tensor,
        f2_seq: torch.Tensor,
        f_lucky: torch.Tensor,
        f_deconv: torch.Tensor,
        sharpness_prior: Optional[torch.Tensor] = None,
        frf_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del f_lucky
        b, t, _, h, w = frames.shape
        if f_deconv.ndim == 5:
            f_deconv = f_deconv.mean(dim=1)

        f2_fused = self.skip_fusion_s2(f2_seq, sharpness_prior)
        f1_fused = self.skip_fusion_s1(f1_seq, sharpness_prior)

        mode = "frf" if self.anchor_mode == "auto" else self.anchor_mode
        if mode == "frf" and frf_static is not None:
            anchor_rgb = frf_static
        elif self.skip_fusion_rgb is not None:
            anchor_rgb = self.skip_fusion_rgb(frames, sharpness_prior)
        else:
            anchor_rgb = frames.mean(dim=1)

        restored = self.backbone(anchor_rgb, f_deconv, f2_fused, f1_fused)
        zero_tilt = torch.zeros(
            b, t, 2, h, w, device=frames.device, dtype=frames.dtype
        )
        return restored, frames, zero_tilt
