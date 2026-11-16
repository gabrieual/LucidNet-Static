"""Optical & Turbulence Restoration Evaluation Metrics (PSNR, Gaussian SSIM, CW-SSIM)."""

from functools import lru_cache
from typing import Dict
import torch
import torch.nn.functional as F

from src.losses.cw_ssim import ComplexWaveletSSIMLoss

_CW_SSIM_EVALUATOR = None


def _ensure_4d_float32(tensor: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Converts `[C, H, W]`, `[B, C, H, W]`, or `[B, T, C, H, W]` to clamped float32 4D."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 5:
        b, t, c, h, w = tensor.shape
        tensor = tensor.reshape(b * t, c, h, w)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected 3D, 4D, or 5D tensor for metrics, got shape {tuple(tensor.shape)}"
        )
    return tensor.detach().to(torch.float32).clamp(0.0, data_range)


@lru_cache(maxsize=16)
def _gaussian_window_cpu(
    window_size: int, sigma: float, channels: int
) -> torch.Tensor:
    """Creates a normalized 2D Gaussian kernel shaped `(channels, 1, K, K)`."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    return (
        kernel_2d.view(1, 1, window_size, window_size)
        .repeat(channels, 1, 1, 1)
        .contiguous()
    )


@torch.no_grad()
def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1.0e-12,
) -> float:
    """
    Computes mean batch PSNR (dB) on tensors clamped to `[0, data_range]`.
    Matches ResUNet's `image_metrics.compute_psnr` identically.
    """
    pred_f = _ensure_4d_float32(pred, data_range)
    target_f = _ensure_4d_float32(target, data_range)
    if pred_f.shape != target_f.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_f.shape} vs target {target_f.shape}"
        )

    mse = ((pred_f - target_f) ** 2).mean(dim=(1, 2, 3)).clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return float(psnr.mean().item())


@torch.no_grad()
def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Computes mean batch SSIM with an 11x11 Gaussian window (sigma=1.5) in float32.
    Matches ResUNet's `image_metrics.compute_ssim` identically.
    """
    pred_f = _ensure_4d_float32(pred, data_range)
    target_f = _ensure_4d_float32(target, data_range)
    if pred_f.shape != target_f.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_f.shape} vs target {target_f.shape}"
        )

    channels = pred_f.shape[1]
    pad = window_size // 2
    window = _gaussian_window_cpu(window_size, sigma, channels).to(device=pred_f.device)

    mu1 = F.conv2d(pred_f, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target_f, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred_f * pred_f, window, padding=pad, groups=channels) - mu1_sq
    ).clamp_min(0.0)
    sigma2_sq = (
        F.conv2d(target_f * target_f, window, padding=pad, groups=channels) - mu2_sq
    ).clamp_min(0.0)
    sigma12 = F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1.0e-12)
    return float(ssim_map.mean(dim=(1, 2, 3)).mean().item())


def compute_cw_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes Complex Wavelet SSIM (CW-SSIM) in [0, 1], robust to sub-pixel micro-jitter."""
    global _CW_SSIM_EVALUATOR
    if (
        _CW_SSIM_EVALUATOR is None
        or next(_CW_SSIM_EVALUATOR.buffers()).device != pred.device
    ):
        _CW_SSIM_EVALUATOR = ComplexWaveletSSIMLoss().to(pred.device)
    with torch.no_grad():
        return float(_CW_SSIM_EVALUATOR.compute_similarity(pred, target).item())


class MetricTracker:
    """Running average accumulator for validation and test metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def result(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / max(1, self._counts[k]) for k in self._sums
        }
