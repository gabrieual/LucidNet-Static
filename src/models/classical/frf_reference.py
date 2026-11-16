"""Reference Frame Construction: CDSP Frequency-Aware Reference Frame (FRF) vs Simple Mean."""

import torch
import torch.nn as nn


class FrequencyAwareReferenceBuilder(nn.Module):
    """
    Implements the Frequency-Aware Reference Frame (FRF) from CDSP (Xu et al., 2024)
    and Section 2.1.2 of `arquitetura.md`:
        N_l(t) = sum_{k=1}^T exp( - ||I_l(t) - I_l(k)||_2^2 / (2 * h^2) )
        w_l(t) = exp(sigma * N_l(t)) / sum_{tau=1}^T exp(sigma * N_l(tau))
        I_FRF(x, y) = sum_{t=1}^T w_{x,y}(t) * I_t(x, y)
    Default hyperparameters from specification: h = 0.05, sigma = 0.5.
    """

    def __init__(self, bandwidth_h: float = 0.05, sigma: float = 0.5) -> None:
        super().__init__()
        self.bandwidth_h = bandwidth_h
        self.sigma = sigma

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] in [0, 1]
        Returns:
            frf: [B, C, H, W] sharp tilt-free reference frame
        """
        orig_dtype = video.dtype
        video_f32 = video.float()
        b, t, c, h, w = video_f32.shape
        if t == 1:
            return video[:, 0]

        # Parzen window count N_l(t) in [1, T] computed in O(T*C*H*W) memory
        # so large bursts (e.g., T=50 at 512x512) never trigger O(T^2) OOM.
        denom = 2.0 * (self.bandwidth_h ** 2) + 1e-8
        n_l = torch.empty((b, t, h, w), device=video_f32.device, dtype=torch.float32)
        for k in range(t):
            diff_sq_k = (video_f32 - video_f32[:, k : k + 1]).square().sum(dim=2)
            n_l[:, k] = torch.exp(-diff_sq_k / denom).sum(dim=1)

        # Exponential frequency-aware weighting w_l(t) = Softmax_t(sigma * N_l(t))
        weights = torch.softmax(self.sigma * n_l, dim=1)  # [B, T, H, W]
        frf = (video_f32 * weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
        return frf.to(orig_dtype)


class MeanReferenceBuilder(nn.Module):
    """
    Baseline temporal average reference I_bar = (1/T) * sum(I_t) from proposta.pdf (Eq. 3).
    Kept for Ablation Variant A.
    """

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return video.mean(dim=1)
