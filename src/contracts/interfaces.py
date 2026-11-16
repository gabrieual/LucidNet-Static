"""Abstract Base Classes enforcing Liskov Substitution & Interface Segregation across all blocks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.contracts.batch import ModelOutput, PriorOutput


class BasePriorExtractor(nn.Module, ABC):
    """Contract for Block 0: Classical Physics-Statistical Prior Extractor."""

    @abstractmethod
    def estimate_global_cn2(self, video: torch.Tensor) -> torch.Tensor:
        """
        Computes the global turbulence strength Cn^2 [B] across the full-resolution frame
        before spatial tiling in production inference.
        """

    @abstractmethod
    def forward(
        self,
        video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> PriorOutput:
        """
        Args:
            video: Degraded input sequence [B, T, C, H, W]
            precomputed_cn2: Optional precomputed scene-wide Cn^2 [B] (used in spatial tiling)
        Returns:
            PriorOutput containing stabilized frames, FRF reference, optional motion mask, and P_t prior.
        """


class BaseMultiScaleEncoder(nn.Module, ABC):
    """Contract for Block 1.1: Shared shallow multi-scale feature extractor."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Frame tensor [N, C, H, W]
        Returns:
            f1: [N, C1, H, W]       (Scale 1/1, C1=64)
            f2: [N, C2, H//2, W//2] (Scale 1/2, C2=96)
            f3: [N, C3, H//4, W//4] (Scale 1/4, C3=128)
        """


class BaseAlignmentBlock(nn.Module, ABC):
    """Contract for Block 1.2-1.4: Frame-to-Reference Registration."""

    @abstractmethod
    def forward(
        self,
        f3_seq: torch.Tensor,
        ref_f3: torch.Tensor,
        frames_low: torch.Tensor,
        ref_low: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            f3_seq: [B, T, C3, H//4, W//4] deep features of input frames
            ref_f3: [B, C3, H//4, W//4] (static) or [B, T, C3, H//4, W//4] reference features
            frames_low: [B, T, 3, H//4, W//4] downsampled RGB frames for optical flow
            ref_low: Optional [B, 3, H//4, W//4] downsampled static reference RGB
        Returns:
            aligned_seq: [B, T, C3, H//4, W//4] aligned & integrated embeddings
        """


class BaseTemporalFusionBlock(nn.Module, ABC):
    """Contract for Block 2: Temporal-Channel Lucky Fusion (MTCSA / Static Collapse)."""

    @abstractmethod
    def forward(
        self,
        aligned_seq: torch.Tensor,
        sharpness_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            aligned_seq: [B, T, C3, H//4, W//4]
            sharpness_prior: [B, T, 1, H//4, W//4] classical priority map P_t
        Returns:
            fused: [B, C3, H//4, W//4] (static T->1) or [B, T, C3, H//4, W//4] (video)
        """


class BaseDeconvBlock(nn.Module, ABC):
    """Contract for Block 3: Blind Wiener Deconvolution in 2D-FFT (or Identity for ablation)."""

    @abstractmethod
    def forward(
        self, f_lucky: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            f_lucky: [B, C3, H//4, W//4] (static) or [B*T, C3, H//4, W//4] (video)
        Returns:
            f_deconv: [N, C_deconv, H//2, W//2] upsampled and deconvolved features (C_deconv=64)
            aux_dict: Dictionary with estimated PSFs 'psf_kernels', 'snr_gamma', 'beta_deconv'
        """


class BaseTwinDecoder(nn.Module, ABC):
    """Contract for Block 4: Static/Twin Decoder (Tilt Rectifier Branch A + NAFNet Branch B)."""

    has_tilt_branch: bool = True

    @abstractmethod
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
        """
        Args:
            frames: [B, T, 3, H, W] stabilized input frames
            f1_seq: [B, T, C1, H, W] scale 1/1 encoder skip connections
            f2_seq: [B, T, C2, H//2, W//2] scale 1/2 encoder skip connections
            f_lucky: [B, C3, H//4, W//4] (static) or [B, T, C3, H//4, W//4]
            f_deconv: [B, C_deconv, H//2, W//2] (static) or [B, T, C_deconv, H//2, W//2]
            sharpness_prior: Optional [B, T, 1, H//4, W//4] P_t map for lucky skip fusion
            frf_static: Optional [B, 3, H, W] static FRF reference image
        Returns:
            restored: [B, 3, H, W] (static) or [B, T, 3, H, W] (video)
            rectified_rgb: [B, T, 3, H, W] tilt-rectified RGB frames
            inverse_tilt_field: [B, T, 2, H, W] predicted inverse tilt fields
        """


BaseStaticDecoder = BaseTwinDecoder


class InferenceModel(nn.Module, ABC):
    """Interface Segregation Principle: Inference only requires degraded video input."""

    @abstractmethod
    def forward(
        self,
        degraded_video: torch.Tensor,
        precomputed_cn2: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Run forward restoration pass on [B, T, C, H, W] degraded input."""


class TrainingLossWrapper(nn.Module, ABC):
    """Interface Segregation Principle: Losses receive ModelOutput and ground-truth batch."""

    @abstractmethod
    def forward(
        self,
        output: ModelOutput,
        batch: Any,
        current_iter: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total training loss and scalar diagnostic dictionary."""
