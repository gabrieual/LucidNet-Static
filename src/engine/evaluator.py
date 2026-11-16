"""Validation & Benchmark Evaluator with CSV, LaTeX Ablation Tables, and Tiled Full-Res Eval."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.inference import VideoInferenceEngine
from src.metrics.optical_metrics import (
    MetricTracker,
    compute_cw_ssim,
    compute_psnr,
    compute_ssim,
)


def _match_gt_shape(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Aligns `gt` ([B, C, H, W] or [B, T, C, H, W]) to match `ref` ndim."""
    if gt.ndim == ref.ndim:
        return gt
    if ref.ndim == 4 and gt.ndim == 5:
        return gt[:, 0]
    if ref.ndim == 5 and gt.ndim == 4:
        return gt.unsqueeze(1).expand(-1, ref.shape[1], -1, -1, -1)
    return gt


class Evaluator:
    """
    Runs quantitative evaluation across paired turbulence test sets (ATSyn-static, etc.)
    and exports results to CSV and LaTeX tables.
    Computes FRF_PSNR, FRF_SSIM, Rectified_PSNR, PSNR, SSIM, CW-SSIM, PSNR_Gain, and SSIM_Gain.
    """

    def __init__(
        self,
        device: torch.device,
        tiled_inference: Optional[VideoInferenceEngine] = None,
    ) -> None:
        self.device = device
        self.tiled_inference = tiled_inference

    @torch.no_grad()
    def evaluate_loader(
        self, model: nn.Module, loader: DataLoader
    ) -> Dict[str, float]:
        model = model.to(self.device).eval()
        tracker = MetricTracker()

        for batch in loader:
            batch = batch.to(self.device)

            if self.tiled_inference is not None:
                pred_restored, pred_rectified, pred_frf = (
                    self.tiled_inference.infer_burst_with_aux(batch.degraded)
                )
            else:
                output = model(batch.degraded)
                pred_restored = output.restored.clamp(0.0, 1.0)
                pred_rectified = output.rectified_rgb.clamp(0.0, 1.0)
                pred_frf = output.frf_reference.clamp(0.0, 1.0)

            if batch.gt is not None:
                b = batch.degraded.shape[0]
                gt_recon = _match_gt_shape(batch.gt, pred_restored)
                gt_rect = _match_gt_shape(batch.gt, pred_rectified)
                gt_frf = _match_gt_shape(batch.gt, pred_frf)

                psnr = compute_psnr(pred_restored, gt_recon)
                ssim = compute_ssim(pred_restored, gt_recon)
                cw_ssim = compute_cw_ssim(pred_restored, gt_recon)
                frf_psnr = compute_psnr(pred_frf, gt_frf)
                frf_ssim = compute_ssim(pred_frf, gt_frf)
                rect_psnr = compute_psnr(pred_rectified, gt_rect)

                tracker.update(
                    {
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "CW-SSIM": cw_ssim,
                        "FRF_PSNR": frf_psnr,
                        "FRF_SSIM": frf_ssim,
                        "PSNR_Gain": psnr - frf_psnr,
                        "SSIM_Gain": ssim - frf_ssim,
                        "Rectified_PSNR": rect_psnr,
                    },
                    n=b,
                )

        return tracker.result()

    @staticmethod
    def export_ablation_tables(
        results_by_variant: Dict[str, Dict[str, float]],
        csv_path: str | Path,
        latex_path: Optional[str | Path] = None,
    ) -> str:
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        metric_cols: List[str] = [
            "FRF_PSNR",
            "FRF_SSIM",
            "Rectified_PSNR",
            "PSNR",
            "PSNR_Gain",
            "SSIM",
            "SSIM_Gain",
            "CW-SSIM",
        ]

        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant"] + metric_cols)
            for var_name, m in results_by_variant.items():
                writer.writerow(
                    [var_name] + [f"{m.get(col, 0.0):.4f}" for col in metric_cols]
                )

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study of LucidNet-Static Variants}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Variant & FRF PSNR & Rectified PSNR & PSNR (dB) $\uparrow$ & Gain $\uparrow$ & SSIM $\uparrow$ & CW-SSIM $\uparrow$ \\",
            r"\midrule",
        ]
        for var_name, m in results_by_variant.items():
            safe_name = var_name.replace("_", r"\_")
            row_vals = " & ".join(
                [
                    f"{m.get('FRF_PSNR', 0.0):.2f}",
                    f"{m.get('Rectified_PSNR', 0.0):.2f}",
                    f"{m.get('PSNR', 0.0):.2f}",
                    f"{m.get('PSNR_Gain', 0.0):+.2f}",
                    f"{m.get('SSIM', 0.0):.4f}",
                    f"{m.get('CW-SSIM', 0.0):.4f}",
                ]
            )
            lines.append(f"{safe_name} & {row_vals} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        latex_str = "\n".join(lines) + "\n"

        if latex_path is not None:
            tex_out = Path(latex_path)
            tex_out.parent.mkdir(parents=True, exist_ok=True)
            tex_out.write_text(latex_str, encoding="utf-8")

        return latex_str
