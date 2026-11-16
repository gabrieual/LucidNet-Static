"""Training & Validation Engine for CWD-Net with AMP, EMA, Gradient Clipping, TensorBoard & W&B."""

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.contracts.batch import VideoBatch
from src.engine.checkpoint_manager import CheckpointManager, ModelEMA
from src.engine.evaluator import _match_gt_shape
from src.losses.composite_loss import CWDCompositeLoss
from src.metrics.optical_metrics import (
    MetricTracker,
    compute_cw_ssim,
    compute_psnr,
    compute_ssim,
)
from src.models.blocks.nafblock import LayerNorm2d
from src.utils.visualization import save_comparison_grid, save_psf_grid


def _build_adamw_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """
    Excludes bias, LayerNorm2d parameters, 1D/scalar tensors, and learnable gating
    scalars (`beta`, `gamma`, `beta_deconv`, `prior_temp`, `prior_alpha`, `temperature`, `temp`)
    from AdamW weight decay.
    """
    no_decay_names = {
        "beta",
        "gamma",
        "beta_deconv",
        "prior_temp",
        "prior_alpha",
        "temperature",
        "temp",
    }
    norm_param_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, LayerNorm2d)
        for p in m.parameters(recurse=False)
    }

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if (
            id(param) in norm_param_ids
            or leaf == "bias"
            or leaf in no_decay_names
            or param.ndim <= 1
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class Trainer:
    """
    Production-grade Trainer implementing the full training lifecycle specified in `arquitetura.md`:
      - Mixed Precision (`bfloat16` or `float16`) while keeping 2D-FFT blocks in `float32`
      - Global gradient norm clipping (`max_norm=20.0`) to protect deformable offsets & Wiener poles
      - Cosine Annealing learning rate schedule with linear warmup
      - Exponential Moving Average (`ModelEMA`, decay=0.999) for validation & inference
      - Tri-channel logging: `metrics_history.csv`, TensorBoard, and Weights & Biases (`wandb`)
        with automatic run resumption and Google Drive checkpoint persistence.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        exp_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.device = device

        train_cfg = config.get("train", {})
        exp_cfg = config.get("experiment", {})
        self.total_iters: int = int(train_cfg.get("total_iters", 200000))
        self.warmup_iters: int = int(train_cfg.get("warmup_iters", 2000))
        self.lr: float = float(train_cfg.get("lr", 2e-4))
        self.min_lr: float = float(train_cfg.get("min_lr", 1e-6))
        self.weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))
        self.grad_clip_norm: float = float(train_cfg.get("grad_clip_norm", 20.0))
        self.log_interval: int = int(train_cfg.get("log_interval", 50))
        self.val_interval: int = int(train_cfg.get("val_interval", 2000))
        self.save_interval: int = int(train_cfg.get("save_interval", 5000))
        self.grad_accum_steps: int = max(1, int(train_cfg.get("grad_accum_steps", 1)))

        # Precision & AMP setup
        amp_dtype_str = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
        self.use_amp = bool(train_cfg.get("use_amp", True)) and (device.type == "cuda")
        if (
            amp_dtype_str == "bfloat16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Optimizer & Cosine Warmup Scheduler (weight_decay=0 on norms, bias & scalar gains)
        param_groups = _build_adamw_param_groups(self.model, self.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.99),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # EMA & Checkpointing (can point to local SSD, Google Drive, and W&B cloud)
        ema_decay = float(train_cfg.get("ema_decay", 0.999))
        self.ema = ModelEMA(self.model, decay=ema_decay)

        # Loss Function (2-Phase Curriculum)
        loss_cfg = dict(config.get("loss", {}))
        loss_cfg.setdefault("total_iters", self.total_iters)
        self.criterion = CWDCompositeLoss(**loss_cfg).to(device)

        # Directories & Logging
        self.vis_dir = self.exp_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.exp_dir / "metrics_history.csv"
        self.global_step = 0
        self.start_epoch = 0

        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=str(self.exp_dir / "tb_logs"))
        except ImportError:
            pass

        # Weights & Biases (W&B) Integration
        self.use_wandb = bool(exp_cfg.get("use_wandb", False))
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                exp_name = str(exp_cfg.get("name", "cwdnet"))
                seed = int(exp_cfg.get("seed", 42))
                run_id = exp_cfg.get("wandb_run_id") or f"{exp_name}_seed{seed}"
                self.wandb_run = wandb.init(
                    project=str(exp_cfg.get("wandb_project", "lucidnet-static")),
                    entity=exp_cfg.get("wandb_entity"),
                    group=str(exp_cfg.get("wandb_group", "lucidnet-static-ablations")),
                    name=f"{exp_name}_seed{seed}",
                    id=str(run_id),
                    resume="allow",
                    config=config,
                    dir=str(self.exp_dir),
                )
            except Exception as exc:
                print(f"[Trainer] Warning: Could not initialize W&B ({exc}). Continuing without W&B.")
                self.use_wandb = False

        self.ckpt_manager = CheckpointManager(
            self.exp_dir / "checkpoints",
            keep_last_k=int(train_cfg.get("keep_last_k_checkpoints", 3)),
            wandb_run=self.wandb_run,
        )

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_iters:
            return float(step + 1) / float(max(1, self.warmup_iters))
        progress = float(step - self.warmup_iters) / float(
            max(1, self.total_iters - self.warmup_iters)
        )
        progress = min(max(0.0, progress), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = self.min_lr / self.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    def resume(self, checkpoint_path: str | Path) -> None:
        ckpt = self.ckpt_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            restore_rng=True,
            map_location=self.device,
        )
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))

    def auto_resume(self) -> bool:
        """
        Attempts to resume training from:
          1. Local / Google Drive `<exp_dir>/checkpoints/last.pt`
          2. Weights & Biases cloud (`wandb.restore('checkpoints/last.pt')`) if not on local disk.
        Returns True if a checkpoint was found and loaded.
        """
        local_last = self.exp_dir / "checkpoints" / "last.pt"
        if local_last.exists():
            print(f"[Trainer] Auto-resuming from local/Drive checkpoint: {local_last}")
            self.resume(local_last)
            return True

        wb_last = self.ckpt_manager.try_restore_from_wandb("last.pt")
        if wb_last is not None and wb_last.exists():
            print(f"[Trainer] Auto-resuming from W&B cloud checkpoint: {wb_last}")
            self.resume(wb_last)
            return True

        return False

    def _append_csv(self, row: Dict[str, Any]) -> None:
        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        eval_model = self.ema.ema_model
        eval_model.eval()
        tracker = MetricTracker()

        comp_path = self.vis_dir / f"step_{self.global_step:07d}_comparison.png"
        psf_path = self.vis_dir / f"step_{self.global_step:07d}_psf_kernels.png"

        for batch_idx, batch in enumerate(self.val_loader):
            batch = batch.to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = eval_model(batch.degraded)

            if batch.gt is not None:
                pred_restored = output.restored.clamp(0.0, 1.0)
                pred_rectified = output.rectified_rgb.clamp(0.0, 1.0)
                frf_clamped = output.frf_reference.clamp(0.0, 1.0)

                gt_recon = _match_gt_shape(batch.gt, pred_restored)
                gt_rect = _match_gt_shape(batch.gt, pred_rectified)
                gt_frf = _match_gt_shape(batch.gt, frf_clamped)

                psnr = compute_psnr(pred_restored, gt_recon)
                ssim = compute_ssim(pred_restored, gt_recon)
                cw_ssim = compute_cw_ssim(pred_restored, gt_recon)
                rect_psnr = compute_psnr(pred_rectified, gt_rect)
                frf_psnr = compute_psnr(frf_clamped, gt_frf)
                frf_ssim = compute_ssim(frf_clamped, gt_frf)
                frf_cw_ssim = compute_cw_ssim(frf_clamped, gt_frf)

                tracker.update(
                    {
                        "val/psnr": psnr,
                        "val/ssim": ssim,
                        "val/cw_ssim": cw_ssim,
                        "val/rectified_psnr": rect_psnr,
                        "val/frf_psnr": frf_psnr,
                        "val/frf_ssim": frf_ssim,
                        "val/frf_cw_ssim": frf_cw_ssim,
                        "val/psnr_gain": psnr - frf_psnr,
                        "val/ssim_gain": ssim - frf_ssim,
                        "val/cw_ssim_gain": cw_ssim - frf_cw_ssim,
                    },
                    n=batch.degraded.shape[0],
                )

            if batch_idx == 0:
                save_comparison_grid(
                    degraded=batch.degraded,
                    rectified=output.rectified_rgb,
                    restored=output.restored,
                    gt=batch.gt,
                    save_path=comp_path,
                    frame_idx=batch.degraded.shape[1] // 2,
                )
                save_psf_grid(
                    psf_kernels=output.aux.get("psf_kernels"),
                    save_path=psf_path,
                )

        results = tracker.result()
        if self.tb_writer is not None:
            for k, v in results.items():
                self.tb_writer.add_scalar(k, v, self.global_step)

        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb  # type: ignore

                wb_payload: Dict[str, Any] = dict(results)
                if comp_path.exists():
                    wb_payload["visuals/comparison_grid"] = wandb.Image(
                        str(comp_path),
                        caption=f"Step {self.global_step}: Degraded | Rectified | Restored | GT",
                    )
                if psf_path.exists():
                    wb_payload["visuals/wiener_psf_kernels"] = wandb.Image(
                        str(psf_path),
                        caption=f"Step {self.global_step}: Learned 15x15 Wiener PSF Kernels",
                    )
                wandb.log(wb_payload, step=self.global_step)
            except Exception:
                pass

        return results

    def close(self, exit_code: int = 0) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.finish(exit_code=exit_code)
            except Exception:
                pass
            self.wandb_run = None

    def fit(self) -> Dict[str, float]:
        self.model.train()
        epoch = self.start_epoch
        last_val_metrics: Dict[str, float] = {}
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        accum_count = 0

        while self.global_step < self.total_iters:
            for batch in self.train_loader:
                if self.global_step >= self.total_iters:
                    break

                batch: VideoBatch = batch.to(self.device)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = self.model(batch.degraded)
                    loss, loss_dict = self.criterion(
                        output, batch, current_iter=self.global_step
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at step {self.global_step}: {loss_dict}"
                    )

                self.scaler.scale(loss / float(self.grad_accum_steps)).backward()
                accum_count += 1
                if accum_count < self.grad_accum_steps:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.ema.update(self.model)
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0

                self.global_step += 1

                if self.global_step % self.log_interval == 0 or self.global_step == 1:
                    lr_curr = self.optimizer.param_groups[0]["lr"]
                    beta_deconv = float(output.aux.get("beta_deconv", 0.0))
                    snr_gamma_t = output.aux.get("snr_gamma")
                    gamma_mean = (
                        float(snr_gamma_t.mean().item())
                        if snr_gamma_t is not None
                        else 0.0
                    )
                    elapsed = time.time() - start_time
                    log_row = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "lr": round(lr_curr, 8),
                        "grad_norm": round(float(grad_norm), 4),
                        "beta_deconv": round(beta_deconv, 5),
                        "gamma_mean": round(gamma_mean, 5),
                        "elapsed_sec": round(elapsed, 1),
                        **{k: round(v, 5) for k, v in loss_dict.items()},
                    }
                    self._append_csv(log_row)
                    if self.tb_writer is not None:
                        for k, v in loss_dict.items():
                            self.tb_writer.add_scalar(k, v, self.global_step)
                        self.tb_writer.add_scalar("train/lr", lr_curr, self.global_step)
                        self.tb_writer.add_scalar(
                            "train/grad_norm", float(grad_norm), self.global_step
                        )
                        self.tb_writer.add_scalar(
                            "train/beta_deconv", beta_deconv, self.global_step
                        )
                        self.tb_writer.add_scalar(
                            "train/gamma_mean", gamma_mean, self.global_step
                        )

                    if self.use_wandb and self.wandb_run is not None:
                        try:
                            import wandb  # type: ignore

                            wandb.log(
                                {
                                    **loss_dict,
                                    "train/lr": lr_curr,
                                    "train/grad_norm": float(grad_norm),
                                    "train/beta_deconv": beta_deconv,
                                    "train/gamma_mean": gamma_mean,
                                    "train/epoch": epoch,
                                },
                                step=self.global_step,
                            )
                        except Exception:
                            pass

                if self.val_loader is not None and (
                    self.global_step % self.val_interval == 0
                    or self.global_step == self.total_iters
                ):
                    last_val_metrics = self.validate(epoch)
                    self.model.train()
                    self.ckpt_manager.save(
                        epoch=epoch,
                        global_step=self.global_step,
                        model=self.model,
                        ema=self.ema,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        config=self.config,
                        metrics=last_val_metrics,
                        is_periodic_step=(self.global_step % self.save_interval == 0),
                    )
                elif self.global_step % self.save_interval == 0:
                    self.ckpt_manager.save(
                        epoch=epoch,
                        global_step=self.global_step,
                        model=self.model,
                        ema=self.ema,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        config=self.config,
                        metrics=last_val_metrics,
                        is_periodic_step=True,
                    )

            epoch += 1

        self.close(exit_code=0)
        return last_val_metrics
