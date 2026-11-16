#!/usr/bin/env python3
"""Main CLI Training Entrypoint for LucidNet-Static (Many-to-One Static CWD-Net)."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.datasets import ATSynDataset, collate_video_batch
from src.engine.trainer import Trainer
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config, save_config
from src.utils.reproducibility import (
    collect_env_and_git_info,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LucidNet-Static (Static Many-to-One CWD-Net) for Atmospheric Turbulence Mitigation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablations/S3_lucid_static_full.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional explicit path to checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from <exp-dir>/checkpoints/last.pt or W&B if it exists.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Optional explicit experiment output directory.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Optional gradient accumulation steps override.",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Optional dot-list config overrides, e.g., train.total_iters=60000 experiment.use_wandb=true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.opts)
    if args.grad_accum_steps is not None:
        overrides.append(f"train.grad_accum_steps={int(args.grad_accum_steps)}")
    cfg = load_config(args.config, overrides=overrides)

    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    seed_everything(seed=seed, deterministic=deterministic)

    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = exp_cfg.get("name", "lucidnet_static")
        output_root = Path(exp_cfg.get("output_root", "experiments"))
        exp_dir = output_root / f"{timestamp}_{exp_name}_seed{seed}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, exp_dir / "config_resolved.yaml")
    env_info = collect_env_and_git_info(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LucidNet-Static] Experiment directory: {exp_dir}")
    print(f"[LucidNet-Static] Device: {device} ({env_info.get('device_name')})")

    data_cfg = cfg.get("data", {})
    static_target = bool(data_cfg.get("static_target", True))
    allow_fallback = bool(data_cfg.get("allow_synthetic_fallback", False))

    train_dataset = ATSynDataset(
        root_dir=data_cfg.get("train_root", "datasets/ATSyn_static/train"),
        num_frames=int(data_cfg.get("num_frames", 8)),
        crop_size=int(data_cfg.get("crop_size", 256)),
        is_train=True,
        allow_synthetic_fallback=allow_fallback,
        fallback_length=int(data_cfg.get("fallback_length", 64)),
        static_target=static_target,
    )
    val_dataset = ATSynDataset(
        root_dir=data_cfg.get("val_root", "datasets/ATSyn_static/test"),
        num_frames=int(data_cfg.get("num_frames", 8)),
        crop_size=int(data_cfg.get("crop_size", 256)),
        is_train=False,
        allow_synthetic_fallback=allow_fallback,
        fallback_length=max(4, int(data_cfg.get("fallback_length", 64)) // 8),
        static_target=static_target,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_video_batch,
    )

    model = build_cwd_net(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LucidNet-Static] Trainable parameters: {total_params / 1e6:.3f} M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        exp_dir=exp_dir,
        device=device,
    )

    if args.resume is not None:
        print(f"[LucidNet-Static] Resuming from checkpoint: {args.resume}")
        trainer.resume(args.resume)
    elif args.auto_resume:
        trainer.auto_resume()

    try:
        final_metrics = trainer.fit()
        print(f"[LucidNet-Static] Training finished. Final validation metrics: {final_metrics}")
    except Exception:
        trainer.close(exit_code=1)
        raise


if __name__ == "__main__":
    main()
