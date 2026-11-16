#!/usr/bin/env python3
"""Evaluation CLI Script for LucidNet-Static (including Zero-Shot Wiener & N-Frame Elasticity)."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.datasets import ATSynDataset, collate_video_batch
from src.engine.evaluator import Evaluator
from src.engine.inference import VideoInferenceEngine
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LucidNet-Static checkpoint on static benchmarks."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Uses EMA weights by default.",
    )
    parser.add_argument(
        "--zero-wiener",
        action="store_true",
        help="Diagnostic S3-ZeroWiener: forces beta_deconv = 0 at inference time.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Override number of frames N per burst (for Stage 2 N=8,16,32,50 elasticity).",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate using all frames available in each burst.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Run full-resolution 2D Hann Overlap-Tile inference (crop_size=0, tile_size from config).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary as JSON.",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Optional dot-list config overrides.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    seed_everything(
        int(cfg.get("experiment", {}).get("seed", 42)), deterministic=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cwd_net(cfg).to(device).eval()

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"[Evaluate] Loaded EMA weights from {args.checkpoint}")

    if args.zero_wiener and hasattr(model.deconv, "beta_deconv"):
        model.deconv.beta_deconv.zero_()
        print("[Evaluate] Diagnostic S3-ZeroWiener active: forced beta_deconv = 0.0")

    data_cfg = cfg.get("data", {})
    inf_cfg = cfg.get("inference", {})

    num_frames = (
        int(args.num_frames)
        if args.num_frames is not None
        else int(data_cfg.get("num_frames", 8))
    )
    crop_size = 0 if args.tiled else int(data_cfg.get("crop_size", 256))

    test_dataset = ATSynDataset(
        root_dir=data_cfg.get("test_root", "datasets/ATSyn_static/test"),
        num_frames=num_frames,
        crop_size=crop_size,
        is_train=False,
        allow_synthetic_fallback=bool(data_cfg.get("allow_synthetic_fallback", False)),
        fallback_length=16,
        static_target=bool(data_cfg.get("static_target", True)),
        return_all_frames=bool(args.all_frames or data_cfg.get("return_all_frames", False)),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_video_batch,
    )

    tiled_engine = None
    if args.tiled:
        tiled_engine = VideoInferenceEngine(
            model=model,
            device=device,
            tile_size=int(inf_cfg.get("tile_size", 256)),
            tile_overlap=int(inf_cfg.get("tile_overlap", 64)),
            use_amp=bool(cfg.get("train", {}).get("use_amp", True)),
        )

    evaluator = Evaluator(device=device, tiled_inference=tiled_engine)
    summary = evaluator.evaluate_loader(model, test_loader)
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
