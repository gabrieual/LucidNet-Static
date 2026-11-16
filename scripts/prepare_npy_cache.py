#!/usr/bin/env python3
"""Pre-convert ATSyn-static PNG bursts (`turb`, optional `tilt`, `gt`) into uint8 `.npy` mmap caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.atsyn_dataset import (  # noqa: E402
    StaticSampleRecord,
    _list_image_files,
    discover_static_samples,
)


def _read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _convert_sample(
    record: StaticSampleRecord,
    out_root: Path,
    include_tilt: bool = True,
    overwrite: bool = False,
) -> Tuple[str, int, bool, bool]:
    """Converts one sample into `turb.npy`, optional `tilt.npy`, and `gt.npy`."""
    sample_out = out_root / record.sample_id
    turb_dst = sample_out / "turb.npy"
    gt_dst = sample_out / "gt.npy"
    tilt_dst = sample_out / "tilt.npy"

    want_tilt = include_tilt and (record.tilt_path is not None)
    already_done = (
        not overwrite
        and turb_dst.is_file()
        and gt_dst.is_file()
        and (not want_tilt or tilt_dst.is_file())
    )
    if already_done:
        arr = np.load(turb_dst, mmap_mode="r")
        return record.sample_id, int(arr.shape[0]), tilt_dst.is_file(), False

    if record.is_npy:
        turb_arr = np.asarray(np.load(record.turb_path))
        gt_arr = np.asarray(np.load(record.gt_path))
        tilt_arr: Optional[np.ndarray] = (
            np.asarray(np.load(record.tilt_path))
            if (want_tilt and record.tilt_path is not None)
            else None
        )
    else:
        turb_paths = _list_image_files(record.turb_path)
        if not turb_paths:
            raise RuntimeError(f"No image frames found in {record.turb_path}")
        turb_arr = np.stack([_read_rgb_uint8(p) for p in turb_paths], axis=0)

        gt_file = _list_image_files(record.gt_path)[0] if record.gt_path.is_dir() else record.gt_path
        gt_arr = _read_rgb_uint8(gt_file)

        tilt_arr = None
        if want_tilt and record.tilt_path is not None:
            tilt_paths = _list_image_files(record.tilt_path)
            if tilt_paths:
                tilt_arr = np.stack([_read_rgb_uint8(p) for p in tilt_paths], axis=0)

    sample_out.mkdir(parents=True, exist_ok=True)
    if overwrite or not turb_dst.is_file():
        np.save(turb_dst, turb_arr)
    if overwrite or not gt_dst.is_file():
        np.save(gt_dst, gt_arr)
    if tilt_arr is not None and (overwrite or not tilt_dst.is_file()):
        np.save(tilt_dst, tilt_arr)

    return record.sample_id, int(turb_arr.shape[0]), tilt_arr is not None, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-convert ATSyn-static PNG bursts (turb, optional tilt, gt) to uint8 .npy mmap caches."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source dataset split directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination .npy cache directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel I/O conversion threads")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap (0 = all)")
    parser.add_argument(
        "--skip-tilt",
        action="store_true",
        help="Skip converting tilt/ frames (saves ~50%% disk/NVMe space when running S1 without L_tilt).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_static_samples(args.src)
    if not records:
        raise FileNotFoundError(f"No valid static burst samples found under: {args.src}")

    if args.max_samples > 0:
        records = records[: args.max_samples]

    args.dst.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    with_tilt = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                _convert_sample,
                rec,
                args.dst,
                include_tilt=(not args.skip_tilt),
                overwrite=args.overwrite,
            )
            for rec in records
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            _, _, has_tilt, did_convert = fut.result()
            if did_convert:
                converted += 1
            else:
                skipped += 1
            if has_tilt:
                with_tilt += 1
            if idx % 200 == 0 or idx == len(records):
                print(
                    f"[prepare_npy_cache] {idx}/{len(records)} "
                    f"(converted={converted}, skipped={skipped}, with_tilt={with_tilt})"
                )


if __name__ == "__main__":
    main()
