#!/usr/bin/env python3
"""Pre-convert ATSyn-static bursts (`turb`, optional `tilt`, `gt`) into uint8 `.npy` mmap caches.

Supports:
  1. Local extracted directory (`--src ... --dst ...`, with optional `--delete-raw`).
  2. Local `.zip` archive (`--from-zip /path/to/static_datum.zip --dst ...`) without extracting PNGs.
  3. Remote Hugging Face / HTTP `.zip` (`--from-zip https://... --dst ...`) via HTTP Range streaming,
     never storing the 127 GB archive or raw PNGs on disk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import zipfile

import numpy as np
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.atsyn_dataset import (  # noqa: E402
    IMG_EXTENSIONS,
    StaticSampleRecord,
    _list_image_files,
    discover_static_samples,
)

_MODALITY_NAMES = {"turb", "degraded", "tilt", "tilt_only", "gt"}


@dataclass(frozen=True)
class ZipSampleRecord:
    """Paired sample located directly inside a `.zip` archive."""

    split_name: str
    sample_id: str
    turb_members: Tuple[str, ...]
    gt_member: str
    tilt_members: Tuple[str, ...] = ()


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file-like object backed by HTTP Range requests."""

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._pos = 0
        self._size = self._probe_size()

    def _probe_size(self) -> int:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    self.url,
                    headers={"Range": "bytes=0-0"},
                    allow_redirects=True,
                    timeout=self.timeout,
                )
                if resp.status_code == 206:
                    cr = resp.headers.get("Content-Range", "")
                    match = re.search(r"/(\d+)$", cr)
                    if match:
                        return int(match.group(1))
                elif resp.status_code == 200:
                    raise RuntimeError(
                        "Servidor HTTP não suporta Range requests (retornou 200 em vez de 206)."
                    )
                resp.raise_for_status()
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                time.sleep(min(8.0, 0.5 * (2**attempt)))

        try:
            head = self.session.head(self.url, allow_redirects=True, timeout=self.timeout)
            head.raise_for_status()
            if "Content-Length" in head.headers:
                return int(head.headers["Content-Length"])
        except Exception as exc:
            last_exc = exc
        raise RuntimeError(f"Não foi possível obter tamanho do ZIP remoto: {self.url}") from last_exc

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self._pos = new_pos
        return self._pos

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        mv = memoryview(buffer).cast("B")
        n = len(mv)
        if n == 0 or self._pos >= self._size:
            return 0
        end = min(self._size - 1, self._pos + n - 1)
        range_hdr = f"bytes={self._pos}-{end}"

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    self.url,
                    headers={"Range": range_hdr},
                    allow_redirects=True,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    raise RuntimeError(
                        "Servidor HTTP ignorou o header Range (retornou 200 em vez de 206)."
                    )
                resp.raise_for_status()
                data = resp.content
                got = len(data)
                mv[:got] = data
                self._pos += got
                return got
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                time.sleep(min(8.0, 0.5 * (2**attempt)))

        raise RuntimeError(f"Falha ao ler range {range_hdr} de {self.url}") from last_exc

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            super().close()


class _ZipHandlePool:
    """Maintains one independent ZipFile per worker thread for lock-free thread safety."""

    def __init__(self, source: str, token: Optional[str] = None, buffer_bytes: int = 2 * 1024 * 1024) -> None:
        self.source = source
        self.token = token
        self.buffer_bytes = buffer_bytes
        self.is_remote = source.startswith(("http://", "https://"))
        self._local = threading.local()
        self._lock = threading.Lock()
        self._resources: List[Tuple[zipfile.ZipFile, Optional[io.BufferedReader]]] = []

    def get_zip(self) -> zipfile.ZipFile:
        zf = getattr(self._local, "zf", None)
        if zf is not None:
            return zf

        buf: Optional[io.BufferedReader] = None
        if self.is_remote:
            raw = HTTPRangeFile(self.source, token=self.token)
            buf = io.BufferedReader(raw, buffer_size=self.buffer_bytes)
            zf = zipfile.ZipFile(buf, mode="r")
        else:
            zf = zipfile.ZipFile(self.source, mode="r")

        self._local.zf = zf
        with self._lock:
            self._resources.append((zf, buf))
        return zf

    def close_all(self) -> None:
        with self._lock:
            for zf, buf in self._resources:
                try:
                    zf.close()
                except Exception:
                    pass
                if buf is not None:
                    try:
                        buf.close()
                    except Exception:
                        pass
            self._resources.clear()


def _normalize_split_name(split_tokens: Tuple[str, ...]) -> str:
    joined = "/".join(split_tokens).lower()
    if "train" in joined:
        return "train"
    if "test" in joined or "val" in joined:
        return "val"
    return split_tokens[-1] if split_tokens else "default"


def discover_zip_samples(zf: zipfile.ZipFile) -> Dict[str, List[ZipSampleRecord]]:
    """
    Discovers paired samples inside a `.zip` archive across both layouts:
      - Modality-split: `<split>/{turb, [tilt], gt}/[<cat>/]<sample_id>/*.png`
      - Per-sample dir: `<split>/<sample_id>/{turb, [tilt], gt}/*.png` (or `gt.png`)
    """
    turb_groups: Dict[Tuple[Tuple[str, ...], str, str], List[str]] = {}
    tilt_groups: Dict[Tuple[Tuple[str, ...], str], List[str]] = {}
    gt_groups: Dict[Tuple[Tuple[str, ...], str], List[str]] = {}

    # Pass 1: find all `turb`/`degraded` image members to establish split roots & layout style
    raw_entries: List[Tuple[Tuple[str, ...], str]] = []
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        parts = tuple(p for p in member.split("/") if p and p != ".")
        if not parts or any(p == "__MACOSX" or p.startswith(".") for p in parts):
            continue
        if Path(parts[-1]).suffix.lower() not in IMG_EXTENSIONS:
            continue
        raw_entries.append((parts, member))

    split_layouts: Dict[Tuple[str, ...], str] = {}
    known_ids: Dict[Tuple[str, ...], set[str]] = {}
    for parts, member in raw_entries:
        for idx, token in enumerate(parts[:-1]):
            low = token.lower()
            if low in ("turb", "degraded"):
                after = parts[idx + 1 :]
                if len(after) == 1:
                    split_tokens = parts[: idx - 1]
                    sample_id = parts[idx - 1] if idx >= 1 else "sample"
                    layout = "per_sample"
                else:
                    split_tokens = parts[:idx]
                    sample_id = "_".join(after[:-1])
                    layout = "modality"
                split_layouts[split_tokens] = layout
                known_ids.setdefault(split_tokens, set()).add(sample_id)
                turb_groups.setdefault((split_tokens, layout, sample_id), []).append(member)
                break

    # Pass 2: associate `tilt` and `gt` members to the discovered split roots
    for parts, member in raw_entries:
        for idx, token in enumerate(parts[:-1]):
            low = token.lower()
            if low in ("tilt", "tilt_only"):
                after = parts[idx + 1 :]
                if len(after) == 1 and idx >= 1 and split_layouts.get(parts[: idx - 1]) == "per_sample":
                    tilt_groups.setdefault((parts[: idx - 1], parts[idx - 1]), []).append(member)
                elif len(after) >= 2 and parts[:idx] in split_layouts:
                    tilt_groups.setdefault((parts[:idx], "_".join(after[:-1])), []).append(member)
                break
            if low == "gt":
                after = parts[idx + 1 :]
                if len(after) == 1 and idx >= 1 and split_layouts.get(parts[: idx - 1]) == "per_sample":
                    gt_groups.setdefault((parts[: idx - 1], parts[idx - 1]), []).append(member)
                elif parts[:idx] in split_layouts:
                    split_tok = parts[:idx]
                    valid_ids = known_ids.get(split_tok, set())
                    dir_key = "_".join(after[:-1]) if len(after) >= 2 else ""
                    stem_key = "_".join((*after[:-1], Path(after[-1]).stem))
                    if dir_key and dir_key in valid_ids:
                        gt_groups.setdefault((split_tok, dir_key), []).append(member)
                    elif stem_key in valid_ids:
                        gt_groups.setdefault((split_tok, stem_key), []).append(member)
                break
        else:
            # Handle `<split>/<sample_id>/gt.png` where parent dir isn't named `gt`
            if len(parts) >= 2 and Path(parts[-1]).stem.lower() == "gt":
                split_tokens = parts[:-2]
                if split_layouts.get(split_tokens) == "per_sample":
                    gt_groups.setdefault((split_tokens, parts[-2]), []).append(member)

    by_split: Dict[str, List[ZipSampleRecord]] = {}
    for (split_tokens, _, sample_id), turb_members in sorted(turb_groups.items()):
        gt_candidates = gt_groups.get((split_tokens, sample_id))
        if not gt_candidates:
            continue
        split_name = _normalize_split_name(split_tokens)
        tilt_candidates = tilt_groups.get((split_tokens, sample_id), [])
        by_split.setdefault(split_name, []).append(
            ZipSampleRecord(
                split_name=split_name,
                sample_id=sample_id,
                turb_members=tuple(sorted(turb_members)),
                gt_member=sorted(gt_candidates)[0],
                tilt_members=tuple(sorted(tilt_candidates)),
            )
        )
    return by_split


def _read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _read_rgb_uint8_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _save_npy_atomic(target: Path, arr: np.ndarray) -> None:
    tmp = target.with_name(f".{target.stem}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, target)


def _cleanup_raw_record(record: StaticSampleRecord) -> None:
    """Safely removes raw files/directories for a converted sample without touching modality roots."""
    for p in (record.turb_path, record.tilt_path, record.gt_path):
        if p is None or not p.exists():
            continue
        if p.is_dir():
            if p.name.lower() not in _MODALITY_NAMES:
                shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    parent = record.turb_path.parent
    if parent.is_dir() and parent.name == record.sample_id:
        try:
            shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            pass


def _convert_sample(
    record: StaticSampleRecord,
    out_root: Path,
    include_tilt: bool = True,
    overwrite: bool = False,
    delete_raw: bool = False,
) -> Tuple[str, int, bool, bool]:
    """Converts one local sample into `turb.npy`, optional `tilt.npy`, and `gt.npy`."""
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
        if delete_raw and not record.is_npy:
            _cleanup_raw_record(record)
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
        _save_npy_atomic(turb_dst, turb_arr)
    if overwrite or not gt_dst.is_file():
        _save_npy_atomic(gt_dst, gt_arr)
    if tilt_arr is not None and (overwrite or not tilt_dst.is_file()):
        _save_npy_atomic(tilt_dst, tilt_arr)

    if delete_raw and not record.is_npy:
        _cleanup_raw_record(record)

    return record.sample_id, int(turb_arr.shape[0]), tilt_arr is not None, True


def _convert_zip_sample(
    record: ZipSampleRecord,
    out_root: Path,
    zip_pool: _ZipHandlePool,
    include_tilt: bool = True,
    overwrite: bool = False,
) -> Tuple[str, int, bool, bool]:
    """Converts one sample directly from a local or HTTP-Range `.zip` into `.npy`."""
    sample_out = out_root / record.sample_id
    turb_dst = sample_out / "turb.npy"
    gt_dst = sample_out / "gt.npy"
    tilt_dst = sample_out / "tilt.npy"

    want_tilt = include_tilt and bool(record.tilt_members)
    already_done = (
        not overwrite
        and turb_dst.is_file()
        and gt_dst.is_file()
        and (not want_tilt or tilt_dst.is_file())
    )
    if already_done:
        arr = np.load(turb_dst, mmap_mode="r")
        return record.sample_id, int(arr.shape[0]), tilt_dst.is_file(), False

    zf = zip_pool.get_zip()
    turb_arr = np.stack(
        [_read_rgb_uint8_bytes(zf.read(m)) for m in record.turb_members], axis=0
    )
    gt_arr = _read_rgb_uint8_bytes(zf.read(record.gt_member))
    tilt_arr: Optional[np.ndarray] = None
    if want_tilt:
        tilt_arr = np.stack(
            [_read_rgb_uint8_bytes(zf.read(m)) for m in record.tilt_members], axis=0
        )

    sample_out.mkdir(parents=True, exist_ok=True)
    if overwrite or not turb_dst.is_file():
        _save_npy_atomic(turb_dst, turb_arr)
    if overwrite or not gt_dst.is_file():
        _save_npy_atomic(gt_dst, gt_arr)
    if tilt_arr is not None and (overwrite or not tilt_dst.is_file()):
        _save_npy_atomic(tilt_dst, tilt_arr)

    return record.sample_id, int(turb_arr.shape[0]), tilt_arr is not None, True


def _run_pool(
    items: list,
    worker_fn: Callable,
    workers: int,
    label: str,
) -> None:
    converted = 0
    skipped = 0
    with_tilt = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(worker_fn, item) for item in items]
        for idx, fut in enumerate(as_completed(futures), start=1):
            _, _, has_tilt, did_convert = fut.result()
            if did_convert:
                converted += 1
            else:
                skipped += 1
            if has_tilt:
                with_tilt += 1
            if idx % 200 == 0 or idx == total:
                print(
                    f"[prepare_npy_cache:{label}] {idx}/{total} "
                    f"(converted={converted}, skipped={skipped}, with_tilt={with_tilt})",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-convert ATSyn-static bursts (turb, optional tilt, gt) to uint8 .npy mmap caches."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--src", type=Path, help="Source local extracted dataset split directory")
    src_group.add_argument(
        "--from-zip",
        type=str,
        help="Local .zip path OR http(s):// remote .zip URL (streamed via HTTP Range).",
    )
    parser.add_argument("--dst", type=Path, required=True, help="Destination .npy cache directory")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="When using --from-zip: 'all' (writes <dst>/{train,val}/...) or specific split name ('train', 'val').",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF Bearer token for remote ZIP.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel I/O conversion threads")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap per split (0 = all)")
    parser.add_argument(
        "--skip-tilt",
        action="store_true",
        help="Skip converting tilt/ frames (saves ~50%% disk/NVMe space when running S1 without L_tilt).",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete each raw sample immediately after its .npy files are written (--src mode).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_tilt = not args.skip_tilt
    args.dst.mkdir(parents=True, exist_ok=True)

    if args.src is not None:
        records = discover_static_samples(args.src)
        if not records:
            raise FileNotFoundError(f"No valid static burst samples found under: {args.src}")
        if args.max_samples > 0:
            records = records[: args.max_samples]
        _run_pool(
            records,
            lambda rec: _convert_sample(
                rec,
                args.dst,
                include_tilt=include_tilt,
                overwrite=args.overwrite,
                delete_raw=args.delete_raw,
            ),
            workers=args.workers,
            label=args.src.name,
        )
        return

    assert args.from_zip is not None
    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    pool = _ZipHandlePool(args.from_zip, token=token)
    try:
        zf = pool.get_zip()
        splits = discover_zip_samples(zf)
        if not splits:
            raise FileNotFoundError(f"No valid static burst samples found inside ZIP: {args.from_zip}")

        req_split = args.split.strip().lower()
        if req_split == "all":
            target_splits = list(splits.items())
        else:
            norm_req = "val" if req_split in ("test", "val", "validation") else req_split
            if norm_req not in splits:
                raise KeyError(
                    f"Split '{args.split}' not found in ZIP. Available: {list(splits.keys())}"
                )
            target_splits = [(norm_req, splits[norm_req])]

        for split_name, recs in target_splits:
            if args.max_samples > 0:
                recs = recs[: args.max_samples]
            out_dir = (args.dst / split_name) if req_split == "all" else args.dst
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[prepare_npy_cache] Streaming split='{split_name}' ({len(recs)} samples) -> {out_dir}",
                flush=True,
            )
            _run_pool(
                recs,
                lambda rec, _od=out_dir: _convert_zip_sample(
                    rec,
                    _od,
                    pool,
                    include_tilt=include_tilt,
                    overwrite=args.overwrite,
                ),
                workers=args.workers,
                label=split_name,
            )
    finally:
        pool.close_all()


if __name__ == "__main__":
    main()
