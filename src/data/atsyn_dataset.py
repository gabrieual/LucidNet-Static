"""Loader for ATSyn-Static and ATSyn-Dynamic Paired Benchmarks (.npy mmap cache, image folders & .mp4)."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.contracts.batch import VideoBatch
from src.data.p2s_simulator import PhysicsTurbulenceSimulator

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class StaticSampleRecord:
    """Immutable descriptor of a paired turbulence sample on disk."""

    sample_id: str
    turb_path: Path
    gt_path: Path
    tilt_path: Optional[Path] = None
    is_npy: bool = False


def _list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def _list_video_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return TF.to_tensor(rgb)  # [3, H, W] float32 in [0, 1]


def _get_video_frame_count(video_path: Path) -> int:
    """Returns total number of frames in an .mp4 video file without decoding all frames."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count > 0:
            return count
    except ImportError:
        pass

    from torchvision.io import read_video

    frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    return int(frames.shape[0])


def _load_video_clip_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Decodes frames from a video file into `[T, 3, H, W]` float32 in `[0, 1]`."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if indices is not None and total > 0:
            frames_list: List[torch.Tensor] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx % total))
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(frame_rgb))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                frames_list.append(tensor)
            cap.release()
            if len(frames_list) == len(indices):
                return torch.stack(frames_list, dim=0)
        else:
            if total > 0 and start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_list = []
            for _ in range(num_frames):
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(frame_rgb))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                frames_list.append(tensor)
            cap.release()
            if len(frames_list) > 0:
                while len(frames_list) < num_frames:
                    frames_list.append(frames_list[-1].clone())
                return torch.stack(frames_list[:num_frames], dim=0)
    except ImportError:
        pass

    from torchvision.io import read_video

    vframes, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    total_v = vframes.shape[0]
    if total_v == 0:
        raise RuntimeError(f"Could not decode any frames from video: {video_path}")
    sel = indices if indices is not None else [(start_frame + i) % total_v for i in range(num_frames)]
    return vframes[[i % total_v for i in sel]].float() / 255.0


def _load_all_video_frames(
    video_path: Path, max_frames: Optional[int] = None
) -> torch.Tensor:
    """Loads all frames (or up to `max_frames`) from an `.mp4` video file for inference."""
    total = _get_video_frame_count(video_path)
    n = min(total, max_frames) if (max_frames is not None and total > 0) else max(1, total)
    return _load_video_clip_frames(video_path, start_frame=0, num_frames=n)


def collate_video_batch(samples: List[VideoBatch]) -> VideoBatch:
    """Custom collate_fn assembling per-clip VideoBatch objects into a batched VideoBatch."""
    degraded = torch.stack([s.degraded for s in samples], dim=0)
    gt = (
        torch.stack([s.gt for s in samples if s.gt is not None], dim=0)
        if all(s.gt is not None for s in samples)
        else None
    )
    tilt_only = (
        torch.stack([s.tilt_only for s in samples if s.tilt_only is not None], dim=0)
        if all(s.tilt_only is not None for s in samples)
        else None
    )
    meta = {
        "clip_names": [
            s.meta.get("clip_name", f"clip_{i}") for i, s in enumerate(samples)
        ],
        "d_r0": [s.meta.get("d_r0", 0.0) for s in samples],
    }
    return VideoBatch(degraded=degraded, gt=gt, tilt_only=tilt_only, meta=meta)


def _find_matching_entry(
    parent_dir: Path, exact_name: str, stem: str
) -> Optional[Path]:
    """Finds a matching image file, video file, or subdirectory in `parent_dir`."""
    if not parent_dir.exists():
        return None
    for cand_name in (exact_name, stem):
        cand = parent_dir / cand_name
        if cand.is_dir():
            if len(_list_image_files(cand)) > 0 or len(_list_video_files(cand)) > 0:
                return cand
        elif cand.is_file() and cand.suffix.lower() in (VIDEO_EXTENSIONS | IMG_EXTENSIONS):
            return cand

    for ext in sorted(VIDEO_EXTENSIONS | IMG_EXTENSIONS):
        for suffix in (ext, ext.upper()):
            candidate = parent_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _resolve_split_root(root_dir: str | Path) -> Path:
    raw_root = Path(root_dir)
    if raw_root.exists():
        return raw_root
    if raw_root.name == "val":
        for alt_name in ("test", "test_static"):
            alt = raw_root.parent / alt_name
            if alt.exists():
                return alt
    return raw_root


def _has_valid_media(entry: Path) -> bool:
    if entry.is_file():
        return entry.suffix.lower() in VIDEO_EXTENSIONS
    if entry.is_dir():
        return len(_list_image_files(entry)) > 0 or len(_list_video_files(entry)) > 0
    return False


def discover_static_samples(root: Path) -> List[StaticSampleRecord]:
    """
    Discovers paired samples across all supported layouts:
      1. Fast mmap `.npy` cache: `<root>/<sample_id>/{turb.npy, [tilt.npy], gt.npy}`
      2. Per-sample directory: `<root>/<sample_id>/{turb/, [tilt/], gt/}`
      3. Modality-split (flat or nested categories):
         `<root>/{turb, [tilt], gt}/<sample_id>` or `<root>/{turb, [tilt], gt}/<cat>/<sample_id>`
    """
    if not root.exists():
        return []

    records: List[StaticSampleRecord] = []

    # Layout 3: Modality-split `<root>/{turb|degraded, [tilt|tilt_only], gt}/...`
    deg_root = (root / "turb") if (root / "turb").exists() else (root / "degraded")
    gt_root = root / "gt"
    tilt_root = (
        (root / "tilt")
        if (root / "tilt").exists()
        else (root / "tilt_only")
    )

    if deg_root.exists() and gt_root.exists():
        for entry in sorted(deg_root.iterdir()):
            if _has_valid_media(entry):
                stem = entry.stem if entry.is_file() else entry.name
                gt_match = _find_matching_entry(gt_root, entry.name, stem)
                if gt_match is not None:
                    tilt_match = (
                        _find_matching_entry(tilt_root, entry.name, stem)
                        if tilt_root.exists()
                        else None
                    )
                    records.append(
                        StaticSampleRecord(
                            sample_id=stem,
                            turb_path=entry,
                            gt_path=gt_match,
                            tilt_path=tilt_match,
                            is_npy=False,
                        )
                    )
            elif entry.is_dir() and (gt_root / entry.name).is_dir():
                # Nested category folder: `<root>/turb/<category>/<sample_id>`
                cat_gt_root = gt_root / entry.name
                cat_tilt_root = (tilt_root / entry.name) if tilt_root.exists() else None
                for sub_entry in sorted(entry.iterdir()):
                    if not _has_valid_media(sub_entry):
                        continue
                    sub_stem = sub_entry.stem if sub_entry.is_file() else sub_entry.name
                    sub_gt = _find_matching_entry(cat_gt_root, sub_entry.name, sub_stem)
                    if sub_gt is not None:
                        sub_tilt = (
                            _find_matching_entry(cat_tilt_root, sub_entry.name, sub_stem)
                            if cat_tilt_root is not None
                            else None
                        )
                        records.append(
                            StaticSampleRecord(
                                sample_id=f"{entry.name}_{sub_stem}",
                                turb_path=sub_entry,
                                gt_path=sub_gt,
                                tilt_path=sub_tilt,
                                is_npy=False,
                            )
                        )
        if records:
            return records

    # Layout 1 (.npy cache) & Layout 2 (per-sample subdirectories)
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        turb_npy = sample_dir / "turb.npy"
        gt_npy = sample_dir / "gt.npy"
        if turb_npy.is_file() and gt_npy.is_file():
            tilt_npy = sample_dir / "tilt.npy"
            records.append(
                StaticSampleRecord(
                    sample_id=sample_dir.name,
                    turb_path=turb_npy,
                    gt_path=gt_npy,
                    tilt_path=tilt_npy if tilt_npy.is_file() else None,
                    is_npy=True,
                )
            )
            continue

        sub_turb = (
            (sample_dir / "turb")
            if (sample_dir / "turb").exists()
            else (sample_dir / "degraded")
        )
        sub_gt = _find_matching_entry(sample_dir, "gt", "gt")
        if _has_valid_media(sub_turb) and sub_gt is not None:
            sub_tilt = (
                _find_matching_entry(sample_dir, "tilt", "tilt")
                or _find_matching_entry(sample_dir, "tilt_only", "tilt_only")
            )
            records.append(
                StaticSampleRecord(
                    sample_id=sample_dir.name,
                    turb_path=sub_turb,
                    gt_path=sub_gt,
                    tilt_path=sub_tilt,
                    is_npy=False,
                )
            )

    return records


class ATSynDataset(Dataset):
    """
    Paired atmospheric turbulence dataset supporting ATSyn-static (`.npy` mmap cache,
    per-sample folders, modality-split with/without categories) and ATSyn-dynamic `.mp4`.

    During static training (`is_train=True`), samples `num_frames` random indices
    WITHOUT sorting to encourage frame-permutation invariance.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 8,
        crop_size: int = 256,
        is_train: bool = True,
        allow_synthetic_fallback: bool = False,
        fallback_length: int = 64,
        static_target: bool = True,
        return_all_frames: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = _resolve_split_root(root_dir)
        self.num_frames = int(num_frames)
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)
        self.allow_synthetic_fallback = bool(allow_synthetic_fallback)
        self.fallback_length = int(fallback_length)
        self.static_target = bool(static_target)
        self.return_all_frames = bool(return_all_frames)

        self.samples: List[StaticSampleRecord] = discover_static_samples(self.root_dir)
        if not self.samples and not self.allow_synthetic_fallback:
            raise FileNotFoundError(
                f"No valid turbulence samples found under '{self.root_dir}' "
                "(and `allow_synthetic_fallback=False`)."
            )
        # Keep `.clips` alias for any diagnostics inspecting len(dataset.clips)
        self.clips = self.samples
        self.simulator = PhysicsTurbulenceSimulator()

    def __len__(self) -> int:
        if self.samples:
            return len(self.samples)
        if self.allow_synthetic_fallback:
            return self.fallback_length
        return 0

    def _select_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            raise ValueError("Sample contains 0 frames.")
        if self.return_all_frames:
            return list(range(total_frames))
        if self.is_train:
            if total_frames >= self.num_frames:
                # Random sampling WITHOUT sorting for permutation invariance
                return random.sample(range(total_frames), self.num_frames)
            return random.choices(range(total_frames), k=self.num_frames)
        return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    def _compute_crop_box(self, h: int, w: int) -> Tuple[int, int, int]:
        cs = self.crop_size
        if cs <= 0 or h < cs or w < cs:
            return 0, 0, 0
        if self.is_train:
            top = int(torch.randint(0, h - cs + 1, (1,)).item())
            left = int(torch.randint(0, w - cs + 1, (1,)).item())
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2
        return top, left, cs

    @staticmethod
    def _slice_gt_mmap(
        gt_mm: np.ndarray,
        indices: List[int],
        top: int,
        left: int,
        cs: int,
    ) -> np.ndarray:
        """Reads 3D [H, W, C] or 4D [T_gt, H, W, C] (including singleton T_gt=1) GT mmap."""
        if gt_mm.ndim == 3:
            src = gt_mm[top : top + cs, left : left + cs, :] if cs > 0 else gt_mm
            return np.array(src, copy=True)

        t_gt = int(gt_mm.shape[0])
        if t_gt == 1:
            src = gt_mm[0, top : top + cs, left : left + cs, :] if cs > 0 else gt_mm[0]
            return np.array(src, copy=True)

        safe_idx = [i % t_gt for i in indices]
        src = gt_mm[safe_idx, top : top + cs, left : left + cs, :] if cs > 0 else gt_mm[safe_idx]
        return np.array(src, copy=True)

    def _load_npy_sample(
        self, rec: StaticSampleRecord
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        turb_mm = np.load(rec.turb_path, mmap_mode="r")
        gt_mm = np.load(rec.gt_path, mmap_mode="r")
        tilt_mm = np.load(rec.tilt_path, mmap_mode="r") if rec.tilt_path is not None else None

        total_t, h, w, _ = turb_mm.shape
        indices = self._select_indices(total_t)
        top, left, cs = self._compute_crop_box(h, w)

        if cs > 0:
            turb_np = np.array(turb_mm[indices, top : top + cs, left : left + cs, :], copy=True)
            tilt_np = (
                np.array(tilt_mm[indices, top : top + cs, left : left + cs, :], copy=True)
                if tilt_mm is not None
                else None
            )
        else:
            turb_np = np.array(turb_mm[indices], copy=True)
            tilt_np = np.array(tilt_mm[indices], copy=True) if tilt_mm is not None else None
        gt_np = self._slice_gt_mmap(gt_mm, indices, top, left, cs)

        turb_t = torch.from_numpy(turb_np).permute(0, 3, 1, 2).float() / 255.0
        if gt_np.ndim == 3:
            gt_img = torch.from_numpy(gt_np).permute(2, 0, 1).float() / 255.0
            gt_t = gt_img if self.static_target else gt_img.unsqueeze(0).repeat(len(indices), 1, 1, 1)
        else:
            gt_seq = torch.from_numpy(gt_np).permute(0, 3, 1, 2).float() / 255.0
            gt_t = gt_seq[0] if self.static_target else gt_seq

        tilt_t = (
            torch.from_numpy(tilt_np).permute(0, 3, 1, 2).float() / 255.0
            if tilt_np is not None
            else None
        )
        return turb_t, gt_t, tilt_t

    @staticmethod
    def _get_entry_length(entry_path: Path) -> int:
        if entry_path.is_file():
            if entry_path.suffix.lower() in VIDEO_EXTENSIONS:
                return _get_video_frame_count(entry_path)
            return 1
        img_files = _list_image_files(entry_path)
        if img_files:
            return len(img_files)
        vid_files = _list_video_files(entry_path)
        if vid_files:
            return _get_video_frame_count(vid_files[0])
        return 0

    @staticmethod
    def _load_entry_frames(
        entry_path: Path, indices: List[int], as_single_image: bool = False
    ) -> torch.Tensor:
        if entry_path.is_file():
            ext = entry_path.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                frames = _load_video_clip_frames(
                    entry_path, start_frame=0, num_frames=len(indices), indices=indices
                )
                return frames[0] if as_single_image else frames
            if ext in IMG_EXTENSIONS:
                img = _load_rgb_tensor(entry_path)
                return img if as_single_image else img.unsqueeze(0).repeat(len(indices), 1, 1, 1)

        img_files = _list_image_files(entry_path)
        if len(img_files) == 1:
            img = _load_rgb_tensor(img_files[0])
            return img if as_single_image else img.unsqueeze(0).repeat(len(indices), 1, 1, 1)
        if len(img_files) > 1:
            if as_single_image:
                return _load_rgb_tensor(img_files[0])
            return torch.stack(
                [_load_rgb_tensor(img_files[i % len(img_files)]) for i in indices],
                dim=0,
            )

        vid_files = _list_video_files(entry_path)
        if vid_files:
            frames = _load_video_clip_frames(
                vid_files[0], start_frame=0, num_frames=len(indices), indices=indices
            )
            return frames[0] if as_single_image else frames
        raise RuntimeError(f"No valid video or image frames found in {entry_path}")

    def _load_media_sample(
        self, rec: StaticSampleRecord
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        deg_len = self._get_entry_length(rec.turb_path)
        gt_len = self._get_entry_length(rec.gt_path)
        total = deg_len if gt_len == 1 else min(deg_len, gt_len)
        indices = self._select_indices(max(1, total))

        deg_seq = self._load_entry_frames(rec.turb_path, indices, as_single_image=False)
        gt_tensor = self._load_entry_frames(
            rec.gt_path, indices, as_single_image=self.static_target
        )
        tilt_seq = (
            self._load_entry_frames(rec.tilt_path, indices, as_single_image=False)
            if rec.tilt_path is not None
            else None
        )

        _, _, h, w = deg_seq.shape
        top, left, cs = self._compute_crop_box(h, w)
        if cs > 0:
            deg_seq = deg_seq[:, :, top : top + cs, left : left + cs]
            gt_tensor = (
                gt_tensor[:, top : top + cs, left : left + cs]
                if gt_tensor.ndim == 3
                else gt_tensor[:, :, top : top + cs, left : left + cs]
            )
            if tilt_seq is not None:
                tilt_seq = tilt_seq[:, :, top : top + cs, left : left + cs]

        return deg_seq, gt_tensor, tilt_seq

    def _generate_procedural_clean_clip(self, idx: int) -> torch.Tensor:
        h = w = max(64, self.crop_size)
        t = self.num_frames
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        r = torch.sqrt(xx.square() + yy.square() + 1e-6)
        theta = torch.atan2(yy, xx)
        freq = 6.0 + (idx % 7) * 2.0
        star = torch.sigmoid(8.0 * torch.cos(freq * theta) * torch.exp(-r * 0.8))
        bars = ((torch.sin(xx * 25.0) * torch.cos(yy * 25.0)) > 0).float() * 0.6
        base = torch.stack(
            [
                0.6 * star + 0.4 * bars,
                0.5 * star + 0.3 * (1.0 - r.clamp(0, 1)),
                0.7 * bars + 0.3 * star,
            ],
            dim=0,
        ).clamp(0.0, 1.0)
        return base.unsqueeze(0).repeat(t, 1, 1, 1)

    def __getitem__(self, idx: int) -> VideoBatch:
        if not self.samples:
            clean_clip = self._generate_procedural_clean_clip(idx)
            degraded, tilt_only, d_r0 = self.simulator(clean_clip)
            gt_out = clean_clip[0] if self.static_target else clean_clip
            return VideoBatch(
                degraded=degraded,
                gt=gt_out,
                tilt_only=tilt_only,
                meta={"clip_name": f"synthetic_p2s_{idx:04d}", "d_r0": d_r0},
            )

        rec = self.samples[idx]
        if rec.is_npy:
            deg_seq, gt_tensor, tilt_seq = self._load_npy_sample(rec)
        else:
            deg_seq, gt_tensor, tilt_seq = self._load_media_sample(rec)

        if self.is_train and torch.rand(1).item() > 0.5:
            deg_seq = torch.flip(deg_seq, dims=[-1])
            gt_tensor = torch.flip(gt_tensor, dims=[-1])
            if tilt_seq is not None:
                tilt_seq = torch.flip(tilt_seq, dims=[-1])

        return VideoBatch(
            degraded=deg_seq,
            gt=gt_tensor,
            tilt_only=tilt_seq,
            meta={"clip_name": rec.sample_id, "d_r0": 0.0},
        )
