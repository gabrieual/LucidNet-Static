"""Comprehensive unit tests for LucidNet-Static many-to-one pipeline, dataset cache & inference."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from scripts.prepare_npy_cache import (
    _ZipHandlePool,
    _convert_sample,
    _convert_zip_sample,
    discover_zip_samples,
)
from src.data.atsyn_dataset import ATSynDataset, collate_video_batch, discover_static_samples
from src.engine.evaluator import Evaluator
from src.engine.inference import VideoInferenceEngine
from src.models.blocks.mtcsa import StaticCollapseMTCSA
from src.models.blocks.nafblock import LayerNorm2d
from src.models.classical.frf_reference import FrequencyAwareReferenceBuilder
from src.models.cwd_net import build_cwd_net
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _save_dummy_png(path: Path, val: int, h: int = 48, w: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w, 3), val, dtype=np.uint8)
    Image.fromarray(arr).save(path)


class TestStaticPipeline(unittest.TestCase):
    def test_static_collapse_mtcsa_permutation_invariance_and_t_elasticity(self) -> None:
        torch.manual_seed(0)
        block = StaticCollapseMTCSA(channels=32, spatial_window=8, num_heads=4).eval()

        feat = torch.randn(2, 6, 32, 16, 16)
        prior = torch.rand(2, 6, 1, 16, 16)
        perm = torch.tensor([4, 0, 5, 2, 1, 3])

        with torch.no_grad():
            out_a = block(feat, prior)
            out_b = block(feat[:, perm], prior[:, perm])
            out_t16 = block(torch.randn(1, 16, 32, 24, 24), torch.rand(1, 16, 1, 24, 24))

        self.assertEqual(out_a.shape, (2, 32, 16, 16))
        self.assertEqual(out_t16.shape, (1, 32, 24, 24))
        max_diff = (out_a - out_b).abs().max().item()
        self.assertLess(max_diff, 1e-5)

    def test_no_clamp_inside_static_forward(self) -> None:
        for cfg_name in (
            "configs/ablations/S1_daab_mtcsa_no_wiener.yaml",
            "configs/ablations/S3_lucid_static_full.yaml",
        ):
            cfg = load_config(PROJECT_ROOT / cfg_name)
            cfg["model"]["dims"] = {"c1": 16, "c2": 24, "c3": 32, "deconv_channels": 16}
            cfg["model"]["decoder"]["num_blocks_s2"] = 1
            cfg["model"]["decoder"]["num_blocks_s1"] = 1
            model = build_cwd_net(cfg).eval()
            with torch.no_grad():
                model.decoder.backbone.rgb_head.bias.fill_(2.0)
                out = model(torch.full((1, 4, 3, 64, 64), 0.5))
            self.assertGreater(float(out.restored.max().item()), 1.0)

    def test_layernorm2d_fp16_stability_and_frf_large_t(self) -> None:
        ln = LayerNorm2d(16)
        x = torch.randn(2, 16, 32, 32, dtype=torch.float16) * 1e2
        y = ln(x)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.isfinite(y).all())

        frf = FrequencyAwareReferenceBuilder()
        burst = torch.rand(1, 20, 3, 32, 32)
        ref = frf(burst)
        self.assertEqual(ref.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(ref).all())

    def test_dataset_layouts_npy_cache_and_fallback_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_dir = root / "empty"
            empty_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                ATSynDataset(empty_dir, allow_synthetic_fallback=False)

            raw_split = root / "raw_static"
            # Nested category layout `<root>/{turb,tilt,gt}/<cat>/<sample>`
            for t_idx in range(6):
                _save_dummy_png(raw_split / "turb" / "catA" / "s0" / f"{t_idx:03d}.png", 40 + t_idx)
                _save_dummy_png(raw_split / "tilt" / "catA" / "s0" / f"{t_idx:03d}.png", 80 + t_idx)
            _save_dummy_png(raw_split / "gt" / "catA" / "s0.png", 120)

            records = discover_static_samples(raw_split)
            self.assertEqual(len(records), 1)
            self.assertIsNotNone(records[0].tilt_path)

            npy_split = root / "npy_static"
            sample_id, n_frames, has_tilt, converted = _convert_sample(
                records[0], npy_split, include_tilt=True
            )
            self.assertEqual(sample_id, "catA_s0")
            self.assertEqual(n_frames, 6)
            self.assertTrue(has_tilt)
            self.assertTrue(converted)

            ds = ATSynDataset(
                npy_split,
                num_frames=4,
                crop_size=32,
                is_train=True,
                allow_synthetic_fallback=False,
                static_target=True,
            )
            sample = ds[0]
            self.assertEqual(sample.degraded.shape, (4, 3, 32, 32))
            self.assertEqual(sample.gt.shape, (3, 32, 32))
            self.assertIsNotNone(sample.tilt_only)
            self.assertEqual(sample.tilt_only.shape, (4, 3, 32, 32))

            ds_all = ATSynDataset(
                npy_split,
                num_frames=4,
                crop_size=0,
                is_train=False,
                allow_synthetic_fallback=False,
                static_target=True,
                return_all_frames=True,
            )
            sample_all = ds_all[0]
            self.assertEqual(sample_all.degraded.shape, (6, 3, 48, 48))
            self.assertEqual(sample_all.gt.shape, (3, 48, 48))

            # Regression guard: 4D singleton GT `(1, H, W, 3)` must not raise IndexError
            npy_4d = root / "npy_4d_gt" / "s0"
            npy_4d.mkdir(parents=True)
            np.save(npy_4d / "turb.npy", np.full((6, 48, 48, 3), 50, dtype=np.uint8))
            np.save(npy_4d / "gt.npy", np.full((1, 48, 48, 3), 120, dtype=np.uint8))
            ds_4d = ATSynDataset(
                npy_4d.parent,
                num_frames=4,
                crop_size=32,
                is_train=True,
                allow_synthetic_fallback=False,
                static_target=True,
            )
            s_4d = ds_4d[0]
            self.assertEqual(s_4d.gt.shape, (3, 32, 32))

    def test_s0_s1_no_unused_params_with_auto_frf_anchor(self) -> None:
        for cfg_name in (
            "configs/ablations/S0_flow_only.yaml",
            "configs/ablations/S1_daab_mtcsa_no_wiener.yaml",
        ):
            cfg = load_config(PROJECT_ROOT / cfg_name)
            cfg["model"]["dims"] = {"c1": 16, "c2": 24, "c3": 32, "deconv_channels": 16}
            cfg["model"]["decoder"]["num_blocks_s2"] = 1
            cfg["model"]["decoder"]["num_blocks_s1"] = 1
            model = build_cwd_net(cfg)
            out = model(torch.rand(1, 4, 3, 64, 64))
            out.restored.sum().backward()
            unused = [
                name for name, p in model.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            self.assertEqual(unused, [], f"Found unused params in {cfg_name}: {unused}")

    def test_tiled_inference_and_evaluator(self) -> None:
        cfg = load_config(PROJECT_ROOT / "configs/ablations/S3_lucid_static_full.yaml")
        cfg["model"]["dims"] = {"c1": 16, "c2": 24, "c3": 32, "deconv_channels": 16}
        cfg["model"]["decoder"]["num_blocks_s2"] = 1
        cfg["model"]["decoder"]["num_blocks_s1"] = 1
        model = build_cwd_net(cfg).eval()

        engine = VideoInferenceEngine(
            model=model,
            device=torch.device("cpu"),
            tile_size=32,
            tile_overlap=8,
            use_amp=False,
        )
        burst = torch.rand(5, 3, 50, 50)
        out = engine.infer_video(burst, mode="static_burst")
        self.assertEqual(out.shape, (1, 3, 50, 50))
        self.assertGreaterEqual(float(out.min().item()), 0.0)
        self.assertLessEqual(float(out.max().item()), 1.0)
        out_vid = engine.infer_video(burst, mode="video")
        self.assertEqual(out_vid.shape, (5, 3, 50, 50))

        with tempfile.TemporaryDirectory() as tmp:
            raw_split = Path(tmp) / "test"
            for t_idx in range(4):
                _save_dummy_png(raw_split / "s1" / "turb" / f"{t_idx:03d}.png", 60, h=50, w=50)
                _save_dummy_png(raw_split / "s1" / "tilt" / f"{t_idx:03d}.png", 70, h=50, w=50)
            _save_dummy_png(raw_split / "s1" / "gt" / "gt.png", 80, h=50, w=50)

            ds = ATSynDataset(
                raw_split,
                num_frames=4,
                crop_size=0,
                is_train=False,
                allow_synthetic_fallback=False,
                static_target=True,
            )
            loader = DataLoader(ds, batch_size=1, collate_fn=collate_video_batch)
            evaluator = Evaluator(device=torch.device("cpu"), tiled_inference=engine)
            summary = evaluator.evaluate_loader(model, loader)
            for key in ("PSNR", "SSIM", "CW-SSIM", "FRF_PSNR", "FRF_SSIM", "PSNR_Gain", "SSIM_Gain", "Rectified_PSNR"):
                self.assertIn(key, summary)

    def test_metrics_match_gaussian_ssim_and_clamped_psnr(self) -> None:
        from src.metrics.optical_metrics import compute_cw_ssim, compute_psnr, compute_ssim

        torch.manual_seed(7)
        pred = torch.rand(2, 3, 64, 64) * 1.2 - 0.1  # includes out-of-[0, 1] values
        target = torch.rand(2, 3, 64, 64)

        psnr_val = compute_psnr(pred, target)
        ssim_val = compute_ssim(pred, target)
        cw_ssim_val = compute_cw_ssim(pred, target)

        pred_c = pred.clamp(0.0, 1.0)
        mse = ((pred_c - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1.0e-12)
        expected_psnr = float((10.0 * torch.log10(1.0 / mse)).mean().item())
        self.assertAlmostEqual(psnr_val, expected_psnr, places=6)
        self.assertGreater(ssim_val, 0.0)
        self.assertLess(ssim_val, 1.0)
        self.assertAlmostEqual(compute_cw_ssim(target, target), 1.0, places=5)
        self.assertGreater(cw_ssim_val, 0.0)
        self.assertLess(cw_ssim_val, 1.0)

        # Verify ResUNet image_metrics parity when ResUNet sibling exists in workspace
        resunet_metrics = PROJECT_ROOT.parent / "ResUNet" / "src" / "metrics" / "image_metrics.py"
        if resunet_metrics.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("resunet_image_metrics", resunet_metrics)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertAlmostEqual(psnr_val, mod.compute_psnr(pred, target), places=6)
            self.assertAlmostEqual(ssim_val, mod.compute_ssim(pred, target), places=6)
            self.assertAlmostEqual(cw_ssim_val, mod.compute_cw_ssim(pred, target), places=6)

    def test_zip_and_http_range_conversion_plus_delete_raw(self) -> None:
        def _png_bytes(val: int, h: int = 32, w: int = 32) -> bytes:
            buf = io.BytesIO()
            Image.fromarray(np.full((h, w, 3), val, dtype=np.uint8)).save(buf, format="PNG")
            return buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "atsyn_sample.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for i in range(4):
                    zf.writestr(f"static_new/train_static/turb/catA/s0/{i:03d}.png", _png_bytes(40 + i))
                    zf.writestr(f"static_new/train_static/tilt/catA/s0/{i:03d}.png", _png_bytes(70 + i))
                    zf.writestr(f"static_new/test_static/s1/turb/{i:03d}.png", _png_bytes(50 + i))
                zf.writestr("static_new/train_static/gt/catA/s0.png", _png_bytes(110))
                zf.writestr("static_new/test_static/s1/gt/gt.png", _png_bytes(130))

            zip_bytes = zip_path.read_bytes()
            get_counter = {"data_gets": 0, "redirects": 0}
            counter_lock = threading.Lock()

            class _RangeHandler(BaseHTTPRequestHandler):
                def do_HEAD(self) -> None:
                    self.send_response(200)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(len(zip_bytes)))
                    self.end_headers()

                def do_GET(self) -> None:
                    if self.path == "/redirect/atsyn_sample.zip":
                        with counter_lock:
                            get_counter["redirects"] += 1
                        self.send_response(302)
                        self.send_header("Location", "/cdn/atsyn_sample.zip")
                        self.end_headers()
                        return
                    hdr = self.headers.get("Range")
                    if not hdr or not hdr.startswith("bytes="):
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(zip_bytes)))
                        self.end_headers()
                        self.wfile.write(zip_bytes)
                        return
                    with counter_lock:
                        get_counter["data_gets"] += 1
                    start_s, end_s = hdr[6:].split("-")
                    start = int(start_s)
                    end = int(end_s) if end_s else len(zip_bytes) - 1
                    chunk = zip_bytes[start : end + 1]
                    self.send_response(206)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(zip_bytes)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)

                def log_message(self, format: str, *args) -> None:
                    del format, args

            server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{port}/redirect/atsyn_sample.zip"
                pool = _ZipHandlePool(url)
                try:
                    splits = discover_zip_samples(pool.get_zip())
                    self.assertIn("train", splits)
                    self.assertIn("val", splits)
                    out_base = root / "npy_from_http"
                    with ThreadPoolExecutor(max_workers=4) as ex:
                        futs = [
                            ex.submit(_convert_zip_sample, rec, out_base / "train", pool, include_tilt=True)
                            for rec in splits["train"]
                        ] + [
                            ex.submit(_convert_zip_sample, rec, out_base / "val", pool, include_tilt=False)
                            for rec in splits["val"]
                        ]
                        for fut in futs:
                            fut.result()
                finally:
                    pool.close_all()
            finally:
                server.shutdown()
                server.server_close()

            # 302 redirect resolved once; 14 PNG entries fetched in far fewer coalesced Range GETs
            self.assertEqual(get_counter["redirects"], 1)
            self.assertLessEqual(get_counter["data_gets"], 8)

            train_ds = ATSynDataset(out_base / "train", num_frames=4, crop_size=16, is_train=True)
            val_ds = ATSynDataset(out_base / "val", num_frames=4, crop_size=16, is_train=False)
            self.assertEqual(len(train_ds), 1)
            self.assertEqual(len(val_ds), 1)
            self.assertIsNotNone(train_ds[0].tilt_only)
            self.assertIsNone(val_ds[0].tilt_only)

            # Verify --delete-raw removes source sample directory after writing .npy
            raw_split = root / "raw_del"
            for i in range(4):
                _save_dummy_png(raw_split / "s0" / "turb" / f"{i:03d}.png", 40, h=32, w=32)
            _save_dummy_png(raw_split / "s0" / "gt" / "gt.png", 90, h=32, w=32)
            raw_rec = discover_static_samples(raw_split)[0]
            _convert_sample(raw_rec, root / "npy_del", include_tilt=False, delete_raw=True)
            self.assertFalse((raw_split / "s0").exists())
            self.assertTrue((root / "npy_del" / "s0" / "turb.npy").is_file())


if __name__ == "__main__":
    unittest.main()
