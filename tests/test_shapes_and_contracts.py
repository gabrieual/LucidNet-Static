"""Unit tests for YAML configurations, Schema Validation, AST Signature Alignment & Liskov Contracts."""

import ast
from pathlib import Path
from typing import Dict, Set, Tuple
import unittest

from src.utils.config_loader import load_config
from src.utils.config_schema import (
    VALID_ALIGNMENT_PARAMS,
    VALID_DECODER_PARAMS,
    VALID_DECONV_PARAMS,
    VALID_ENCODER_PARAMS,
    VALID_FUSION_PARAMS,
    VALID_LOSS_PARAMS,
    VALID_PRIOR_PARAMS,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_CONFIGS = [
    "configs/ablations/S0_flow_only.yaml",
    "configs/ablations/S1_daab_mtcsa_no_wiener.yaml",
    "configs/ablations/S2_twin_decoder_tilt.yaml",
    "configs/ablations/S3_lucid_static_full.yaml",
]


def _extract_registered_classes_and_init_args(
    py_path: Path,
) -> Dict[str, Tuple[str, Set[str]]]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    results: Dict[str, Tuple[str, Set[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            reg_names = []
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                ):
                    for arg in dec.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            reg_names.append(arg.value)
            if not reg_names:
                continue
            init_params: Set[str] = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for a in item.args.args:
                        if a.arg != "self":
                            init_params.add(a.arg)
            for rname in reg_names:
                results[rname] = (node.name, init_params)
    return results


class TestShapesAndContracts(unittest.TestCase):
    def test_yaml_inheritance_and_strict_schema(self) -> None:
        for cfg_rel_path in ABLATION_CONFIGS:
            with self.subTest(cfg=cfg_rel_path):
                cfg = load_config(PROJECT_ROOT / cfg_rel_path, validate=True)
                self.assertEqual(cfg["model"]["dims"]["c1"], 64)
                self.assertEqual(cfg["model"]["dims"]["c2"], 96)
                self.assertEqual(cfg["model"]["dims"]["c3"], 128)
                self.assertEqual(cfg["model"]["dims"]["deconv_channels"], 64)
                self.assertEqual(cfg["model"]["decoder"]["num_blocks_s2"], 4)
                self.assertEqual(cfg["model"]["decoder"]["num_blocks_s1"], 6)
                self.assertEqual(cfg["loss"]["lambda_char"], 0.8)
                self.assertEqual(cfg["loss"]["lambda_fft"], 0.05)
                self.assertFalse(cfg["data"]["allow_synthetic_fallback"])
                self.assertTrue(cfg["data"]["static_target"])

    def test_schema_rejects_unknown_registry_or_invalid_constructor_args(self) -> None:
        cfg = load_config(PROJECT_ROOT / "configs/ablations/S3_lucid_static_full.yaml")

        bad_reg = dict(cfg)
        bad_reg["model"] = dict(cfg["model"])
        bad_reg["model"]["fusion"] = {"name": "non_existent_fusion"}
        with self.assertRaises(KeyError):
            validate_config(bad_reg)

        bad_param = load_config(PROJECT_ROOT / "configs/ablations/S3_lucid_static_full.yaml")
        bad_param["model"]["prior_extractor"]["frf_temperature"] = 25.0
        with self.assertRaises(TypeError):
            validate_config(bad_param)

    def test_ast_registry_names_and_constructor_signatures_match_yamls(self) -> None:
        files = [
            PROJECT_ROOT / "src/models/classical/prior_extractor.py",
            PROJECT_ROOT / "src/models/blocks/encoder.py",
            PROJECT_ROOT / "src/models/blocks/daab.py",
            PROJECT_ROOT / "src/models/blocks/mtcsa.py",
            PROJECT_ROOT / "src/models/blocks/wiener_deconv.py",
            PROJECT_ROOT / "src/models/blocks/twin_decoder.py",
        ]
        all_registered: Dict[str, Tuple[str, Set[str]]] = {}
        for f in files:
            all_registered.update(_extract_registered_classes_and_init_args(f))

        for cfg_rel in ABLATION_CONFIGS:
            cfg = load_config(PROJECT_ROOT / cfg_rel, validate=True)
            m = cfg["model"]
            for section in [
                "prior_extractor",
                "encoder",
                "alignment",
                "fusion",
                "deconv",
                "decoder",
            ]:
                sub = dict(m[section])
                name = sub.pop("name")
                self.assertIn(name, all_registered)
                cls_name, init_args = all_registered[name]
                unknown = set(sub.keys()) - init_args
                self.assertFalse(
                    unknown,
                    f"{cfg_rel}: model.{section} passes {unknown} not in {cls_name}.__init__",
                )

        for name, allowed in VALID_PRIOR_PARAMS.items():
            self.assertEqual(allowed, all_registered[name][1])
        self.assertEqual(VALID_ENCODER_PARAMS, all_registered["shallow_multiscale"][1])
        for name, allowed in VALID_ALIGNMENT_PARAMS.items():
            self.assertEqual(allowed, all_registered[name][1])
        for name, allowed in VALID_FUSION_PARAMS.items():
            self.assertEqual(allowed, all_registered[name][1])
        self.assertEqual(VALID_DECONV_PARAMS, all_registered["wiener_fft_multichannel"][1])
        for name, allowed in VALID_DECODER_PARAMS.items():
            self.assertEqual(allowed, all_registered[name][1])

    def test_composite_loss_targets_and_tv_field_via_ast(self) -> None:
        loss_path = PROJECT_ROOT / "src/losses/composite_loss.py"
        src_text = loss_path.read_text(encoding="utf-8")
        tree = ast.parse(src_text)

        init_args: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CWDCompositeLoss":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_args = {a.arg for a in item.args.args if a.arg != "self"}

        self.assertEqual(init_args, VALID_LOSS_PARAMS)
        self.assertIn("flow_warp(tilt_only_flat, inv_tilt_flat)", src_text)
        self.assertIn("self.charbonnier(rectified_tilt, j_gt_seq)", src_text)
        self.assertIn("self.tv_loss(output.inverse_tilt_field)", src_text)
        self.assertNotIn("self.tv_loss(j_hat)", src_text)

    def test_liskov_substitution_all_ablation_variants(self) -> None:
        import torch
        from src.models.cwd_net import build_cwd_net

        for cfg_rel_path in ABLATION_CONFIGS:
            with self.subTest(cfg=cfg_rel_path):
                cfg = load_config(PROJECT_ROOT / cfg_rel_path)
                cfg["model"]["dims"] = {"c1": 16, "c2": 24, "c3": 32, "deconv_channels": 16}
                cfg["model"]["decoder"]["num_blocks_s2"] = 1
                cfg["model"]["decoder"]["num_blocks_s1"] = 1
                model = build_cwd_net(cfg).eval()

                b, t, c, h, w = 1, 4, 3, 64, 64
                x = torch.rand(b, t, c, h, w)
                with torch.no_grad():
                    out = model(x)

                self.assertEqual(out.restored.shape, (b, c, h, w))
                self.assertEqual(out.rectified_rgb.shape, (b, t, c, h, w))
                self.assertEqual(out.inverse_tilt_field.shape, (b, t, 2, h, w))
                self.assertEqual(out.frf_reference.shape, (b, c, h, w))
                self.assertTrue(torch.isfinite(out.restored).all())


if __name__ == "__main__":
    unittest.main()
