"""Sanity-check test: Overfitting LucidNet-Static (S3) on a single synthetic turbulent burst."""

from pathlib import Path
import unittest

from src.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestOverfitSingleBatch(unittest.TestCase):
    def test_overfit_single_batch_with_tilt_supervision(self) -> None:
        import torch
        from src.contracts.batch import VideoBatch
        from src.data.p2s_simulator import PhysicsTurbulenceSimulator
        from src.losses.composite_loss import CWDCompositeLoss
        from src.models.cwd_net import build_cwd_net
        from src.utils.reproducibility import seed_everything

        seed_everything(42, deterministic=True)
        cfg = load_config(PROJECT_ROOT / "configs/ablations/S3_lucid_static_full.yaml")
        cfg["model"]["dims"] = {"c1": 16, "c2": 32, "c3": 48, "deconv_channels": 16}
        cfg["model"]["decoder"]["num_blocks_s1"] = 1
        cfg["model"]["decoder"]["num_blocks_s2"] = 1

        model = build_cwd_net(cfg).train()
        criterion = CWDCompositeLoss(total_iters=20, use_lpips=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        t, c, h, w = 4, 3, 64, 64
        clean_img = torch.rand(c, h, w)
        clean_seq = clean_img.unsqueeze(0).repeat(t, 1, 1, 1)
        simulator = PhysicsTurbulenceSimulator()
        degraded, tilt_only, _ = simulator(clean_seq)

        batch = VideoBatch(
            degraded=degraded.unsqueeze(0),
            gt=clean_img.unsqueeze(0),
            tilt_only=tilt_only.unsqueeze(0),
        )

        initial_loss = None
        final_loss = None
        initial_tilt = None
        final_tilt = None
        for step in range(12):
            optimizer.zero_grad()
            out = model(batch.degraded)
            loss, loss_dict = criterion(out, batch, current_iter=step)
            self.assertTrue(torch.isfinite(loss), f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()

            if step == 0:
                initial_loss = float(loss.item())
                initial_tilt = loss_dict["loss/tilt"]
            final_loss = float(loss.item())
            final_tilt = loss_dict["loss/tilt"]

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertLess(final_loss, initial_loss)
        self.assertGreater(initial_tilt, 0.0)
        self.assertLess(final_tilt, initial_tilt)


if __name__ == "__main__":
    unittest.main()
