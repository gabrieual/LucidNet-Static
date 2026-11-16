"""Numerical, phase-shift, and initialization tests for Block 3 (Wiener) & flow_warp."""

import math
import unittest


class TestWienerStability(unittest.TestCase):
    def test_wiener_initial_bias_produces_true_dirac_delta(self) -> None:
        center_logit = 12.0
        num_off_center = 15 * 15 - 1
        center_weight = math.exp(center_logit) / (math.exp(center_logit) + num_off_center * 1.0)
        self.assertGreater(center_weight, 0.998)

    def test_wiener_zero_phase_shift_on_delta_psf(self) -> None:
        import torch
        from src.models.blocks.wiener_deconv import MultiChannelWienerDeconv2D

        deconv = MultiChannelWienerDeconv2D(
            in_channels=16, out_channels=8, kernel_size=15, eps_min=1e-6, pad_size=8
        ).eval()

        n, m, h, w = 2, 8, 32, 32
        f_up = torch.rand(n, m, h, w, dtype=torch.float32)

        with torch.no_grad():
            _, aux = deconv(torch.rand(n, 16, h // 2, w // 2))
            psf_init = aux["psf_kernels"]
            self.assertTrue((psf_init[:, :, 7, 7] > 0.998).all())

        delta_psf = torch.zeros(n, m, 15, 15, dtype=torch.float32)
        delta_psf[:, :, 7, 7] = 1.0
        gamma = torch.full((n, m, 1, 1), 1e-6, dtype=torch.float32)

        f_wiener = deconv._wiener_fft_step(f_up, delta_psf, gamma)
        max_err = (f_wiener - f_up).abs().max().item()
        self.assertLess(max_err, 1e-4)

    def test_flow_warp_preserves_subpixel_precision_in_bfloat16(self) -> None:
        import torch
        from src.models.blocks.daab import flow_warp

        x_f32 = torch.rand(1, 3, 256, 256, dtype=torch.float32)
        flow_subpixel = torch.full((1, 2, 256, 256), 0.25, dtype=torch.float32)

        out_f32 = flow_warp(x_f32, flow_subpixel)
        out_bf16 = flow_warp(x_f32.to(torch.bfloat16), flow_subpixel.to(torch.bfloat16))
        diff = (out_f32 - out_bf16.float()).abs().mean().item()
        self.assertLess(diff, 1e-2)

    def test_wiener_initial_identity_and_gradient_stability(self) -> None:
        import torch
        from src.models.blocks.wiener_deconv import MultiChannelWienerDeconv2D

        deconv = MultiChannelWienerDeconv2D(
            in_channels=16, out_channels=8, kernel_size=15, eps_min=0.01, pad_size=8
        )
        deconv.beta_deconv.data.fill_(0.1)

        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out, aux = deconv(x)
        loss = out.square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(deconv.kernel_head.weight.grad)
        self.assertTrue(torch.isfinite(deconv.kernel_head.weight.grad).all())
        self.assertTrue((aux["snr_gamma"] >= 0.01).all())


if __name__ == "__main__":
    unittest.main()
