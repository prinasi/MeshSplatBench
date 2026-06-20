"""Tests for evaluation metrics."""

import torch
import pytest

from tribench.core.metrics import compute_psnr


class TestPSNR:
    """Tests for PSNR computation."""

    def test_identical_images(self):
        img = torch.rand(48, 64, 3)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self):
        torch.manual_seed(42)
        img1 = torch.rand(48, 64, 3)
        img2 = torch.rand(48, 64, 3)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float)
        assert psnr > 0
        assert psnr < 100  # PSNR should be finite for random images

    def test_batch_input(self):
        torch.manual_seed(42)
        img1 = torch.rand(2, 48, 64, 3)
        img2 = torch.rand(2, 48, 64, 3)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float)
        assert psnr > 0

    def test_nearly_identical(self):
        img = torch.rand(48, 64, 3)
        noise = img + 1e-6 * torch.rand_like(img)
        psnr = compute_psnr(img, noise)
        assert psnr > 50  # Should be very high for nearly identical images

    def test_zero_images(self):
        img = torch.zeros(48, 64, 3)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")


class TestSSIMLPIPS:
    """Tests for SSIM and LPIPS (require external packages)."""

    def test_ssim_import(self):
        from tribench.core.metrics import compute_ssim
        assert callable(compute_ssim)

    def test_lpips_import(self):
        from tribench.core.metrics import compute_lpips
        assert callable(compute_lpips)

    def test_compute_all_metrics_import(self):
        from tribench.core.metrics import compute_all_metrics
        assert callable(compute_all_metrics)

    def test_lpips_default_matches_triangle_splatting(self):
        import inspect

        from tribench.core.metrics import compute_lpips

        assert inspect.signature(compute_lpips).parameters["net"].default == "vgg"
