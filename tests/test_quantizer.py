"""Tests for TurboQuant MSE and inner-product quantizers."""

from __future__ import annotations

import math

import pytest
import torch

from turboquant.quantizer import OutlierAwareQuantizer, TurboQuantMSE, TurboQuantProd


@pytest.fixture
def dim():
    return 128


@pytest.fixture
def n_vectors():
    return 256


@pytest.fixture
def random_vectors(dim, n_vectors):
    torch.manual_seed(42)
    return torch.randn(n_vectors, dim)


class TestTurboQuantMSE:
    def test_round_trip_shape(self, dim, random_vectors):
        quant = TurboQuantMSE(dim, bit_width=2)
        recon = quant.round_trip(random_vectors)
        assert recon.shape == random_vectors.shape

    def test_quantize_indices_in_range(self, dim, random_vectors):
        for b in [1, 2, 3, 4]:
            quant = TurboQuantMSE(dim, bit_width=b)
            q = quant.quantize(random_vectors)
            assert q.indices.min() >= 0
            assert q.indices.max() < (1 << b)

    def test_norms_positive(self, dim, random_vectors):
        quant = TurboQuantMSE(dim, bit_width=2)
        q = quant.quantize(random_vectors)
        assert (q.norms > 0).all()

    def test_mse_decreases_with_bits(self, dim, random_vectors):
        mses = []
        for b in [1, 2, 3, 4]:
            quant = TurboQuantMSE(dim, bit_width=b)
            mse = quant.mse(random_vectors).item()
            mses.append(mse)
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], f"MSE should decrease: {mses}"

    def test_mse_bounded(self, dim):
        torch.manual_seed(0)
        x = torch.randn(512, dim)
        for b in [2, 3, 4]:
            quant = TurboQuantMSE(dim, bit_width=b)
            mse = quant.mse(x).item()
            per_vec_norm_sq = (x**2).sum(dim=-1).mean().item()
            relative_mse = mse / per_vec_norm_sq
            upper_bound = (math.sqrt(3) * math.pi / 2) * (1.0 / (4**b))
            assert relative_mse < upper_bound * 3, (
                f"b={b}: relative_mse={relative_mse:.6f} > 3x bound={upper_bound:.6f}"
            )

    def test_zero_vector(self, dim):
        quant = TurboQuantMSE(dim, bit_width=2)
        x = torch.zeros(1, dim)
        recon = quant.round_trip(x)
        assert torch.allclose(recon, x, atol=1e-5)

    def test_batch_dimensions(self, dim):
        quant = TurboQuantMSE(dim, bit_width=2)
        x = torch.randn(4, 8, dim)
        q = quant.quantize(x)
        assert q.indices.shape == (4, 8, dim)
        assert q.norms.shape == (4, 8)
        recon = quant.dequantize(q)
        assert recon.shape == x.shape

    def test_storage_bytes(self, dim):
        for b in [1, 2, 3, 4]:
            quant = TurboQuantMSE(dim, bit_width=b)
            bytes_per_vec = quant.storage_bytes_per_vector()
            expected_index_bytes = (dim * b + 7) // 8
            assert bytes_per_vec == expected_index_bytes + 4


class TestTurboQuantProd:
    def test_requires_min_2_bits(self, dim):
        with pytest.raises(ValueError):
            TurboQuantProd(dim, bit_width=1)

    def test_round_trip_shape(self, dim, random_vectors):
        quant = TurboQuantProd(dim, bit_width=3)
        recon = quant.round_trip(random_vectors)
        assert recon.shape == random_vectors.shape

    def test_unbiased_inner_product(self, dim):
        torch.manual_seed(42)
        n = 1000
        x = torch.randn(n, dim)
        y = torch.randn(n, dim)

        quant = TurboQuantProd(dim, bit_width=3)
        result = quant.inner_product_error(x, y)

        true_ip_std = (x * y).sum(dim=-1).std().item()
        assert abs(result["bias"]) < 0.1 * true_ip_std, (
            f"Bias too large: {result['bias']:.6f} vs ip_std={true_ip_std:.4f}"
        )

    def test_variance_decreases_with_bits(self, dim):
        torch.manual_seed(42)
        x = torch.randn(256, dim)
        y = torch.randn(256, dim)

        variances = []
        for b in [2, 3, 4]:
            quant = TurboQuantProd(dim, bit_width=b)
            result = quant.inner_product_error(x, y)
            variances.append(result["variance"])

        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1], f"Variance should decrease: {variances}"

    def test_qjl_signs_binary(self, dim, random_vectors):
        quant = TurboQuantProd(dim, bit_width=2)
        q = quant.quantize(random_vectors)
        unique_vals = torch.unique(q.qjl_signs)
        assert set(unique_vals.tolist()).issubset({-1, 1})


class TestOutlierAwareQuantizer:
    def test_effective_bits_25(self, dim):
        quant = OutlierAwareQuantizer(dim, outlier_bits=3, normal_bits=2, n_outlier_channels=64)
        assert abs(quant.effective_bits - 2.5) < 1e-6

    def test_effective_bits_35(self, dim):
        quant = OutlierAwareQuantizer(dim, outlier_bits=4, normal_bits=3, n_outlier_channels=32)
        assert abs(quant.effective_bits - 3.25) < 0.5

    def test_round_trip_shape(self, dim, random_vectors):
        quant = OutlierAwareQuantizer(dim, outlier_bits=3, normal_bits=2)
        recon = quant.round_trip(random_vectors)
        assert recon.shape == random_vectors.shape

    def test_outlier_calibration(self, dim):
        torch.manual_seed(0)
        x = torch.randn(100, dim)
        x[:, :10] *= 50.0  # make first 10 channels outliers

        quant = OutlierAwareQuantizer(dim, outlier_bits=4, normal_bits=2, n_outlier_channels=10)
        quant.calibrate_outliers(x)
        assert quant._outlier_mask is not None
        assert quant._outlier_mask[:10].all(), "First 10 channels should be detected as outliers"

    def test_better_than_flat(self, dim):
        torch.manual_seed(0)
        x = torch.randn(256, dim)
        x[:, :16] *= 20.0

        flat_quant = TurboQuantMSE(dim, bit_width=2)
        flat_mse = flat_quant.mse(x).item()

        outlier_quant = OutlierAwareQuantizer(
            dim, outlier_bits=3, normal_bits=2, n_outlier_channels=16
        )
        outlier_quant.calibrate_outliers(x)
        recon = outlier_quant.round_trip(x)
        outlier_mse = ((x.float() - recon) ** 2).sum(dim=-1).mean().item()

        assert outlier_mse < flat_mse, (
            f"Outlier-aware ({outlier_mse:.4f}) should beat flat ({flat_mse:.4f}) on outlier data"
        )
