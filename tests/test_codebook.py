"""Tests for Lloyd-Max codebook solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from turboquant.codebook import REFERENCE_MSE, BetaCodebook, beta_pdf, validate_codebook


class TestBetaPdf:
    def test_integrates_to_one(self):
        from scipy.integrate import quad

        for d in [16, 64, 128, 256]:
            result, _ = quad(lambda x: beta_pdf(x, d), -1, 1)
            assert abs(result - 1.0) < 1e-6, f"d={d}: integral={result}"

    def test_symmetric(self):
        for d in [32, 128]:
            for x in [0.01, 0.05, 0.1]:
                assert abs(beta_pdf(x, d) - beta_pdf(-x, d)) < 1e-10

    def test_zero_outside_bounds(self):
        assert beta_pdf(1.1, 128) == 0.0
        assert beta_pdf(-1.5, 128) == 0.0

    def test_converges_to_gaussian(self):
        d = 1024
        sigma = 1.0 / math.sqrt(d)
        for x in [0.0, 0.01, 0.02]:
            beta_val = beta_pdf(x, d)
            gauss_val = (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
                -0.5 * (x / sigma) ** 2
            )
            assert abs(beta_val - gauss_val) / gauss_val < 0.05, (
                f"d={d}, x={x}: beta={beta_val:.4f}, gauss={gauss_val:.4f}"
            )


class TestBetaCodebook:
    @pytest.fixture
    def builder(self):
        return BetaCodebook(dimension=128)

    def test_centroid_count(self, builder):
        for b in range(1, 5):
            cb = builder.build(b)
            assert len(cb.centroids) == (1 << b)

    def test_centroids_sorted(self, builder):
        for b in range(1, 5):
            cb = builder.build(b)
            diffs = np.diff(cb.centroids)
            assert np.all(diffs > 0), f"b={b}: centroids not strictly increasing"

    def test_centroids_symmetric(self, builder):
        for b in range(1, 5):
            cb = builder.build(b)
            n = len(cb.centroids)
            for i in range(n // 2):
                assert abs(cb.centroids[i] + cb.centroids[n - 1 - i]) < 1e-6

    def test_boundaries_between_centroids(self, builder):
        for b in range(1, 5):
            cb = builder.build(b)
            for i, boundary in enumerate(cb.boundaries):
                assert cb.centroids[i] < boundary < cb.centroids[i + 1]

    def test_mse_matches_paper(self, builder):
        for b in [1, 2, 3, 4]:
            cb = builder.build(b)
            info = validate_codebook(cb)
            ref = REFERENCE_MSE[b]
            ratio = info["total_mse"] / ref
            assert 0.8 < ratio < 1.5, (
                f"b={b}: total_mse={info['total_mse']:.6f}, ref={ref}, ratio={ratio:.4f}"
            )

    def test_mse_decreases_with_bits(self, builder):
        mses = []
        for b in range(1, 5):
            cb = builder.build(b)
            mses.append(cb.mse_per_coord)
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], "MSE should decrease as bit-width increases"

    def test_different_dimensions(self):
        for d in [64, 128, 256]:
            builder = BetaCodebook(dimension=d)
            cb = builder.build(2)
            assert cb.dimension == d
            assert len(cb.centroids) == 4

    def test_caching(self, builder):
        cb1 = builder.build(2)
        cb2 = builder.build(2)
        assert cb1 is cb2

    def test_invalid_bit_width(self, builder):
        with pytest.raises(ValueError):
            builder.build(0)
        with pytest.raises(ValueError):
            builder.build(9)
