"""Tests for Triton kernel wrappers and PyTorch fallback paths."""

from __future__ import annotations

import pytest
import torch

from turboquant.codebook import BetaCodebook
from turboquant.triton_kernels import (
    DequantizeReadKernel,
    QuantizeWriteKernel,
    rotate_query,
)


@pytest.fixture
def dim():
    return 64


@pytest.fixture
def setup(dim):
    torch.manual_seed(42)
    builder = BetaCodebook(dimension=dim)
    cb = builder.build(2)

    rotation = torch.randn(dim, dim)
    q, r = torch.linalg.qr(rotation)
    diag_sign = torch.sign(torch.diagonal(r))
    rotation = q * diag_sign.unsqueeze(0)

    centroids = torch.from_numpy(cb.centroids).float()
    boundaries = torch.from_numpy(cb.boundaries).float()

    return rotation, centroids, boundaries


class TestQuantizeWriteKernel:
    def test_output_shapes(self, dim, setup):
        rotation, centroids, boundaries = setup
        kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)

        x = torch.randn(32, dim)
        packed, norms = kernel(x)

        expected_packed_cols = (dim + 3) // 4
        assert packed.shape == (32, expected_packed_cols)
        assert norms.shape == (32,)

    def test_norms_match(self, dim, setup):
        rotation, centroids, boundaries = setup
        kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)

        x = torch.randn(16, dim)
        _, norms = kernel(x)

        expected_norms = torch.norm(x, dim=-1)
        assert torch.allclose(norms, expected_norms, atol=1e-4)

    def test_packed_values_in_range(self, dim, setup):
        rotation, centroids, boundaries = setup
        kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)

        x = torch.randn(16, dim)
        packed, _ = kernel(x)
        assert packed.dtype == torch.uint8

    def test_deterministic(self, dim, setup):
        rotation, centroids, boundaries = setup
        kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)

        x = torch.randn(8, dim)
        p1, n1 = kernel(x)
        p2, n2 = kernel(x)
        assert torch.equal(p1, p2)
        assert torch.allclose(n1, n2)

    def test_4bit_packing(self, dim):
        builder = BetaCodebook(dimension=dim)
        cb = builder.build(4)

        rotation = torch.randn(dim, dim)
        q, r = torch.linalg.qr(rotation)
        rotation = q * torch.sign(torch.diagonal(r)).unsqueeze(0)

        centroids = torch.from_numpy(cb.centroids).float()
        boundaries = torch.from_numpy(cb.boundaries).float()

        kernel = QuantizeWriteKernel(dim, 4, rotation, centroids, boundaries)
        x = torch.randn(8, dim)
        packed, norms = kernel(x)

        expected_cols = (dim + 1) // 2
        assert packed.shape == (8, expected_cols)


class TestDequantizeReadKernel:
    def test_output_shape(self, dim, setup):
        rotation, centroids, boundaries = setup
        write_kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)
        read_kernel = DequantizeReadKernel(dim, 2, rotation, centroids)

        x = torch.randn(16, dim)
        packed, norms = write_kernel(x)
        recon = read_kernel(packed, norms, apply_inverse_rotation=True)

        assert recon.shape == (16, dim)

    def test_round_trip_quality(self, dim, setup):
        rotation, centroids, boundaries = setup
        write_kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)
        read_kernel = DequantizeReadKernel(dim, 2, rotation, centroids)

        x = torch.randn(64, dim)
        packed, norms = write_kernel(x)
        recon = read_kernel(packed, norms, apply_inverse_rotation=True)

        mse = ((x - recon) ** 2).sum(dim=-1).mean().item()
        norm_sq = (x**2).sum(dim=-1).mean().item()
        relative_mse = mse / norm_sq
        assert relative_mse < 0.5, f"Relative MSE too high: {relative_mse:.4f}"

    def test_no_rotation_gives_rotated_space(self, dim, setup):
        rotation, centroids, boundaries = setup
        write_kernel = QuantizeWriteKernel(dim, 2, rotation, centroids, boundaries)
        read_kernel = DequantizeReadKernel(dim, 2, rotation, centroids)

        x = torch.randn(8, dim)
        packed, norms = write_kernel(x)

        rotated_recon = read_kernel(packed, norms, apply_inverse_rotation=False)
        full_recon = read_kernel(packed, norms, apply_inverse_rotation=True)

        assert not torch.allclose(rotated_recon, full_recon, atol=1e-3)


class TestRotateQuery:
    def test_preserves_norm(self, dim, setup):
        rotation, _, _ = setup
        q = torch.randn(8, dim)
        rotated_q = rotate_query(q, rotation)

        q_norms = torch.norm(q, dim=-1)
        rq_norms = torch.norm(rotated_q, dim=-1)
        assert torch.allclose(q_norms, rq_norms, atol=1e-4)

    def test_query_rotation_equivalence(self, dim, setup):
        """Verify: dot(q, Pi^T * y) == dot(Pi * q, y)"""
        rotation, centroids, _ = setup
        q = torch.randn(1, dim)
        y = torch.randn(1, dim)

        rotated_q = rotate_query(q, rotation)
        inv_rotated_y = y @ rotation

        dot1 = (q * inv_rotated_y).sum()
        dot2 = (rotated_q * y).sum()

        assert torch.allclose(dot1, dot2, atol=1e-4), (
            f"Query rotation trick failed: {dot1.item():.6f} vs {dot2.item():.6f}"
        )
