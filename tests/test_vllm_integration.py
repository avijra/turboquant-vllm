"""Integration tests for TurboQuant vLLM integration.

Validates config registration, dtype mapping, sub-byte packing,
query-rotation trick, cache shape computation, and both rotation modes
(Hadamard and full matrix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

VLLM_SRC = Path(__file__).resolve().parent.parent / "vllm-src"
if str(VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLLM_SRC))


class TestCacheDTypeRegistration:
    def test_turboquant_in_source(self):
        src = (VLLM_SRC / "vllm" / "config" / "cache.py").read_text()
        assert '"turboquant"' in src

    def test_str_dtype_mapping(self):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        assert "turboquant" in STR_DTYPE_TO_TORCH_DTYPE
        assert STR_DTYPE_TO_TORCH_DTYPE["turboquant"] == torch.int8


class TestIsQuantizedKvCache:
    def test_turboquant_is_quantized(self):
        from vllm.v1.attention.backend import is_quantized_kv_cache

        assert is_quantized_kv_cache("turboquant")

    def test_fp8_still_quantized(self):
        from vllm.v1.attention.backend import is_quantized_kv_cache

        assert is_quantized_kv_cache("fp8")

    def test_auto_not_quantized(self):
        from vllm.v1.attention.backend import is_quantized_kv_cache

        assert not is_quantized_kv_cache("auto")


class TestPackedDim:
    def test_2bit_packing(self):
        from vllm.v1.attention.ops.turboquant_cache import packed_dim

        assert packed_dim(128, 2) == 32
        assert packed_dim(64, 2) == 16
        assert packed_dim(256, 2) == 64

    def test_4bit_packing(self):
        from vllm.v1.attention.ops.turboquant_cache import packed_dim

        assert packed_dim(128, 4) == 64
        assert packed_dim(64, 4) == 32


class TestPackUnpack:
    def test_roundtrip_2bit(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _pytorch_pack,
            _pytorch_unpack,
        )

        indices = torch.randint(0, 4, (8, 128))
        packed = _pytorch_pack(indices, 128, 2)
        assert packed.shape == (8, 32)
        assert packed.dtype == torch.uint8

        unpacked = _pytorch_unpack(packed, 128, 2)
        assert torch.equal(indices, unpacked)

    def test_roundtrip_4bit(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _pytorch_pack,
            _pytorch_unpack,
        )

        indices = torch.randint(0, 16, (4, 64))
        packed = _pytorch_pack(indices, 64, 4)
        assert packed.shape == (4, 32)

        unpacked = _pytorch_unpack(packed, 64, 4)
        assert torch.equal(indices, unpacked)

    def test_batched_shape(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _pytorch_pack,
            _pytorch_unpack,
        )

        indices = torch.randint(0, 4, (2, 4, 128))
        packed = _pytorch_pack(indices, 128, 2)
        assert packed.shape == (2, 4, 32)

        unpacked = _pytorch_unpack(packed, 128, 2)
        assert torch.equal(indices, unpacked)


class TestFastHadamardTransform:
    def test_orthogonality(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _fast_hadamard_transform,
        )

        d = 128
        x = torch.randn(d)
        hx = _fast_hadamard_transform(x)
        assert torch.allclose(
            torch.dot(x, x), torch.dot(hx, hx), atol=1e-4,
        ), "Hadamard should preserve L2 norm"

    def test_is_own_inverse(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _fast_hadamard_transform,
        )

        d = 64
        x = torch.randn(8, d)
        hx = _fast_hadamard_transform(x)
        hhx = _fast_hadamard_transform(hx)
        assert torch.allclose(x, hhx, atol=1e-5), (
            "Normalized WHT should be its own inverse"
        )

    def test_batched(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            _fast_hadamard_transform,
        )

        x = torch.randn(4, 8, 128)
        hx = _fast_hadamard_transform(x)
        assert hx.shape == x.shape


class TestTurboQuantStateHadamard:
    @pytest.fixture
    def tq_state(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        return TurboQuantState(
            head_size=64,
            num_kv_heads=4,
            device=torch.device("cpu"),
            bit_width=2,
            layer_idx=42,
            rotation_mode="hadamard",
        )

    def test_packed_dim(self, tq_state):
        assert tq_state.packed_dim == 16
        assert tq_state.vals_per_byte == 4

    def test_hadamard_signs_stored(self, tq_state):
        assert tq_state.signs is not None
        assert tq_state.signs.shape == (64,)
        assert tq_state.rotation is None
        assert set(tq_state.signs.tolist()) == {-1.0, 1.0}

    def test_rotation_mode(self, tq_state):
        assert tq_state.rotation_mode == "hadamard"

    def test_rotate_preserves_norm(self, tq_state):
        x = torch.randn(8, 64)
        rotated = tq_state.rotate_forward(x)
        orig_norms = torch.norm(x, dim=-1)
        rot_norms = torch.norm(rotated, dim=-1)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-4)

    def test_rotate_inverse_roundtrip(self, tq_state):
        x = torch.randn(8, 64)
        rotated = tq_state.rotate_forward(x)
        recovered = tq_state.rotate_inverse(rotated)
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_deterministic_signs(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        s1 = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=7,
            rotation_mode="hadamard",
        )
        s2 = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=7,
            rotation_mode="hadamard",
        )
        assert torch.equal(s1.signs, s2.signs)

    def test_different_layers_different_signs(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        s1 = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=0,
            rotation_mode="hadamard",
        )
        s2 = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=1,
            rotation_mode="hadamard",
        )
        assert not torch.equal(s1.signs, s2.signs)

    def test_non_power_of_2_raises(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        with pytest.raises(ValueError, match="power of 2"):
            TurboQuantState(
                65, 4, torch.device("cpu"), 2, layer_idx=0,
                rotation_mode="hadamard",
            )

    def test_codebook_for_2bit(self, tq_state):
        assert tq_state.centroids.shape == (4,)
        assert tq_state.boundaries.shape == (3,)

    def test_norm_cache_allocation(self, tq_state):
        nc = tq_state.get_norm_cache(10, 16, kv_idx=0)
        assert nc.shape == (10 * 16 * 4,)
        assert nc.dtype == torch.float32

        nc2 = tq_state.get_norm_cache(10, 16, kv_idx=0)
        assert nc is nc2

        nc_v = tq_state.get_norm_cache(10, 16, kv_idx=1)
        assert nc_v is not nc


class TestTurboQuantStateMatrix:
    def test_matrix_mode_stores_rotation(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        tq = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=99,
            rotation_mode="matrix",
        )
        assert tq.rotation is not None
        assert tq.rotation.shape == (64, 64)
        assert tq.signs is None
        identity = tq.rotation @ tq.rotation.T
        assert torch.allclose(identity, torch.eye(64), atol=1e-5)

    def test_matrix_rotate_roundtrip(self):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        tq = TurboQuantState(
            64, 4, torch.device("cpu"), 2, layer_idx=99,
            rotation_mode="matrix",
        )
        x = torch.randn(8, 64)
        rotated = tq.rotate_forward(x)
        recovered = tq.rotate_inverse(rotated)
        assert torch.allclose(x, recovered, atol=1e-4)


class TestWriteAndReadRoundTrip:
    @pytest.fixture(params=["hadamard", "matrix"])
    def setup(self, request):
        from vllm.v1.attention.ops.turboquant_cache import TurboQuantState

        tq = TurboQuantState(
            head_size=64,
            num_kv_heads=4,
            device=torch.device("cpu"),
            bit_width=2,
            layer_idx=99,
            rotation_mode=request.param,
        )
        num_blocks = 2
        block_size = 16
        return tq, num_blocks, block_size

    def test_write_read_quality(self, setup):
        from vllm.v1.attention.ops.turboquant_cache import (
            turboquant_dequantize_cache,
            turboquant_reshape_and_cache,
        )

        tq, num_blocks, block_size = setup
        num_tokens = 4
        num_heads = 4
        head_size = 64
        p_dim = tq.packed_dim

        key = torch.randn(num_tokens, num_heads, head_size)
        value = torch.randn(num_tokens, num_heads, head_size)

        k_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        v_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        k_norms = tq.get_norm_cache(num_blocks, block_size, 0)
        v_norms = tq.get_norm_cache(num_blocks, block_size, 1)

        turboquant_reshape_and_cache(
            key, value, k_cache, v_cache,
            slot_mapping, tq, k_norms, v_norms,
        )

        assert k_cache[0, 0].any()
        assert v_cache[0, 0].any()

        k_fp, v_fp = turboquant_dequantize_cache(
            k_cache, v_cache, tq, k_norms, v_norms,
        )
        assert k_fp.shape == (
            num_blocks, block_size, num_heads, head_size,
        )

        k_recon = k_fp[0, :num_tokens]
        mse = ((key.float() - k_recon.float()) ** 2).mean()
        signal = (key.float() ** 2).mean()
        relative_mse = mse / signal
        assert relative_mse < 1.0, (
            f"Relative MSE too high: {relative_mse:.4f}"
        )

    def test_padding_tokens_skipped(self, setup):
        from vllm.v1.attention.ops.turboquant_cache import (
            turboquant_reshape_and_cache,
        )

        tq, num_blocks, block_size = setup
        num_heads = 4
        head_size = 64
        p_dim = tq.packed_dim

        key = torch.randn(2, num_heads, head_size)
        value = torch.randn(2, num_heads, head_size)

        k_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        v_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        slot_mapping = torch.tensor([0, -1], dtype=torch.int64)
        k_norms = tq.get_norm_cache(num_blocks, block_size, 0)
        v_norms = tq.get_norm_cache(num_blocks, block_size, 1)

        turboquant_reshape_and_cache(
            key, value, k_cache, v_cache,
            slot_mapping, tq, k_norms, v_norms,
        )

        assert k_cache[0, 0].any()
        assert not k_cache[0, 1].any()


class TestQueryRotationTrick:
    @pytest.fixture(params=["hadamard", "matrix"])
    def rotation_mode(self, request):
        return request.param

    def test_rotation_preserves_dot_product(self, rotation_mode):
        from vllm.v1.attention.ops.turboquant_cache import (
            TurboQuantState,
            turboquant_dequant_rotated,
            turboquant_dequantize_cache,
            turboquant_reshape_and_cache,
        )

        tq = TurboQuantState(
            64, 4, torch.device("cpu"), bit_width=2, layer_idx=200,
            rotation_mode=rotation_mode,
        )
        num_blocks, block_size = 1, 16
        num_tokens, num_heads, head_size = 4, 4, 64
        p_dim = tq.packed_dim

        key = torch.randn(num_tokens, num_heads, head_size)
        value = torch.randn(num_tokens, num_heads, head_size)
        query = torch.randn(num_tokens, num_heads, head_size)

        k_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        v_cache = torch.zeros(
            num_blocks, block_size, num_heads, p_dim,
            dtype=torch.uint8,
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
        k_norms = tq.get_norm_cache(num_blocks, block_size, 0)
        v_norms = tq.get_norm_cache(num_blocks, block_size, 1)

        turboquant_reshape_and_cache(
            key, value, k_cache, v_cache,
            slot_mapping, tq, k_norms, v_norms,
        )

        k_full, _ = turboquant_dequantize_cache(
            k_cache, v_cache, tq, k_norms, v_norms,
        )
        k_full_tokens = k_full[0, :num_tokens].float()
        dot_full = torch.einsum(
            "nhd,nhd->nh", query.float(), k_full_tokens,
        )

        k_rotated = turboquant_dequant_rotated(
            k_cache, k_norms, tq, torch.float32,
        )
        k_rot_tokens = k_rotated[0, :num_tokens]
        q_rot = tq.rotate_forward(query)
        dot_rotated = torch.einsum(
            "nhd,nhd->nh", q_rot, k_rot_tokens,
        )

        assert torch.allclose(dot_full, dot_rotated, atol=1e-2), (
            f"Max diff: {(dot_full - dot_rotated).abs().max():.6f}"
        )


class TestCacheShape:
    def test_packed_shape(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            TURBOQUANT_BIT_WIDTH,
            packed_dim,
        )

        p_dim = packed_dim(128, TURBOQUANT_BIT_WIDTH)
        assert p_dim == 32

    def test_compression_ratio(self):
        from vllm.v1.attention.ops.turboquant_cache import (
            TURBOQUANT_BIT_WIDTH,
            packed_dim,
        )

        head_size = 128
        p_dim = packed_dim(head_size, TURBOQUANT_BIT_WIDTH)
        fp16_bytes = head_size * 2
        tq_bytes = p_dim * 1 + 4
        ratio = fp16_bytes / tq_bytes
        assert ratio > 7.0, f"Expected >7x compression, got {ratio:.1f}x"


class TestCodebookSolver:
    def test_codebook_2bit(self):
        from vllm.v1.attention.ops.turboquant_cache import get_codebook

        centroids, boundaries = get_codebook(128, 2, "cpu")
        assert centroids.shape == (4,)
        assert boundaries.shape == (3,)
        assert (centroids[1:] > centroids[:-1]).all()

    def test_codebook_caching(self):
        from vllm.v1.attention.ops.turboquant_cache import get_codebook

        c1, _ = get_codebook(128, 2, "cpu")
        c2, _ = get_codebook(128, 2, "cpu")
        assert c1 is c2
