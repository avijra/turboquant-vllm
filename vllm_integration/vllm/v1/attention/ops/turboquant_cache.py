# SPDX-License-Identifier: Apache-2.0

"""TurboQuant KV cache ops for vLLM — GPU-optimized with sub-byte packing.

Write path:  normalize → rotate by Π → scalar quantize → pack b-bit → scatter
Read path:   gather packed → unpack → centroid lookup → scale by norm
             (stays in rotated space; query-rotation trick avoids Π^T)

Rotation modes:
  - "hadamard" (default): Randomized Walsh-Hadamard Transform.
    O(d log d) per vector, stores only d random signs (512 B for d=128).
  - "matrix": Full random orthogonal matrix via QR decomposition.
    O(d²) per vector, stores d×d float32 (64 KB for d=128).

Storage layout per token per KV head (4-bit default, head_size=128):
  - 64 bytes packed indices (2 values per uint8)
  - 4 bytes float32 norm (stored in separate auxiliary buffer)
  Total: 68 bytes vs 256 bytes fp16 → 3.76× compression
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

TURBOQUANT_BIT_WIDTH = 4


def packed_dim(head_size: int, bit_width: int = TURBOQUANT_BIT_WIDTH) -> int:
    vals_per_byte = 8 // bit_width
    return (head_size + vals_per_byte - 1) // vals_per_byte


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform — O(d log d) rotation
# ---------------------------------------------------------------------------


def _fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Fast Walsh-Hadamard Transform along last dimension.

    Butterfly decomposition: log2(d) vectorized steps, O(d log d) total.
    The normalized WHT matrix is symmetric and orthogonal (its own inverse).
    Requires d to be a power of 2.
    """
    d = x.shape[-1]
    batch_shape = x.shape[:-1]
    result = x.float()
    h = 1
    while h < d:
        groups = d // (2 * h)
        result = result.view(*batch_shape, groups, 2, h)
        a = result[..., 0, :] + result[..., 1, :]
        b = result[..., 0, :] - result[..., 1, :]
        result = torch.stack([a, b], dim=-2).reshape(*batch_shape, d)
        h *= 2
    return result * (1.0 / math.sqrt(d))


def _generate_rademacher_signs(
    dim: int, device: torch.device, seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (
        torch.randint(0, 2, (dim,), generator=gen, device=device) * 2 - 1
    ).float()


# ---------------------------------------------------------------------------
# Codebook solver (runs once at init, cached)
# ---------------------------------------------------------------------------


def _beta_pdf_centroids(
    dim: int, n_centroids: int, max_iters: int = 500,
) -> list[float]:
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import gammaln
    from scipy.stats import norm

    log_c = (
        gammaln(dim / 2)
        - 0.5 * math.log(math.pi)
        - gammaln((dim - 1) / 2)
    )
    exp = (dim - 3) / 2

    def pdf(x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0
        return math.exp(log_c + exp * math.log(1 - x * x))

    sigma = 1.0 / math.sqrt(dim)
    q_lo = 1 / (2 * n_centroids)
    quantiles = np.linspace(q_lo, 1 - q_lo, n_centroids)
    centroids = norm.ppf(quantiles, scale=sigma).astype(np.float64)

    for _ in range(max_iters):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0
            for i in range(n_centroids - 1)
        ]
        new = np.zeros(n_centroids)
        for i in range(n_centroids):
            lo = boundaries[i - 1] if i > 0 else -1.0
            hi = boundaries[i] if i < n_centroids - 1 else 1.0
            mass, _ = quad(pdf, lo, hi, limit=200)
            if mass < 1e-30:
                new[i] = (lo + hi) / 2.0
            else:
                mom, _ = quad(
                    lambda x: x * pdf(x), lo, hi, limit=200,
                )
                new[i] = mom / mass
        if np.max(np.abs(new - centroids)) < 1e-12:
            centroids = new
            break
        centroids = new

    return centroids.tolist()


@lru_cache(maxsize=32)
def get_codebook(
    dim: int, bit_width: int, device_str: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = 1 << bit_width
    c_list = _beta_pdf_centroids(dim, n)
    centroids = torch.tensor(
        c_list, dtype=torch.float32, device=device_str,
    )
    boundaries = torch.tensor(
        [(c_list[i] + c_list[i + 1]) / 2.0 for i in range(n - 1)],
        dtype=torch.float32,
        device=device_str,
    )
    return centroids, boundaries


def generate_rotation_matrix(
    dim: int, device: torch.device, seed: int | None = None,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
    gaussian = torch.randn(
        dim, dim, device=device, dtype=torch.float32, generator=gen,
    )
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diagonal(r))
    return q * diag_sign.unsqueeze(0)


# ---------------------------------------------------------------------------
# Triton GPU kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _tq_write_kernel(
        kv_ptr,
        cache_ptr,
        norm_ptr,
        rotation_ptr,
        boundaries_ptr,
        slot_mapping_ptr,
        num_tokens,
        num_heads: tl.constexpr,
        D: tl.constexpr,
        block_size: tl.constexpr,
        packed_dim: tl.constexpr,
        n_boundaries: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BIT_WIDTH: tl.constexpr,
    ):
        """Fused normalize -> rotate -> quantize -> pack -> scatter.

        Grid: (num_tokens, num_heads)
        Uses full rotation matrix path (for non-Hadamard mode).
        """
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)
        if token_id >= num_tokens:
            return

        slot = tl.load(slot_mapping_ptr + token_id)
        if slot < 0:
            return

        src_off = (token_id * num_heads + head_id) * D
        d_range = tl.arange(0, BLOCK_D)
        d_mask = d_range < D

        x = tl.load(
            kv_ptr + src_off + d_range, mask=d_mask, other=0.0,
        ).to(tl.float32)

        norm_sq = tl.sum(x * x, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        x = x / norm

        rotated = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(D):
            x_k = tl.load(kv_ptr + src_off + k).to(tl.float32)
            x_k = x_k / norm
            rot_row = tl.load(
                rotation_ptr + d_range * D + k,
                mask=d_mask,
                other=0.0,
            )
            rotated += x_k * rot_row

        indices = tl.zeros([BLOCK_D], dtype=tl.int32)
        for bi in range(n_boundaries):
            b = tl.load(boundaries_ptr + bi)
            indices += tl.where(rotated >= b, 1, 0)

        block_idx = slot // block_size
        block_off = slot % block_size
        cache_head_off = (
            block_idx * block_size * num_heads * packed_dim
            + block_off * num_heads * packed_dim
            + head_id * packed_dim
        )

        vals_per_byte: tl.constexpr = 8 // BIT_WIDTH
        bit_mask: tl.constexpr = (1 << BIT_WIDTH) - 1

        pack_range = tl.arange(0, BLOCK_D)
        pack_mask = pack_range < packed_dim
        packed = tl.zeros([BLOCK_D], dtype=tl.uint8)
        for sub in range(vals_per_byte):
            col = pack_range * vals_per_byte + sub
            col_mask = col < D
            idx_at_col = tl.where(
                col_mask,
                tl.load(
                    kv_ptr + src_off + col,
                    mask=col_mask,
                    other=0.0,
                ),
                tl.zeros([BLOCK_D], dtype=tl.float32),
            )
            _ = idx_at_col
            rotated_at_col = tl.zeros([BLOCK_D], dtype=tl.float32)
            for k in range(D):
                x_k = tl.load(kv_ptr + src_off + k).to(tl.float32)
                x_k = x_k / norm
                rot_elem = tl.where(
                    col_mask,
                    tl.load(
                        rotation_ptr + col * D + k,
                        mask=col_mask,
                        other=0.0,
                    ),
                    tl.zeros([BLOCK_D], dtype=tl.float32),
                )
                rotated_at_col += x_k * rot_elem
            idx_val = tl.zeros([BLOCK_D], dtype=tl.int32)
            for bi in range(n_boundaries):
                b = tl.load(boundaries_ptr + bi)
                idx_val += tl.where(
                    col_mask & (rotated_at_col >= b), 1, 0,
                )
            shift = (sub * BIT_WIDTH)
            packed |= (
                (idx_val & bit_mask).to(tl.uint8) << shift
            ).to(tl.uint8)

        tl.store(
            cache_ptr + cache_head_off + pack_range,
            packed,
            mask=pack_mask,
        )

        norm_off = (
            block_idx * block_size * num_heads
            + block_off * num_heads
            + head_id
        )
        tl.store(norm_ptr + norm_off, norm)

    @triton.jit
    def _tq_read_kernel(
        cache_ptr,
        norm_ptr,
        centroids_ptr,
        out_ptr,
        num_blocks,
        block_size: tl.constexpr,
        num_heads: tl.constexpr,
        D: tl.constexpr,
        packed_dim: tl.constexpr,
        BIT_WIDTH: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Unpack -> centroid gather -> scale by norm.

        Output is in rotated space (no inverse rotation).
        Grid: (num_blocks * block_size, num_heads)
        """
        flat_id = tl.program_id(0)
        head_id = tl.program_id(1)
        if flat_id >= num_blocks * block_size:
            return

        cache_off = (
            flat_id * num_heads * packed_dim
            + head_id * packed_dim
        )

        vals_per_byte: tl.constexpr = 8 // BIT_WIDTH
        bit_mask: tl.constexpr = (1 << BIT_WIDTH) - 1

        d_range = tl.arange(0, BLOCK_D)
        d_mask = d_range < D

        byte_idx = d_range // vals_per_byte
        sub_idx = d_range % vals_per_byte

        packed_bytes = tl.load(
            cache_ptr + cache_off + byte_idx,
            mask=d_mask & (byte_idx < packed_dim),
            other=0,
        ).to(tl.uint8)

        shift = (sub_idx * BIT_WIDTH).to(tl.uint8)
        indices = ((packed_bytes >> shift) & bit_mask).to(tl.int32)

        centroid_vals = tl.load(
            centroids_ptr + indices, mask=d_mask, other=0.0,
        )

        norm_off = flat_id * num_heads + head_id
        norm = tl.load(norm_ptr + norm_off)

        out_off = flat_id * num_heads * D + head_id * D
        tl.store(
            out_ptr + out_off + d_range,
            centroid_vals * norm,
            mask=d_mask,
        )


# ---------------------------------------------------------------------------
# PyTorch fallback implementations (CPU / no-Triton)
# ---------------------------------------------------------------------------

def _pytorch_pack(
    indices: torch.Tensor, dim: int, bit_width: int,
) -> torch.Tensor:
    vals_per_byte = 8 // bit_width
    p_dim = packed_dim(dim, bit_width)
    orig_shape = indices.shape[:-1]
    flat = indices.reshape(-1, dim).to(torch.uint8)
    N = flat.shape[0]
    packed = torch.zeros(
        N, p_dim, dtype=torch.uint8, device=flat.device,
    )
    for sub in range(vals_per_byte):
        cols = torch.arange(sub, dim, vals_per_byte, device=flat.device)
        byte_col = torch.arange(len(cols), device=flat.device)
        shift = sub * bit_width
        packed[:, byte_col] |= flat[:, cols] << shift
    return packed.reshape(*orig_shape, p_dim)


def _pytorch_unpack(
    packed: torch.Tensor, head_size: int, bit_width: int,
) -> torch.Tensor:
    vals_per_byte = 8 // bit_width
    mask = (1 << bit_width) - 1
    orig_shape = packed.shape[:-1]
    p_dim = packed.shape[-1]
    flat = packed.reshape(-1, p_dim)
    N = flat.shape[0]
    indices = torch.zeros(
        N, head_size, dtype=torch.long, device=packed.device,
    )
    for sub in range(vals_per_byte):
        cols = torch.arange(
            sub, head_size, vals_per_byte, device=packed.device,
        )
        byte_col = torch.arange(len(cols), device=packed.device)
        shift = sub * bit_width
        indices[:, cols] = (flat[:, byte_col].long() >> shift) & mask
    return indices.reshape(*orig_shape, head_size)


# ---------------------------------------------------------------------------
# TurboQuantState — per-layer state
# ---------------------------------------------------------------------------

ROTATION_BASE_SEED = 0xDEAD_BEEF
_tq_layer_counter = 0


class TurboQuantState:
    """Per-layer state for TurboQuant KV cache quantization.

    Holds rotation parameters, codebook, and per-token norm buffers.
    Initialized once per attention layer at model load time.

    Two rotation modes:
      - "hadamard": Randomized Walsh-Hadamard Transform.
        Stores d random ±1 signs (512 B for d=128).
        Forward/inverse are O(d log d) via butterfly decomposition.
      - "matrix": Full random orthogonal matrix via QR.
        Stores d×d matrix (64 KB for d=128). O(d²) per vector.
    """

    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        device: torch.device,
        bit_width: int = TURBOQUANT_BIT_WIDTH,
        layer_idx: int | None = None,
        rotation_mode: str = "hadamard",
    ):
        global _tq_layer_counter  # noqa: PLW0603
        if layer_idx is None:
            layer_idx = _tq_layer_counter
            _tq_layer_counter += 1
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.device = device
        self.bit_width = bit_width
        self.packed_dim = packed_dim(head_size, bit_width)
        self.vals_per_byte = 8 // bit_width
        self.rotation_mode = rotation_mode

        seed = ROTATION_BASE_SEED + layer_idx

        if rotation_mode == "hadamard":
            if head_size & (head_size - 1) != 0:
                raise ValueError(
                    f"Hadamard rotation requires head_size to be a "
                    f"power of 2, got {head_size}",
                )
            self.signs = _generate_rademacher_signs(
                head_size, device, seed,
            )
            self.rotation = None
        else:
            self.rotation = generate_rotation_matrix(
                head_size, device, seed=seed,
            )
            self.signs = None

        centroids, boundaries = get_codebook(
            head_size, bit_width, str(device),
        )
        self.centroids = centroids
        self.boundaries = boundaries

        self._norm_caches: dict[int, torch.Tensor] = {}
        self._use_triton = HAS_TRITON and device.type == "cuda"

    def rotate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rotation Π to x.  O(d log d) for Hadamard, O(d²) for matrix."""
        if self.rotation_mode == "hadamard":
            return _fast_hadamard_transform(x * self.signs)
        return (x.float() @ self.rotation.T).to(x.dtype)

    def rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply Π^T to y (undo rotation).  Same cost as forward."""
        if self.rotation_mode == "hadamard":
            return _fast_hadamard_transform(y) * self.signs
        return (y.float() @ self.rotation).to(y.dtype)

    def get_norm_cache(
        self, num_blocks: int, block_size: int, kv_idx: int,
    ) -> torch.Tensor:
        needed = num_blocks * block_size * self.num_kv_heads
        buf = self._norm_caches.get(kv_idx)
        if buf is None or buf.numel() < needed:
            new_buf = torch.zeros(
                needed,
                dtype=torch.float32,
                device=self.device,
            )
            if buf is not None:
                new_buf[:buf.numel()] = buf
            self._norm_caches[kv_idx] = new_buf
        return self._norm_caches[kv_idx][:needed]


# ---------------------------------------------------------------------------
# Write path — vectorized scatter (no Python per-token loop)
# ---------------------------------------------------------------------------


def turboquant_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    tq_state: TurboQuantState,
    k_norm_cache: torch.Tensor,
    v_norm_cache: torch.Tensor,
):
    """Write K/V into packed quantized cache using TurboQuant.

    Fully vectorized: quantizes all tokens in parallel, then scatters
    to the paged cache via advanced indexing (no Python for-loop).

    Args:
        key:         [num_tokens, num_kv_heads, head_size]
        value:       [num_tokens, num_kv_heads, head_size]
        key_cache:   [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        value_cache: [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        slot_mapping: [num_tokens]  (-1 = padding, skip)
        tq_state:    TurboQuantState
        k_norm_cache: [num_blocks * block_size * num_kv_heads] float32
        v_norm_cache: [num_blocks * block_size * num_kv_heads] float32
    """
    head_size = tq_state.head_size
    bit_width = tq_state.bit_width
    boundaries = tq_state.boundaries
    block_size = key_cache.shape[1]
    num_heads = key.shape[1]

    safe_slots = slot_mapping.clamp(min=0)
    block_idx = safe_slots // block_size
    block_off = safe_slots % block_size
    valid_mask = (slot_mapping >= 0).unsqueeze(-1).unsqueeze(-1)

    for tensor, cache, norm_buf in [
        (key, key_cache, k_norm_cache),
        (value, value_cache, v_norm_cache),
    ]:
        flat = tensor.float()
        norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = flat / norms

        rotated = tq_state.rotate_forward(normalized)

        indices = torch.bucketize(rotated, boundaries)
        packed = _pytorch_pack(indices, head_size, bit_width)

        packed_masked = torch.where(
            valid_mask, packed.to(cache.dtype), cache[block_idx, block_off],
        )
        cache[block_idx, block_off] = packed_masked

        flat_base = (
            block_idx * block_size * num_heads
            + block_off * num_heads
        )
        head_offsets = torch.arange(
            num_heads, device=flat_base.device,
        )
        scatter_idx = (
            flat_base.unsqueeze(1) + head_offsets.unsqueeze(0)
        ).reshape(-1)
        norm_vals = norms[:, :, 0].reshape(-1)
        valid_1d = (slot_mapping >= 0).unsqueeze(-1).expand(
            -1, num_heads,
        ).reshape(-1)
        norm_vals = torch.where(
            valid_1d, norm_vals, norm_buf[scatter_idx],
        )
        norm_buf.scatter_(0, scatter_idx, norm_vals)


# ---------------------------------------------------------------------------
# Read path — query-rotation trick
# ---------------------------------------------------------------------------


def turboquant_dequant_rotated(
    cache: torch.Tensor,
    norm_buf: torch.Tensor,
    tq_state: TurboQuantState,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize cache to float in ROTATED space (no inverse rotation).

    Args:
        cache:    [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        norm_buf: [num_blocks * block_size * num_kv_heads] float32

    Returns:
        [num_blocks, block_size, num_kv_heads, head_size] in target_dtype
    """
    num_blocks, block_size, num_heads, p_dim = cache.shape
    head_size = tq_state.head_size
    centroids = tq_state.centroids
    bit_width = tq_state.bit_width

    if tq_state._use_triton:
        out = torch.empty(
            num_blocks * block_size * num_heads * head_size,
            dtype=torch.float32,
            device=cache.device,
        )
        flat_cache = cache.reshape(
            num_blocks * block_size, num_heads, p_dim,
        ).contiguous()
        BLOCK_D = triton.next_power_of_2(head_size)
        grid = (num_blocks * block_size, num_heads)
        _tq_read_kernel[grid](
            flat_cache,
            norm_buf,
            centroids,
            out,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            p_dim,
            bit_width,
            BLOCK_D,
        )
        return out.reshape(
            num_blocks, block_size, num_heads, head_size,
        ).to(target_dtype)

    indices = _pytorch_unpack(cache, head_size, bit_width)
    reconstructed = centroids[indices]
    norms = norm_buf.reshape(
        num_blocks, block_size, num_heads, 1,
    )
    return (reconstructed * norms).to(target_dtype)


def turboquant_dequantize_for_attention(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tq_state: TurboQuantState,
    k_norm_cache: torch.Tensor,
    v_norm_cache: torch.Tensor,
    query: torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize KV cache and rotate query (query-rotation trick).

    Instead of applying Π^T to every cache entry, rotate q by Π once.
    Returns (rotated_query, key_dequant_rotated, value_dequant_rotated).
    """
    rotated_query = tq_state.rotate_forward(query).to(target_dtype)

    key_deq = turboquant_dequant_rotated(
        key_cache, k_norm_cache, tq_state, target_dtype,
    )
    val_deq = turboquant_dequant_rotated(
        value_cache, v_norm_cache, tq_state, target_dtype,
    )
    return rotated_query, key_deq, val_deq


def turboquant_dequantize_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tq_state: TurboQuantState,
    k_norm_cache: torch.Tensor,
    v_norm_cache: torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full dequantization with inverse rotation (legacy path, slower)."""
    head_size = tq_state.head_size

    results: list[torch.Tensor] = []
    for cache, norm_buf in [
        (key_cache, k_norm_cache),
        (value_cache, v_norm_cache),
    ]:
        rotated = turboquant_dequant_rotated(
            cache, norm_buf, tq_state, torch.float32,
        )
        shape = rotated.shape
        flat = rotated.reshape(-1, head_size)
        unrotated = tq_state.rotate_inverse(flat)
        results.append(
            unrotated.reshape(shape).to(target_dtype)
        )
    return results[0], results[1]
