"""Triton kernels for TurboQuant KV cache quantization.

Write path:  normalize -> rotate -> scalar quantize -> pack into int8
Read path:   unpack -> gather centroids (dequantized rotated space)

The query-rotation trick avoids per-cache-entry inverse rotation:
instead of dequanting K by multiplying by Pi^T, we multiply the query
by Pi once.  Then dot(Pi*q, centroids[idx]) == dot(q, Pi^T * centroids[idx]).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _turboquant_write_2bit_kernel(
        x_ptr,
        rotation_ptr,
        boundaries_ptr,
        norms_out_ptr,
        packed_out_ptr,
        N,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused normalize -> rotate -> 2-bit quantize -> pack 4 values per byte.

        Each program instance handles one vector (one row of the batch).
        """
        row = tl.program_id(0)
        if row >= N:
            return

        x_offset = row * D
        x_vals = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D)

        norm_sq = tl.sum(x_vals * x_vals, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        tl.store(norms_out_ptr + row, norm)
        x_vals = x_vals / norm

        b0 = tl.load(boundaries_ptr + 0)
        b1 = tl.load(boundaries_ptr + 1)
        b2 = tl.load(boundaries_ptr + 2)

        packed_cols = (D + 3) // 4
        for col_group in range(0, D, 4):
            packed_byte = tl.zeros([1], dtype=tl.uint8)
            for sub in range(4):
                col = col_group + sub
                if col < D:
                    dot_val = tl.zeros([1], dtype=tl.float32)
                    for k in range(0, D, BLOCK_D):
                        k_range = k + tl.arange(0, BLOCK_D)
                        k_mask = k_range < D
                        x_k = tl.load(
                            x_ptr + x_offset + k_range, mask=k_mask, other=0.0
                        ) / norm
                        rot_k = tl.load(
                            rotation_ptr + col * D + k_range, mask=k_mask, other=0.0
                        )
                        dot_val += tl.sum(x_k * rot_k, axis=0)

                    idx = tl.where(
                        dot_val < b0, 0,
                        tl.where(dot_val < b1, 1, tl.where(dot_val < b2, 2, 3)),
                    )
                    packed_byte = packed_byte | (idx.to(tl.uint8) << (sub * 2).to(tl.uint8))

            pack_col = col_group // 4
            tl.store(packed_out_ptr + row * packed_cols + pack_col, packed_byte)

    @triton.jit
    def _turboquant_read_2bit_kernel(
        packed_ptr,
        centroids_ptr,
        norms_ptr,
        out_ptr,
        N,
        D: tl.constexpr,
    ):
        """Unpack 2-bit indices, gather centroids -> output in rotated space.

        To get original-space vectors, caller must multiply by Pi^T.
        For attention, use the query-rotation trick instead.
        """
        row = tl.program_id(0)
        col = tl.program_id(1)
        if row >= N or col >= D:
            return

        packed_cols = (D + 3) // 4
        pack_col = col // 4
        sub = col % 4

        packed_byte = tl.load(packed_ptr + row * packed_cols + pack_col)
        idx = (packed_byte >> (sub * 2).to(tl.uint8)) & 0x03

        centroid_val = tl.load(centroids_ptr + idx.to(tl.int32))
        norm = tl.load(norms_ptr + row)

        tl.store(out_ptr + row * D + col, centroid_val * norm)


def triton_quantize_2bit(
    x: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated 2-bit quantization write path.

    Returns (packed_indices [N, D//4] uint8, norms [N] float32).
    """
    assert HAS_TRITON, "Triton not available"
    N, D = x.shape
    assert D == rotation.shape[0] == rotation.shape[1]

    packed_cols = (D + 3) // 4
    packed = torch.zeros(N, packed_cols, dtype=torch.uint8, device=x.device)
    norms = torch.zeros(N, dtype=torch.float32, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    _turboquant_write_2bit_kernel[(N,)](
        x, rotation, boundaries, norms, packed, N, D, BLOCK_D
    )
    return packed, norms


def triton_dequantize_2bit(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    norms: torch.Tensor,
    D: int,
) -> torch.Tensor:
    """Triton-accelerated 2-bit dequantization (in rotated space)."""
    assert HAS_TRITON, "Triton not available"
    N = packed.shape[0]
    out = torch.zeros(N, D, dtype=torch.float32, device=packed.device)

    grid = (N, D)
    _turboquant_read_2bit_kernel[grid](packed, centroids, norms, out, N, D)
    return out


class QuantizeWriteKernel:
    """High-level wrapper for the Triton write path.

    Falls back to PyTorch when Triton is unavailable (CPU, non-NVIDIA GPU).
    """

    def __init__(
        self,
        dim: int,
        bit_width: int,
        rotation: torch.Tensor,
        centroids: torch.Tensor,
        boundaries: torch.Tensor,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.rotation = rotation
        self.centroids = centroids
        self.boundaries = boundaries
        self._use_triton = HAS_TRITON and rotation.is_cuda

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_triton and self.bit_width == 2:
            return triton_quantize_2bit(x.reshape(-1, self.dim), self.rotation, self.boundaries)
        return self._pytorch_fallback(x)

    def _pytorch_fallback(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.dim).float()
        norms = torch.norm(flat, dim=-1)
        normalized = flat / norms.unsqueeze(-1).clamp(min=1e-8)
        rotated = normalized @ self.rotation.T
        indices = torch.bucketize(rotated, self.boundaries)

        if self.bit_width <= 4:
            indices = indices.to(torch.uint8)
            vals_per_byte = 8 // self.bit_width
            D = self.dim
            packed_cols = (D + vals_per_byte - 1) // vals_per_byte
            N = flat.shape[0]
            packed = torch.zeros(N, packed_cols, dtype=torch.uint8, device=flat.device)
            for i in range(vals_per_byte):
                start = i
                cols = torch.arange(start, D, vals_per_byte, device=flat.device)
                if len(cols) == 0:
                    break
                byte_col = cols // vals_per_byte
                shift = (cols % vals_per_byte) * self.bit_width
                packed.scatter_add_(
                    1,
                    byte_col.unsqueeze(0).expand(N, -1),
                    (indices[:, cols] << shift.unsqueeze(0)).to(torch.uint8),
                )
            return packed, norms
        return indices.to(torch.int16), norms


class DequantizeReadKernel:
    """High-level wrapper for the Triton read path."""

    def __init__(
        self,
        dim: int,
        bit_width: int,
        rotation: torch.Tensor,
        centroids: torch.Tensor,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.rotation = rotation
        self.centroids = centroids
        self._use_triton = HAS_TRITON and rotation.is_cuda

    def __call__(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        apply_inverse_rotation: bool = True,
    ) -> torch.Tensor:
        if self._use_triton and self.bit_width == 2:
            rotated = triton_dequantize_2bit(packed, self.centroids, norms, self.dim)
        else:
            rotated = self._pytorch_unpack_and_gather(packed, norms)

        if apply_inverse_rotation:
            return rotated @ self.rotation
        return rotated

    def _pytorch_unpack_and_gather(
        self, packed: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        N = packed.shape[0]
        D = self.dim

        if self.bit_width <= 4:
            vals_per_byte = 8 // self.bit_width
            mask = (1 << self.bit_width) - 1
            indices = torch.zeros(N, D, dtype=torch.long, device=packed.device)
            for i in range(vals_per_byte):
                cols = torch.arange(i, D, vals_per_byte, device=packed.device)
                if len(cols) == 0:
                    break
                byte_col = cols // vals_per_byte
                shift = (cols % vals_per_byte) * self.bit_width
                extracted = (packed[:, byte_col].long() >> shift.unsqueeze(0)) & mask
                indices[:, cols] = extracted
        else:
            indices = packed.long()

        reconstructed = self.centroids[indices] * norms.unsqueeze(-1)
        return reconstructed


def rotate_query(q: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply rotation to query vectors for the query-rotation trick.

    Instead of dequantizing every KV cache entry by multiplying by Pi^T,
    rotate q by Pi once:  dot(Pi*q, centroids[idx]) == dot(q, Pi^T*centroids[idx])
    """
    return q.to(rotation.dtype) @ rotation.T
