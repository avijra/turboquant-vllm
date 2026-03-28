"""TurboQuant vector quantizers for KV cache compression.

Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
from the paper.  Both operate on batches of vectors and are fully
differentiable-free (online, data-oblivious, no calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from turboquant.codebook import BetaCodebook, Codebook


def _generate_rotation_matrix(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix."""
    gaussian = torch.randn(d, d, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diagonal(r))
    q = q * diag_sign.unsqueeze(0)
    return q.to(dtype)


def _generate_qjl_matrix(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate i.i.d. N(0,1) projection matrix for QJL transform."""
    return torch.randn(d, d, device=device, dtype=torch.float32).to(dtype)


@dataclass
class QuantizedMSE:
    indices: torch.Tensor    # (..., d) int16 or int8 — codebook indices
    norms: torch.Tensor      # (...,) float32 — original L2 norms
    bit_width: int


@dataclass
class QuantizedProd:
    mse_indices: torch.Tensor   # (..., d) — codebook indices from MSE stage
    qjl_signs: torch.Tensor     # (..., d) — {-1, +1} sign bits
    norms: torch.Tensor         # (...,) — original L2 norms
    residual_norms: torch.Tensor  # (...,) — ||r||_2
    bit_width: int


class TurboQuantMSE:
    """MSE-optimal vector quantizer (Algorithm 1).

    Quantize:  x -> normalize -> rotate by Pi -> scalar quantize each coord
    Dequantize: look up centroids -> rotate back by Pi^T -> rescale by norm
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 2,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.device = torch.device(device)
        self.dtype = dtype

        cb_builder = BetaCodebook(dimension=dim)
        self._codebook: Codebook = cb_builder.build(bit_width)

        self._rotation = _generate_rotation_matrix(dim, self.device, self.dtype)

        self._centroids = torch.from_numpy(self._codebook.centroids).to(
            device=self.device, dtype=self.dtype
        )
        self._boundaries = torch.from_numpy(self._codebook.boundaries).to(
            device=self.device, dtype=self.dtype
        )

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return self._rotation

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids

    def quantize(self, x: torch.Tensor) -> QuantizedMSE:
        """Quantize a batch of vectors.

        Args:
            x: (..., dim) tensor of input vectors.

        Returns:
            QuantizedMSE with indices and norms.
        """
        original_shape = x.shape
        assert original_shape[-1] == self.dim
        flat = x.reshape(-1, self.dim).to(self.dtype)

        norms = torch.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-8)
        normalized = flat / safe_norms

        rotated = normalized @ self._rotation.T

        indices = torch.bucketize(rotated, self._boundaries)

        return QuantizedMSE(
            indices=indices.reshape(original_shape).to(torch.int16),
            norms=norms.squeeze(-1).reshape(original_shape[:-1]),
            bit_width=self.bit_width,
        )

    def dequantize(self, q: QuantizedMSE) -> torch.Tensor:
        """Reconstruct vectors from quantized representation."""
        flat_indices = q.indices.reshape(-1, self.dim).long()
        reconstructed_rotated = self._centroids[flat_indices]
        reconstructed = reconstructed_rotated @ self._rotation
        norms_flat = q.norms.reshape(-1, 1).to(self.dtype)
        reconstructed = reconstructed * norms_flat
        target_shape = list(q.indices.shape[:-1]) + [self.dim]
        return reconstructed.reshape(target_shape)

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

    def mse(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-vector MSE distortion."""
        recon = self.round_trip(x)
        return ((x.to(self.dtype) - recon) ** 2).sum(dim=-1).mean()

    def storage_bytes_per_vector(self) -> int:
        index_bits = self.dim * self.bit_width
        norm_bytes = 4
        return (index_bits + 7) // 8 + norm_bytes


class TurboQuantProd:
    """Inner-product-optimal vector quantizer (Algorithm 2).

    Two-stage: (b-1)-bit MSE quantizer + 1-bit QJL on the residual.
    Produces unbiased inner product estimates.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 3,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if bit_width < 2:
            raise ValueError("TurboQuantProd requires bit_width >= 2 (uses b-1 for MSE stage)")
        self.dim = dim
        self.bit_width = bit_width
        self.device = torch.device(device)
        self.dtype = dtype

        self._mse_quant = TurboQuantMSE(dim, bit_width - 1, device, dtype)
        self._qjl_matrix = _generate_qjl_matrix(dim, self.device, self.dtype)

    @property
    def mse_quantizer(self) -> TurboQuantMSE:
        return self._mse_quant

    def quantize(self, x: torch.Tensor) -> QuantizedProd:
        original_shape = x.shape
        assert original_shape[-1] == self.dim
        flat = x.reshape(-1, self.dim).to(self.dtype)

        norms = torch.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-8)
        normalized = flat / safe_norms

        mse_q = self._mse_quant.quantize(normalized)
        mse_recon = self._mse_quant.dequantize(mse_q)

        residual = normalized - mse_recon
        residual_norms = torch.norm(residual, dim=-1, keepdim=True)

        projected = residual @ self._qjl_matrix.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return QuantizedProd(
            mse_indices=mse_q.indices.reshape(original_shape[:-1] + (self.dim,)),
            qjl_signs=qjl_signs.reshape(original_shape).to(torch.int8),
            norms=norms.squeeze(-1).reshape(original_shape[:-1]),
            residual_norms=residual_norms.squeeze(-1).reshape(original_shape[:-1]),
            bit_width=self.bit_width,
        )

    def dequantize(self, q: QuantizedProd) -> torch.Tensor:
        flat_indices = q.mse_indices.reshape(-1, self.dim).long()
        mse_q_inner = QuantizedMSE(
            indices=flat_indices.to(torch.int16),
            norms=torch.ones(flat_indices.shape[0], device=self.device, dtype=self.dtype),
            bit_width=self.bit_width - 1,
        )
        x_mse = self._mse_quant.dequantize(mse_q_inner)

        flat_signs = q.qjl_signs.reshape(-1, self.dim).to(self.dtype)
        import math

        scale = math.sqrt(math.pi / 2) / self.dim
        x_qjl = scale * (flat_signs @ self._qjl_matrix)
        residual_norms_flat = q.residual_norms.reshape(-1, 1).to(self.dtype)
        x_qjl = x_qjl * residual_norms_flat

        reconstructed = x_mse + x_qjl

        norms_flat = q.norms.reshape(-1, 1).to(self.dtype)
        reconstructed = reconstructed * norms_flat

        target_shape = list(q.mse_indices.shape[:-1]) + [self.dim]
        return reconstructed.reshape(target_shape)

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

    def inner_product_error(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """Measure inner product distortion between x and y after quantizing x."""
        true_ip = (x.to(self.dtype) * y.to(self.dtype)).sum(dim=-1)
        recon = self.round_trip(x)
        approx_ip = (recon * y.to(self.dtype)).sum(dim=-1)
        bias = (approx_ip - true_ip).mean()
        variance = ((approx_ip - true_ip) ** 2).mean()
        return {"bias": bias.item(), "variance": variance.item(), "mse": variance.item()}

    def storage_bytes_per_vector(self) -> int:
        mse_bits = self.dim * (self.bit_width - 1)
        qjl_bits = self.dim
        norm_bytes = 4
        residual_norm_bytes = 4
        return (mse_bits + qjl_bits + 7) // 8 + norm_bytes + residual_norm_bytes


class OutlierAwareQuantizer:
    """Mixed-precision quantizer that allocates more bits to outlier channels.

    Reproduces the paper's 2.5-bit and 3.5-bit configurations:
    - 2.5-bit: 32 outlier channels at 3 bits, 96 channels at 2 bits
    - 3.5-bit: 32 outlier channels at 4 bits, 96 channels at 3 bits
    """

    def __init__(
        self,
        dim: int,
        outlier_bits: int,
        normal_bits: int,
        n_outlier_channels: int = 32,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.n_outlier = min(n_outlier_channels, dim)
        self.n_normal = dim - self.n_outlier
        self.device = torch.device(device)
        self.dtype = dtype
        self.outlier_bits = outlier_bits
        self.normal_bits = normal_bits

        self._outlier_quant = TurboQuantMSE(self.n_outlier, outlier_bits, device, dtype)
        self._normal_quant = TurboQuantMSE(self.n_normal, normal_bits, device, dtype)

        self._outlier_mask: torch.Tensor | None = None

    @property
    def effective_bits(self) -> float:
        return (self.n_outlier * self.outlier_bits + self.n_normal * self.normal_bits) / self.dim

    def calibrate_outliers(self, sample: torch.Tensor) -> None:
        """Identify outlier channels from a sample of KV cache vectors."""
        flat = sample.reshape(-1, self.dim).to(self.dtype)
        channel_var = flat.var(dim=0)
        _, top_indices = torch.topk(channel_var, self.n_outlier)
        mask = torch.zeros(self.dim, dtype=torch.bool, device=self.device)
        mask[top_indices] = True
        self._outlier_mask = mask

    def quantize(self, x: torch.Tensor) -> dict:
        if self._outlier_mask is None:
            self.calibrate_outliers(x)
        assert self._outlier_mask is not None

        outlier_x = x[..., self._outlier_mask]
        normal_x = x[..., ~self._outlier_mask]

        return {
            "outlier": self._outlier_quant.quantize(outlier_x),
            "normal": self._normal_quant.quantize(normal_x),
            "outlier_mask": self._outlier_mask,
        }

    def dequantize(self, q: dict) -> torch.Tensor:
        outlier_recon = self._outlier_quant.dequantize(q["outlier"])
        normal_recon = self._normal_quant.dequantize(q["normal"])
        mask = q["outlier_mask"]

        batch_shape = outlier_recon.shape[:-1]
        result = torch.zeros(*batch_shape, self.dim, device=self.device, dtype=self.dtype)
        result[..., mask] = outlier_recon
        result[..., ~mask] = normal_recon
        return result

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))
