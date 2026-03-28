"""Lloyd-Max codebook solver for the Beta distribution induced by random rotation.

When a unit-norm vector in R^d is multiplied by a random orthogonal matrix,
each coordinate follows a scaled Beta distribution:

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

for x in [-1, 1].  In high dimensions this converges to N(0, 1/d).

This module solves the continuous 1-D k-means (Lloyd-Max) problem for this
distribution to produce optimal scalar quantization codebooks at bit-widths 1–4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln


def _log_beta_pdf_coeff(d: int) -> float:
    return gammaln(d / 2) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2)


def beta_pdf(x: float, d: int) -> float:
    if abs(x) >= 1.0:
        return 0.0
    log_coeff = _log_beta_pdf_coeff(d)
    exponent = (d - 3) / 2
    return math.exp(log_coeff + exponent * math.log(1 - x * x))


def _moment(a: float, b: float, d: int, power: int) -> float:
    """Compute integral of x^power * f_X(x) dx over [a, b]."""
    result, _ = quad(lambda x: (x**power) * beta_pdf(x, d), a, b, limit=200)
    return result


def _mass(a: float, b: float, d: int) -> float:
    return _moment(a, b, d, 0)


@dataclass(frozen=True)
class Codebook:
    centroids: np.ndarray  # shape (2^b,)
    boundaries: np.ndarray  # shape (2^b - 1,), sorted decision thresholds
    bit_width: int
    dimension: int
    mse_per_coord: float  # optimal scalar MSE cost C(f_X, b)


class BetaCodebook:
    """Builds optimal Lloyd-Max codebooks for the rotation-induced Beta distribution."""

    def __init__(self, dimension: int = 128, max_iters: int = 500, tol: float = 1e-12):
        self._dim = dimension
        self._max_iters = max_iters
        self._tol = tol

    @lru_cache(maxsize=16)
    def build(self, bit_width: int) -> Codebook:
        if bit_width < 1 or bit_width > 8:
            raise ValueError(f"bit_width must be in [1, 8], got {bit_width}")

        d = self._dim
        n_centroids = 1 << bit_width

        centroids = self._initial_centroids(n_centroids, d)
        boundaries = np.zeros(n_centroids - 1)

        for _ in range(self._max_iters):
            for i in range(n_centroids - 1):
                boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0

            new_centroids = np.zeros(n_centroids)
            for i in range(n_centroids):
                lo = boundaries[i - 1] if i > 0 else -1.0
                hi = boundaries[i] if i < n_centroids - 1 else 1.0
                mass = _mass(lo, hi, d)
                if mass < 1e-30:
                    new_centroids[i] = (lo + hi) / 2.0
                else:
                    new_centroids[i] = _moment(lo, hi, d, 1) / mass

            if np.max(np.abs(new_centroids - centroids)) < self._tol:
                centroids = new_centroids
                break
            centroids = new_centroids

        mse = self._compute_mse(centroids, boundaries, d)
        return Codebook(
            centroids=centroids,
            boundaries=boundaries,
            bit_width=bit_width,
            dimension=d,
            mse_per_coord=mse,
        )

    def _initial_centroids(self, n: int, d: int) -> np.ndarray:
        sigma = 1.0 / math.sqrt(d)
        quantiles = np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
        from scipy.stats import norm

        return norm.ppf(quantiles, scale=sigma)

    def _compute_mse(
        self, centroids: np.ndarray, boundaries: np.ndarray, d: int
    ) -> float:
        n = len(centroids)
        total = 0.0
        for i in range(n):
            lo = boundaries[i - 1] if i > 0 else -1.0
            hi = boundaries[i] if i < n - 1 else 1.0
            c = centroids[i]
            integrand_result, _ = quad(
                lambda x, c=c: ((x - c) ** 2) * beta_pdf(x, d), lo, hi, limit=200
            )
            total += integrand_result
        return total


REFERENCE_MSE = {
    1: 0.36,
    2: 0.117,
    3: 0.03,
    4: 0.009,
}


def validate_codebook(codebook: Codebook) -> dict:
    """Check codebook MSE against paper's reference values."""
    b = codebook.bit_width
    d = codebook.dimension
    total_mse = d * codebook.mse_per_coord
    ref = REFERENCE_MSE.get(b)
    result = {
        "bit_width": b,
        "dimension": d,
        "total_mse": total_mse,
        "mse_per_coord": codebook.mse_per_coord,
    }
    if ref is not None:
        result["reference_mse"] = ref
        result["ratio"] = total_mse / ref
    return result
