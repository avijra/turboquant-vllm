"""Benchmark TurboQuant distortion on synthetic KV cache tensors.

Generates KV cache-like activations (Gaussian and heavy-tailed) and measures:
- MSE distortion vs bit-width
- Inner product error (bias + variance) for TurboQuant_prod
- Compression ratio
- Comparison with uniform scalar quantization baseline

Run:  python -m turboquant.benchmark
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from turboquant.codebook import BetaCodebook, validate_codebook
from turboquant.quantizer import OutlierAwareQuantizer, TurboQuantMSE, TurboQuantProd


@dataclass
class BenchmarkResult:
    name: str
    bit_width: float
    mse: float
    compression_ratio: float
    throughput_vecs_per_sec: float | None = None
    bias: float | None = None
    variance: float | None = None


def generate_kv_cache_tensors(
    n_tokens: int = 4096,
    n_heads: int = 8,
    head_dim: int = 128,
    distribution: str = "gaussian",
    device: str = "cpu",
) -> torch.Tensor:
    """Generate synthetic KV cache activation tensors."""
    shape = (n_tokens, n_heads, head_dim)
    if distribution == "gaussian":
        x = torch.randn(shape, device=device)
    elif distribution == "heavy_tail":
        x = torch.randn(shape, device=device)
        outlier_mask = torch.rand(shape, device=device) < 0.05
        x[outlier_mask] *= 10.0
    elif distribution == "uniform":
        x = torch.rand(shape, device=device) * 2 - 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return x


def benchmark_mse_quantizer(
    x: torch.Tensor,
    bit_widths: list[int] | None = None,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    if bit_widths is None:
        bit_widths = [1, 2, 3, 4]

    results = []
    n_tokens, n_heads, head_dim = x.shape
    fp16_bytes = head_dim * 2

    for b in bit_widths:
        quant = TurboQuantMSE(head_dim, bit_width=b, device=device)

        flat = x.reshape(-1, head_dim)

        start = time.perf_counter()
        mse_val = quant.mse(flat).item()
        elapsed = time.perf_counter() - start

        n_vecs = flat.shape[0]
        throughput = n_vecs / elapsed if elapsed > 0 else float("inf")

        quant_bytes = quant.storage_bytes_per_vector()
        compression = fp16_bytes / quant_bytes

        results.append(BenchmarkResult(
            name=f"TurboQuant_mse b={b}",
            bit_width=b,
            mse=mse_val,
            compression_ratio=compression,
            throughput_vecs_per_sec=throughput,
        ))

    return results


def benchmark_prod_quantizer(
    x: torch.Tensor,
    y: torch.Tensor,
    bit_widths: list[int] | None = None,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    if bit_widths is None:
        bit_widths = [2, 3, 4]

    results = []
    head_dim = x.shape[-1]
    fp16_bytes = head_dim * 2

    for b in bit_widths:
        quant = TurboQuantProd(head_dim, bit_width=b, device=device)

        flat_x = x.reshape(-1, head_dim)
        flat_y = y.reshape(-1, head_dim)

        ip_result = quant.inner_product_error(flat_x, flat_y)

        quant_bytes = quant.storage_bytes_per_vector()
        compression = fp16_bytes / quant_bytes

        results.append(BenchmarkResult(
            name=f"TurboQuant_prod b={b}",
            bit_width=b,
            mse=ip_result["mse"],
            compression_ratio=compression,
            bias=ip_result["bias"],
            variance=ip_result["variance"],
        ))

    return results


def benchmark_outlier_aware(
    x: torch.Tensor,
    configs: list[tuple[int, int, int]] | None = None,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark outlier-aware mixed-precision configurations.

    configs: list of (outlier_bits, normal_bits, n_outlier_channels)
    """
    if configs is None:
        configs = [
            (3, 2, 32),  # effective 2.5 bits
            (4, 3, 32),  # effective 3.5 bits
        ]

    results = []
    head_dim = x.shape[-1]
    fp16_bytes = head_dim * 2

    for outlier_b, normal_b, n_outlier in configs:
        quant = OutlierAwareQuantizer(
            head_dim, outlier_b, normal_b, n_outlier, device=device
        )
        flat = x.reshape(-1, head_dim)
        recon = quant.round_trip(flat)
        mse = ((flat.float() - recon) ** 2).sum(dim=-1).mean().item()

        eff_bits = quant.effective_bits
        quant_bytes = int(head_dim * eff_bits / 8) + 8
        compression = fp16_bytes / quant_bytes

        results.append(BenchmarkResult(
            name=f"OutlierAware {outlier_b}/{normal_b}bit ({n_outlier} outliers)",
            bit_width=eff_bits,
            mse=mse,
            compression_ratio=compression,
        ))

    return results


def baseline_uniform_quantize(x: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Simple uniform scalar quantization baseline (no rotation)."""
    n_levels = 1 << bit_width
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-8)
    normalized = (x - x_min) / x_range
    quantized = torch.clamp(torch.round(normalized * (n_levels - 1)), 0, n_levels - 1)
    return quantized / (n_levels - 1) * x_range + x_min


def benchmark_baseline(
    x: torch.Tensor,
    bit_widths: list[int] | None = None,
) -> list[BenchmarkResult]:
    if bit_widths is None:
        bit_widths = [2, 4, 8]

    results = []
    head_dim = x.shape[-1]
    fp16_bytes = head_dim * 2

    for b in bit_widths:
        flat = x.reshape(-1, head_dim).float()
        recon = baseline_uniform_quantize(flat, b)
        mse = ((flat - recon) ** 2).sum(dim=-1).mean().item()
        quant_bytes = (head_dim * b + 7) // 8 + 4 * 2
        compression = fp16_bytes / quant_bytes

        results.append(BenchmarkResult(
            name=f"Uniform baseline b={b}",
            bit_width=b,
            mse=mse,
            compression_ratio=compression,
        ))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    header = f"{'Method':<45} {'Bits':>5} {'MSE':>12} {'Compress':>9}"
    extras = ""
    if any(r.bias is not None for r in results):
        extras = f" {'Bias':>10} {'Var':>12}"
    if any(r.throughput_vecs_per_sec is not None for r in results):
        extras += f" {'Throughput':>12}"
    print(header + extras)
    print("-" * len(header + extras))

    for r in results:
        line = f"{r.name:<45} {r.bit_width:>5.1f} {r.mse:>12.6f} {r.compression_ratio:>8.2f}x"
        if r.bias is not None:
            line += f" {r.bias:>10.6f} {r.variance:>12.8f}"
        elif any(r2.bias is not None for r2 in results):
            line += f" {'':>10} {'':>12}"
        if r.throughput_vecs_per_sec is not None:
            line += f" {r.throughput_vecs_per_sec:>10.0f}/s"
        elif any(r2.throughput_vecs_per_sec is not None for r2 in results):
            line += f" {'':>12}"
        print(line)


def run_codebook_validation(dim: int = 128) -> None:
    print(f"\n=== Codebook validation (d={dim}) ===")
    builder = BetaCodebook(dimension=dim)
    for b in range(1, 5):
        cb = builder.build(b)
        info = validate_codebook(cb)
        ref_str = ""
        if "reference_mse" in info:
            ref_str = f" (paper: {info['reference_mse']:.4f}, ratio: {info['ratio']:.4f})"
        print(f"  b={b}: MSE/vector = {info['total_mse']:.6f}{ref_str}")
        print(f"         centroids = {cb.centroids}")


def main() -> None:
    device = "cpu"
    head_dim = 128
    n_tokens = 1024
    n_heads = 8

    run_codebook_validation(dim=head_dim)

    for dist in ["gaussian", "heavy_tail"]:
        print(f"\n=== Distribution: {dist} (n={n_tokens}, heads={n_heads}, d={head_dim}) ===")

        x = generate_kv_cache_tensors(n_tokens, n_heads, head_dim, dist, device)

        print("\n--- TurboQuant MSE ---")
        mse_results = benchmark_mse_quantizer(x, device=device)
        print_results(mse_results)

        print("\n--- TurboQuant Inner Product ---")
        y = generate_kv_cache_tensors(n_tokens, n_heads, head_dim, "gaussian", device)
        prod_results = benchmark_prod_quantizer(x, y, device=device)
        print_results(prod_results)

        print("\n--- Outlier-Aware Mixed Precision ---")
        outlier_results = benchmark_outlier_aware(x, device=device)
        print_results(outlier_results)

        print("\n--- Uniform Baseline ---")
        baseline_results = benchmark_baseline(x)
        print_results(baseline_results)


if __name__ == "__main__":
    main()
