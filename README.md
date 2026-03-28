# TurboQuant — Near-Optimal KV Cache Quantization for vLLM

Implementation of [TurboQuant: Online Vector Quantization for Quantized KV Cache](https://arxiv.org/abs/2504.19874) integrated into [vLLM](https://github.com/vllm-project/vllm).

TurboQuant compresses the KV cache to **2 bits per coordinate** using data-oblivious vector quantization — no calibration data, no checkpoint changes, one CLI flag.

## Results

| Metric | Standard vLLM (fp16) | + TurboQuant (2-bit) |
|---|---|---|
| KV cache per token (Llama-3-8B) | 128 KB | **18 KB** |
| Compression ratio | 1× | **7.1×** |
| Max tokens in 8 GB budget | 65K | **466K** |
| Concurrent 4K chats on A100-80GB | 80 | **568** |
| Calibration required | N/A | **None** |
| Checkpoint changes | N/A | **None** |

## How It Works

1. **Random orthogonal rotation** — Each KV vector is rotated by a fixed matrix Π, making coordinate distributions follow a Beta distribution regardless of the model.
2. **Optimal scalar quantizer** — Lloyd-Max codebook computed analytically for the Beta distribution. No training data needed.
3. **Sub-byte packing** — 4 indices packed per `uint8` byte at 2-bit precision.
4. **Query-rotation trick** — Instead of inverse-rotating every cached KV entry at read time, the query is rotated once by Π. Same dot products, far less work.

## Project Structure

```
turboquant/              # Standalone library (PyTorch + optional Triton)
├── codebook.py          # Lloyd-Max codebook solver for Beta distribution
├── quantizer.py         # TurboQuantMSE, TurboQuantProd, OutlierAwareQuantizer
├── triton_kernels.py    # GPU kernels with PyTorch fallback
└── benchmark.py         # Distortion & throughput benchmarks

vllm_integration/        # Modified vLLM files (drop-in replacements)
└── vllm/
    ├── config/cache.py                            # Added "turboquant" to CacheDType
    ├── utils/torch_utils.py                       # Added dtype mapping
    ├── v1/attention/backend.py                    # Extended is_quantized_kv_cache
    ├── v1/attention/ops/turboquant_cache.py       # Core: codebook, rotation, pack/unpack, Triton kernels
    ├── v1/attention/backends/triton_attn.py       # Triton backend integration
    ├── v1/attention/backends/flash_attn.py        # FlashAttention backend integration
    └── model_executor/layers/
        ├── quantization/kv_cache.py               # BaseKVCacheMethod bypass for turboquant
        └── attention/attention.py                 # Cache spec with packed dimensions

tests/                   # Unit + integration tests
comparison.html          # Interactive comparison dashboard
```

## Quick Start

### Standalone Library

```bash
pip install -e .

python -c "
from turboquant import TurboQuantMSE
import torch

q = TurboQuantMSE(dimension=128, bit_width=2)
x = torch.randn(32, 128)
compressed = q.quantize(x)
reconstructed = q.dequantize(compressed)
print(f'MSE: {((x - reconstructed)**2).mean():.6f}')
"
```

### vLLM Integration

Copy the modified files into your vLLM installation, then launch with one flag:

```bash
vllm serve meta-llama/Llama-3-8B --kv-cache-dtype turboquant
```

Works with both **Triton** and **FlashAttention** backends. Compatible with any transformer model — no per-model calibration or fine-tuning required.

## Reference

```bibtex
@article{shahverdi2025turboquant,
  title={TurboQuant: Online Vector Quantization for Quantized KV Cache},
  author={Shahverdi, Ehsan and Modarressi, Ali and Pilehvar, Mohammad Taher},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```
