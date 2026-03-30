"""Apply all TurboQuant patches to a fresh vLLM 0.18.0 installation."""

import os

VLLM = "/usr/local/lib/python3.11/dist-packages/vllm"


def patch_file(path, old, new, label):
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        if new.strip().split('\n')[0].strip() in content:
            print(f"  SKIP (already applied): {label}")
            return True
        print(f"  FAIL: anchor not found for {label}")
        return False
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"  OK: {label}")
    return True


print("=== 1. Patching cache.py (CacheDType + validator) ===")
patch_file(
    f"{VLLM}/config/cache.py",
    '    "fp8_ds_mla",\n]',
    '    "fp8_ds_mla",\n    "turboquant",\n]',
    "Add turboquant to CacheDType",
)
patch_file(
    f"{VLLM}/config/cache.py",
    '        return cache_dtype',
    '''        if cache_dtype == "turboquant":
            logger.info(
                "Using TurboQuant (rotation + optimal codebook) to store "
                "kv cache as 4-bit packed indices. Gives 3.76x compression "
                "vs fp16 with near-optimal distortion. No calibration needed."
            )
        return cache_dtype''',
    "Add turboquant log message",
)

print("\n=== 2. Patching torch_utils.py (dtype mapping) ===")
patch_file(
    f"{VLLM}/utils/torch_utils.py",
    '    "fp8_ds_mla": torch.uint8,\n}',
    '    "fp8_ds_mla": torch.uint8,\n    "turboquant": torch.int8,\n}',
    "Add turboquant dtype mapping",
)

print("\n=== 3. Patching attention/backend.py (is_quantized_kv_cache) ===")
patch_file(
    f"{VLLM}/v1/attention/backend.py",
    'return kv_cache_dtype.startswith("fp8")',
    'return kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "turboquant"',
    "Recognize turboquant as quantized KV cache",
)

print("\n=== 4. Patching attention.py (get_kv_cache_spec) ===")
patch_file(
    f"{VLLM}/model_executor/layers/attention/attention.py",
    """            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=spec_head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )""",
    """            cache_cfg = vllm_config.cache_config
            if cache_cfg.cache_dtype == "turboquant":
                from vllm.v1.attention.ops.turboquant_cache import (
                    TURBOQUANT_BIT_WIDTH,
                    packed_dim,
                )
                spec_head_size = packed_dim(
                    self.head_size, TURBOQUANT_BIT_WIDTH,
                )
            else:
                spec_head_size = spec_head_size
            spec_head_size_v = self.head_size_v
            if cache_cfg.cache_dtype == "turboquant" and spec_head_size_v is not None:
                spec_head_size_v = packed_dim(
                    spec_head_size_v, TURBOQUANT_BIT_WIDTH,
                )
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=spec_head_size,
                head_size_v=spec_head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )""",
    "Pack head_size for turboquant in FullAttentionSpec",
)

print("\n=== 5. Patching kv_cache.py (process_weights_after_loading) ===")
patch_file(
    f"{VLLM}/model_executor/layers/quantization/kv_cache.py",
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If the model is not FP8 quantized, but has kv_cache_dtype FP8,""",
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "kv_cache_dtype", "") == "turboquant":
            layer._k_scale.copy_(1.0)
            layer._v_scale.copy_(1.0)
            layer._k_scale_float = 1.0
            layer._v_scale_float = 1.0
            layer._q_scale.copy_(1.0)
            layer._q_scale_float = 1.0
            layer._prob_scale.copy_(1.0)
            del layer.k_scale
            del layer.v_scale
            del layer.q_scale
            del layer.prob_scale
            return
        # If the model is not FP8 quantized, but has kv_cache_dtype FP8,""",
    "Skip fp8 logic for turboquant",
)

print("\n=== 6. Patching triton_attn.py ===")
triton_path = f"{VLLM}/v1/attention/backends/triton_attn.py"
with open(triton_path, "r") as f:
    tc = f.read()

# Add imports at top
if "turboquant_cache" not in tc:
    tc = tc.replace(
        "from vllm.v1.attention.backend import (",
        "from vllm.v1.attention.ops.turboquant_cache import (\n"
        "    TurboQuantState,\n"
        "    turboquant_dequantize_for_attention,\n"
        "    turboquant_reshape_and_cache,\n"
        ")\n"
        "from vllm.v1.attention.backend import (",
        1,
    )
    print("  OK: Added turboquant imports to triton_attn.py")

# Add _tq_state init
if "_tq_state" not in tc:
    tc = tc.replace(
        "        self.vllm_flash_attn_version",
        "        self._tq_state: TurboQuantState | None = None\n"
        "        if self.kv_cache_dtype == 'turboquant':\n"
        "            self._tq_state = TurboQuantState(\n"
        "                head_size=self.head_size,\n"
        "                num_kv_heads=self.num_kv_heads,\n"
        "                device=torch.device('cuda'),\n"
        "            )\n"
        "        self.vllm_flash_attn_version",
        1,
    )
    print("  OK: Added _tq_state init to triton_attn.py")

# Add TQ dequantize in forward
if "turboquant_dequantize_for_attention" not in tc:
    tc = tc.replace(
        """        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)""",
        """        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype == "turboquant" and self._tq_state is not None:
            tq = self._tq_state
            nb, bs = key_cache.shape[0], key_cache.shape[1]
            k_norms = tq.get_norm_cache(nb, bs, kv_idx=0)
            v_norms = tq.get_norm_cache(nb, bs, kv_idx=1)
            query_for_attn, key_cache, value_cache = (
                turboquant_dequantize_for_attention(
                    key_cache, value_cache,
                    tq, k_norms, v_norms,
                    query[:num_actual_tokens],
                    target_dtype=query.dtype,
                )
            )
            query = torch.cat([
                query_for_attn,
                query[num_actual_tokens:],
            ], dim=0) if num_actual_tokens < query.shape[0] else (
                query_for_attn
            )""",
        1,
    )
    print("  OK: Added TQ dequantize in triton_attn forward")

# Add inverse rotation after unified_attention
if "rotate_inverse" not in tc:
    tc = tc.replace(
        """            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output""",
        """            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        if self.kv_cache_dtype == "turboquant" and self._tq_state is not None:
            out_slice = output[:num_actual_tokens]
            orig_shape = out_slice.shape
            flat = out_slice.reshape(-1, self._tq_state.head_size)
            output[:num_actual_tokens] = (
                self._tq_state.rotate_inverse(flat)
                .to(out_slice.dtype)
                .reshape(orig_shape)
            )

        return output""",
        1,
    )
    print("  OK: Added inverse rotation after attention in triton_attn")

# Add TQ write path in do_kv_cache_update
if "turboquant_reshape_and_cache" not in tc:
    tc = tc.replace(
        """        # Reshape the input keys and values and store them in the cache.
        if self.kv_cache_dtype.startswith("fp8"):""",
        """        # Reshape the input keys and values and store them in the cache.
        if self.kv_cache_dtype == "turboquant" and self._tq_state is not None:
            tq = self._tq_state
            nb, bs = key_cache.shape[0], key_cache.shape[1]
            k_norms = tq.get_norm_cache(nb, bs, kv_idx=0)
            v_norms = tq.get_norm_cache(nb, bs, kv_idx=1)
            turboquant_reshape_and_cache(
                key, value, key_cache, value_cache,
                slot_mapping, tq, k_norms, v_norms,
            )
        elif self.kv_cache_dtype.startswith("fp8"):""",
        1,
    )
    print("  OK: Added TQ write path in triton_attn do_kv_cache_update")

# Add turboquant to supported_kv_cache_dtypes
if '"turboquant"' not in tc.split("supported_kv_cache_dtypes")[-1].split("]")[0]:
    tc = tc.replace(
        '        "fp8_e5m2",\n    ]',
        '        "fp8_e5m2",\n        "turboquant",\n    ]',
        1,
    )
    print("  OK: Added turboquant to supported_kv_cache_dtypes")

with open(triton_path, "w") as f:
    f.write(tc)

print("\n=== 7. Patching attention/backend.py (supported_kv_cache_dtypes) ===")
patch_file(
    f"{VLLM}/v1/attention/backend.py",
    '    "bfloat16",\n    ]',
    '    "bfloat16",\n    "turboquant",\n    ]',
    "Add turboquant to base AttentionBackend supported dtypes",
)

print("\n=== All patches applied ===")
