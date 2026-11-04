"""Regression test: GPUKVCache.append_kv must write exact K/V values.

Before the fix, append_kv was a stub (downloaded to CPU, never wrote to the
GPU buffers) and a later slice-based version used slice's SOURCE-offset
semantics as if it were a destination offset, corrupting the cache.
Attention then read uninitialized/garbage memory and produced NaN in the
canonical full-model path (test_100_latent_full.py).

This test appends known tensors at successive positions and asserts the
exported cache matches exactly. Run:
    .venv/bin/python tests/test_gpu_kv_append.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch_vulkan_ext as tvk
from backend.hybrid_offload import _Buf
from gpu_kv_cache import GPUKVCache


def main():
    tvk.init(0)
    cache = GPUKVCache(layer_idx=0, num_kv_heads=8, head_dim=128, max_capacity=1024)
    S, KVD = 1, 8 * 128
    expected_k, expected_v = [], []
    for _ in range(3):
        k = torch.randn(S, 8, 128)
        v = torch.randn(S, 8, 128)
        expected_k.append(k)
        expected_v.append(v)
        cache.append_kv(_Buf(k), _Buf(v), S, KVD)

    assert cache.valid_len == 3, f"valid_len={cache.valid_len}"
    k_cpu, v_cpu = cache.to_cpu(3)
    for i in range(3):
        ek = float((k_cpu[0, :, i] - expected_k[i][0]).abs().max())
        ev = float((v_cpu[0, :, i] - expected_v[i][0]).abs().max())
        assert ek == 0.0, f"K mismatch at pos {i}: max err {ek}"
        assert ev == 0.0, f"V mismatch at pos {i}: max err {ev}"
    assert bool(torch.isfinite(k_cpu).all()) and bool(torch.isfinite(v_cpu).all())

    # Multi-position (prefill seed) append
    cache2 = GPUKVCache(layer_idx=1, num_kv_heads=8, head_dim=128, max_capacity=1024)
    S2 = 4
    k = torch.randn(S2, 8, 128)
    v = torch.randn(S2, 8, 128)
    cache2.append_kv(_Buf(k), _Buf(v), S2, KVD)
    kc, vc = cache2.to_cpu(S2)
    for i in range(S2):
        assert float((kc[0, :, i] - k[i]).abs().max()) == 0.0, f"seed K pos {i}"
        assert float((vc[0, :, i] - v[i]).abs().max()) == 0.0, f"seed V pos {i}"

    cache.free()
    cache2.free()
    print("[PASS] GPUKVCache.append_kv writes exact K/V (single + prefill-seed)")


if __name__ == "__main__":
    main()
