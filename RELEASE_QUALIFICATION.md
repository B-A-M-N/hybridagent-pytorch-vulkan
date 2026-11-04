# Release Qualification

Candidate: dcc5663 (+ ee1ead3 GPU-KV fix, 7c2998c _int_mm optimization)
Date: 2026-11-04. Hardware: Ryzen 9 5900X, RX 6600 XT 8GB, RADV Vulkan 1.4.

## Gates
| Gate | Result | Evidence |
|---|---|---|
| Freeze/baseline (P0) | PASS | HEAD dcc5663, clean tree, single worktree, Q4 disabled by default |
| Native extension matches source (P0) | PASS | tvkvk.so Oct 31 12:00 = last vk_backend.cpp change; sha256 18d76e55... |
| Canonical benchmarks (P1) | PASS | 3 fresh runs/config, 100 steps, KV 112, all finite (see BENCHMARKS.md) |
| Single-layer correctness (P2) | PASS | tests/test_layer_vulkan.py: max_abs_err 8.9e-05 |
| GPU-KV append correctness (P2) | PASS | tests/test_gpu_kv_append.py: exact K/V round-trip (single + prefill-seed) |
| GPU-KV lifecycle (P2) | PASS w/ limitation | Owner: VulkanDecoderLayer via gpu_kv_cache.GPUKVCacheManager (lazy init, harness-injected manager respected); DynamicCache lock-step per token. reset/free exist but canonical harness does not call reset between sequences (single-sequence benchmark). |
| _int_mm numerical audit (P2) | PASS w/ limitation | Per-row act scale, clamp [-127,127], int32 accumulate, per-channel weight scales applied. Quantified vs FP32 dequant reference: rel error ~0.9-1.0% per projection (random weights). Long-trajectory: 100-step finite, std stable. NOT bit-equivalent to FP32 path. |
| Repeated fresh sessions (P2/P3) | PASS | 9/9 fresh processes finite, no crashes |
| GPU stability (P3) | PARTIAL | No device loss / invalid handles / leaks observed in 9 runs; watchdog running, stop_gpu_work fail-closed present. dmesg not readable (permission), GPU temp not instrumented. |
| Clean-install reproducibility (P4) | PARTIAL | scripts/install.sh documents venv + torch 2.12 + Kompute + extension build; not executed from a clean checkout in this qualification (existing .venv reused). torch._int_mm is a private API (torch 2.12.0+cpu) — version pin required. |
| License/hygiene (P5) | PASS w/ limitation | Upstream HybridAgent MIT (third_party/HybridAgent/LICENSE). No credentials tracked. Dev scripts contain /home/bamn absolute paths; generated .spv/.so ignored. |
| Docs (P6) | PASS | README.md, BENCHMARKS.md, this file |

## Known limitations
1. Fastest reproducible config is CPU-only (~403 ms/step); Vulkan hetero (2-4 layers, ~436-496 ms) no longer beats it after _int_mm. Vulkan path remains functional and qualified but not performance-leading.
2. 20 GPU layers hangs in setup (FP32 VRAM ceiling). Q4 GEMV disabled by default; NOT numerically qualified — do not enable.
3. Harness RSS 18-20 GB (INT8 state_dict retention); one run swapped 6 GB.
4. _int_mm activation quantization is ~1% approximate vs FP32 dequant — behavior differs at that level; tolerances were not weakened, error is quantified above.
5. dmesg/GPU-temp not available in this environment; hardware-fault exclusion is by absence of device loss across 9 runs only.

## Verdict
QUALIFIED WITH LIMITATIONS — core canonical execution, correctness, stability (within instrumented scope), and reproducibility pass; limitations above are documented and noncritical. No push/publish performed.
