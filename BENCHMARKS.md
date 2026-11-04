# Benchmarks

## Environment (tested)
- CPU: AMD Ryzen 9 5900X (Zen 3), 12 threads pinned (`HYBRID_CPU_THREADS=12`, `OMP_NUM_THREADS=12`)
- GPU: AMD RX 6600 XT 8 GB, RADV Vulkan 1.4
- Python 3.11, PyTorch 2.12.0+cpu, transformers 5.13.0
- Candidate commit: 838218d (dcc5663 + ee1ead3 GPU-KV fix, 7c2998c _int_mm, 5261d2f vendored third_party, 838218d install.sh fix)
- Clean-checkout tvkvk.so sha256: see git — built fresh in /tmp worktree verification

## Checkpoints (SHA256)
- INT8 (CPU): weights/HybridAgent-8B-Agentic-INT8/model.safetensors `77e44ad29948998b09f702786129a04c585c55944b58ae909f3316dc0faccf08`
- FP32 (Vulkan): weights/HybridAgent-8B-Agentic/model.safetensors `2de074d380f4d7f82812245b8d3fefb075936366136f60d03c57dcc75f4d2c1a`

## Canonical harness
`test_100_latent_full.py` — original `HybridBackbone.forward()` + `HybridLink`, INT8 CPU layers, FP32 Vulkan layers, random 12-token prefill, 100 latent decode steps. KV length 12 -> 112. Sequential fresh processes only.

## Results (3 independent fresh processes per config, warm = drop first 10 steps)

| Config | Warm median (ms/step) | Mean | CPU layers | Vulkan layers | Finite |
|---|---|---|---|---|---|
| CPU-only (0 GPU) | 396 / 398 / 416 | ~403 | 394-414 | 0 | yes (3/3) |
| Hetero 2 GPU | 433 / 439 / 437 | ~436 | 369-373 | 65 | yes (3/3) |
| Hetero 4 GPU | 600 / 438 / 451 | ~496 | 346-513 | 91-94 | yes (3/3) |

- Latent throughput: CPU-only ~2.5 steps/s, 2-GPU ~2.3 steps/s.
- RSS after load: 18.6-20.4 GB (harness retains INT8 state_dict; swap 0-6 GB observed in one run).
- Vulkan per-layer cost (~18-32 ms) exceeds CPU INT8 per-layer cost (~11 ms) after the `_int_mm` optimization, so CPU-only is the fastest reproducible configuration at this checkpoint. 2 vs 4 GPU layers is within run-to-run variation (438 vs 438-451 excluding one 600 ms outlier).

## Historical (not production-comparable)
- FP32-dequant INT8 path: CPU-only 9044 ms/step, hetero-5 6760 ms/step (before `_int_mm`).
- Stock bf16 engine (`tests/run_e2e_vk.py`): ~0.15-0.31 latent steps/s on real prompts.
