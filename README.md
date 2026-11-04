# HybridAgent — Heterogeneous PyTorch CPU INT8 / Vulkan Execution

A PyTorch implementation of HybridAgent that keeps the original
`HybridGenerationEngine` and `HybridLink` unchanged, and changes only
*where* the Qwen3 backbone's transformer layers execute: INT8-quantized
layers on CPU via native `torch._int_mm`, and selected FP32 layers on
an AMD GPU via a custom Vulkan backend (RADV). A GPU-local KV cache
(`gpu_kv_cache.py`, owner: `VulkanDecoderLayer`) keeps K/V history
resident on the device, with per-token lock-step sync to the CPU
`DynamicCache` for mixed CPU/GPU stacks.

## Tested hardware
- AMD Ryzen 9 5900X (Zen 3), 12 CPU threads
- AMD RX 6600 XT 8 GB, RADV Vulkan 1.4
- No ROCm, GGUF, or replacement inference engine. No other GPU has been tested.

## Prerequisites
- Python 3.11, PyTorch 2.12.0+cpu (pinned — `torch._int_mm` is a private API)
- transformers 5.13.0, safetensors, Vulkan loader + RADV driver
- HybridAgent 8B checkpoint (FP32) + INT8 conversion (see `scripts/install.sh`)

## Install
```bash
bash scripts/install.sh   # venv, pinned stack, Kompute, torch_vulkan_ext, shaders
```

## Usage
Canonical benchmark / smoke path (deterministic, 100 latent steps):
```bash
HYBRID_GPU_LAYERS=2 HYBRID_STEPS=100 PROFILE_LATENT=1 \
  .venv/bin/python -u test_100_latent_full.py
```
- `HYBRID_GPU_LAYERS=0` — CPU INT8 only (fastest reproducible config, ~403 ms/step)
- `HYBRID_GPU_LAYERS=2` — heterogeneous (~436 ms/step)
- Default CPU threads: 12 (`HYBRID_CPU_THREADS`). 6/16/24 measured worse.
- Q4 GEMV (`use_q4`) is **disabled by default and not numerically qualified** — do not enable.

Original-engine end-to-end (stock `HybridGenerationEngine` + Vulkan backbone):
```bash
TVK_GPU_LAYERS=4 .venv/bin/python tests/run_e2e_vk.py
```

## Safety
`gpu_watchdog.py` (independent process, fail-closed `stop_gpu_work`) and
`backend/torch_vulkan_ext/.../safety_cross_process.py` guard all native
Vulkan execution. One GPU workload at a time; no auto-retry after device loss.

## Limitations
See `RELEASE_QUALIFICATION.md` and `BENCHMARKS.md` for the full gate
report, measured numbers, and known limitations (VRAM ceiling at high
GPU-layer counts, ~1% `_int_mm` activation-quantization error vs FP32,
harness RAM footprint, dev scripts with absolute paths).

## Upstream
Original HybridAgent code: `third_party/HybridAgent/` (MIT License).
This repository's Vulkan backend, heterogeneous backbone, GPU KV cache,
and INT8 CPU path are the additions.

## Release status
Clean-checkout install is currently BLOCKED: `third_party/` (HybridAgent,
torch-vulkan) is not tracked in git and `scripts/install.sh` does not
fetch it — see `RELEASE_QUALIFICATION.md` (P4). Building from this
working tree (with vendored `third_party/` present) works and is what
all benchmarks above were measured on.
