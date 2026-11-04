# Architecture (summary)

- `backend/hybrid_offload.py` — `HybridBackbone` (per-layer CPU/Vulkan
  placement) + `VulkanDecoderLayer` (Qwen3 layer on Vulkan, owner of the
  GPU-local KV cache). Original `HybridGenerationEngine`/`HybridLink`
  untouched (vendored in `third_party/HybridAgent/`, MIT).
- `int8_linear_v3.py` — CPU INT8 projections via `torch._int_mm`
  (per-row activation INT8 quant, int32 GEMV, per-channel scales).
- `gpu_kv_cache.py` — GPU-resident K/V (`GPUKVCache`/`GPUKVCacheManager`),
  written via the `kv_append` shader kernel; lock-step sync to the CPU
  `DynamicCache` per token.
- `backend/torch_vulkan_ext/` — raw-Vulkan backend (`tvkvk.so` built from
  `vk_backend.cpp` + `vk_cabi.cpp` via CMake; Python via ctypes).
  `third_party/torch-vulkan/` (Kompute-based) is legacy, unused by the
  canonical path, and optional in `scripts/install.sh`.
- `gpu_watchdog.py` — independent-process watchdog, fail-closed
  `stop_gpu_work`; `safety_cross_process.py` guards native Vulkan execution.

See `RELEASE_QUALIFICATION.md` for gates and `BENCHMARKS.md` for numbers.
