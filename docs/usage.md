# Usage

CPU-only (fastest reproducible config):
```bash
HYBRID_GPU_LAYERS=0 HYBRID_STEPS=100 PROFILE_LATENT=1 \
  .venv/bin/python -u test_100_latent_full.py
```
Heterogeneous (2 Vulkan layers):
```bash
HYBRID_GPU_LAYERS=2 HYBRID_STEPS=100 PROFILE_LATENT=1 \
  .venv/bin/python -u test_100_latent_full.py
```
`HYBRID_GPU_LAYERS` selects contiguous first-N layers for Vulkan
placement. Default CPU threads: 12 (`HYBRID_CPU_THREADS`).
Small smoke example: `python examples/cpu_only_smoke.py`.

Original-engine end-to-end: `TVK_GPU_LAYERS=4 .venv/bin/python tests/run_e2e_vk.py`

Q4 GEMV (`use_q4`) is disabled by default and NOT numerically qualified —
do not enable. The GPU watchdog (`gpu_watchdog.py`, fail-closed) must be
running for GPU work; one GPU workload at a time.

Tests: `./scripts/run_tests.sh` (single-layer Vulkan, GPU-KV regression,
stock latent).
