# Installation

Requirements (tested): Python 3.11, PyTorch 2.12.0+cpu (pinned —
`torch._int_mm` is a private API), transformers 5.13.0, Vulkan loader +
RADV driver, glslangValidator, CMake. Hardware: Ryzen 9 5900X +
RX 6600 XT 8GB (only tested configuration).

```bash
git clone https://github.com/B-A-M-N/hybridagent-pytorch-vulkan.git
cd hybridagent-pytorch-vulkan
bash scripts/install.sh
```

`install.sh` creates `.venv`, installs the pinned stack, compiles 26
shaders, builds `tvkvk.so` (the raw-Vulkan backend the canonical path
uses), and installs `torch_vulkan_ext` + vendored HybridAgent.
Legacy Kompute/torch-vulkan is optional and skipped if its (lost)
sources are incomplete.

Checkpoints are not distributed. Obtain the upstream HybridAgent 8B
FP32 checkpoint, then prepare the INT8 checkpoint with per-channel
symmetric quantization (weight_int8 + per-channel scales as loaded by
`int8_linear_v3.Int8Linear`). Place them under `weights/`:
`HybridAgent-8B-Agentic/` (FP32) and `HybridAgent-8B-Agentic-INT8/`.
Hashes are recorded in BENCHMARKS.md.
