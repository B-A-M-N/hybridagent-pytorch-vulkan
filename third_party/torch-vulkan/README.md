# torch-vulkan

A from-scratch **Vulkan compute backend for PyTorch** — registers a real `"vulkan"`
device via PyTorch's `PrivateUse1` dispatch key and runs ops as hand-written SPIR-V
compute shaders. Built to do **GGUF-quantized LLM inference on AMD RDNA 3.5
(gfx1150 / Radeon 890M)** without ROCm.

```python
import torch
import torch_vulkan

torch_vulkan.is_available()      # -> True
torch_vulkan.device_name()       # -> "Vulkan Device 0"

# Create on CPU, then move to the Vulkan device:
a = torch.randn(64, 64).to("vulkan")
b = torch.randn(64, 64).to("vulkan")
c = torch.mm(a, b)               # dispatches matmul.spv on the GPU
c.cpu()                          # bring the result back
```

> Note: build a tensor on CPU and `.to("vulkan")` it, or use
> `torch.empty(..., device="vulkan")` / `torch.tensor(data, device="vulkan")`.
> The `torch.randn(..., device="vulkan")` constructor and a `.vulkan()` method are
> not wired yet (see Limitations).

## Why this exists

ROCm support on consumer AMD APUs is fragile (gfx1150 isn't officially supported;
the usual path is spoofing the arch as gfx1100). Vulkan compute is available on
*every* modern GPU and doesn't care about the vendor's support matrix. This backend
treats Vulkan as a first-class PyTorch device so quantized models can run on hardware
ROCm won't touch.

## What works today

Ops wired end-to-end through PyTorch dispatch (`PrivateUse1` → SPIR-V shader):

| aten op | shader | Notes |
|---------|--------|-------|
| `mm` / `addmm` | `matmul_tiled` | fp32; `addmm` composes mm+mul+add |
| `mm_q4` (custom op) | `matmul_q4` | **Q4_K** GGUF superblock, GPU-side dequant |
| `add.Tensor` / `mul.Tensor` | `add` / `mul` | elementwise |
| `mul.Scalar` / `div.Tensor` | `scalar_mul` | scalar ops |
| `relu` / `gelu` | `relu` / `gelu` | activations |
| `_softmax` | `softmax` | |
| `native_layer_norm` | `layer_norm` | |
| `embedding` | `embedding` | |
| `scaled_dot_product_attention` | `attention` | |

Plus an algorithm/sequence cache that reuses pipelines across dispatches
(`torch_vulkan.cache_stats()` reports hits/misses).

### Verified (measured on Radeon 890M / RADV STRIX1, PyTorch 2.12)

`tests/test_algo_cache.py` passes — each op diffed against a CPU reference within
`rtol=atol=1e-3`:

| Op | Correctness | Dispatch latency |
|----|-------------|------------------|
| mm  | ✓ vs CPU | 0.37 ms cold → 0.28 ms cached (1.3×) |
| add | ✓ vs CPU | 0.21 ms cached |
| gelu | ✓ vs CPU | — |
| relu | ✓ vs CPU | — |

(Other wired ops above are registered and dispatch their shader, but don't yet have
committed correctness tests.)

## Written but not yet wired

These GLSL/SPIR-V kernels are implemented in `csrc/shaders/` but have **no host-side
dispatch yet** — they're the roadmap, not working ops: register-tiled & general
matmul, **Q5_K / Q6_K** matmul, batched quantized matmul, fused norm+matmul
(Q4_K/Q5_K/Q6_K), rmsnorm, rope, silu-gate, argmax, residual_add,
attention + KV cache, kv_store.

## Quickstart

```bash
# Requires a Vulkan 1.2+ driver (RADV recommended) and glslangValidator for shader build.

# 1. (Re)compile the GLSL shaders to SPIR-V (pre-built .spv files are committed,
#    so this is only needed if you edit a shader). Targets Vulkan 1.1 / SPIR-V 1.3
#    because the quantized kernels use subgroup ops.
bash csrc/shaders/compile.sh

# 2. Build the C++ extension.
cmake -S . -B build && cmake --build build -j

# 3. Install the Python package (build_ext also invokes CMake).
pip install -e .

python -c "import torch, torch_vulkan; print(torch_vulkan.device_name())"
pytest tests/
```

## How it's built

- **`csrc/`** — C++ extension. `TORCH_LIBRARY` registers ops on `PrivateUse1`;
  Vulkan setup (instance/device/queue, descriptor sets, push constants) is handled
  via [Kompute](https://github.com/KomputeProject/kompute), which dispatches the
  precompiled `.spv` shaders.
- **`csrc/shaders/`** — GLSL compute shaders, one per op. Quantized kernels read raw
  GGUF superblock layouts (scales + packed nibbles) and dequant on-device, so weights
  stay quantized in VRAM/GTT.
- **Algorithm cache** — pipelines and command sequences are cached and reused across
  dispatches; `torch_vulkan.cache_stats()` reports hits/misses.

## Status & limitations

- Targeted and tested on **AMD Radeon 890M (gfx1150)** via RADV. Other Vulkan GPUs
  should work but are unverified.
- Op coverage is inference-oriented; this is not a full autograd training backend.
- fp32 + GGUF-quantized weights; no fp16 storage path yet.
- **Tensor creation is partial**: use `.to("vulkan")`, `torch.empty(device="vulkan")`,
  or `torch.tensor(data, device="vulkan")`. The `torch.randn(..., device="vulkan")`
  constructor (needs `aten::_copy_from_and_resize`) and a `.vulkan()` tensor method
  are not implemented yet.

## Hardware

Developed on a GPD Pocket 4 — AMD Ryzen AI 9 HX 370, Radeon 890M, unified
VRAM+GTT memory pool. Vulkan-only by design.
