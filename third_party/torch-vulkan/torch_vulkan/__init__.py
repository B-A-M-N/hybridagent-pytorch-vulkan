"""
torch-vulkan: Vulkan compute backend for PyTorch

Usage:
    import torch
    import torch_vulkan

    # Check availability
    print(torch_vulkan.is_available())
    print(torch_vulkan.device_name())

    # Create on CPU, then move to the Vulkan device (the supported path):
    a = torch.randn(64, 64).to("vulkan")
    b = torch.randn(64, 64).to("vulkan")
    c = torch.mm(a, b)  # runs matmul_tiled.spv on GPU
    c.cpu()             # bring the result back

    # torch.empty(..., device="vulkan") and torch.tensor(data, device="vulkan")
    # also work. The torch.randn(..., device="vulkan") constructor and the
    # .vulkan() method are only partially wired -- see the README Limitations
    # section. Prefer .to("vulkan").
"""

import os

import torch

# Load the C++ extension — this registers PrivateUse1 as "vulkan"
from torch_vulkan import _C

# Add a `.vulkan()` tensor method (and module method) so x.vulkan() works,
# mirroring x.cuda(). Prefer PyTorch's official generator; fall back to a
# minimal shim if it raises (e.g. on double-registration).
try:
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=False
    )
except Exception as e:
    import warnings
    warnings.warn(
        f"generate_methods_for_privateuse1_backend failed, using .vulkan() shim: {e}"
    )
    torch.Tensor.vulkan = lambda self, *a, **k: self.to("vulkan")

# Point the shader loader at our bundled .spv files.
_shader_dir = os.path.join(os.path.dirname(__file__), "..", "csrc", "shaders")
if os.path.isdir(_shader_dir):
    _abs_shader_dir = os.path.abspath(_shader_dir)
    _C._set_shader_dir(_abs_shader_dir)
    # The raw VulkanEngine (used by the mm_raw path) is constructed lazily and
    # reads its shader directory from this env var, so it resolves to the same
    # bundled shaders instead of a hardcoded default.
    os.environ.setdefault("TORCH_VULKAN_SHADER_DIR", _abs_shader_dir)


def is_available() -> bool:
    """Check if a Vulkan-capable GPU is available."""
    return _C._is_available()


def device_name() -> str:
    """Return the name of the Vulkan device."""
    return _C._device_name()


def device(index: int = 0) -> torch.device:
    """Return a torch.device for the Vulkan backend."""
    return torch.device("vulkan", index)


def cache_stats() -> dict:
    """Return algorithm cache hit/miss statistics (Phase 4).

    Returns a dict with keys:
      algo_hits:   number of times a cached Algorithm was reused
      algo_misses: number of times a new Algorithm was created
      seq_reuses:  number of times a Sequence was reused from pool
      seq_creates: number of times a new Sequence was created
    """
    return _C._cache_stats()


def clear_algorithm_cache():
    """Clear all cached Vulkan algorithms.

    Call this if you notice memory pressure from too many cached pipelines,
    or after model weights change and old descriptors are stale.
    """
    _C._clear_algorithm_cache()


# Register as torch.vulkan so .to("vulkan") works
# PyTorch's PrivateUse1 dispatch does "import torch.<backend_name>"
import sys

sys.modules['torch.vulkan'] = sys.modules[__name__]
