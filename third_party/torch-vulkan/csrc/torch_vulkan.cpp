// torch_vulkan.cpp — Entry point for the PrivateUse1 Vulkan backend.
//
// When Python does `import torch_vulkan`, this shared library loads and:
// 1. Renames PrivateUse1 -> "vulkan" so you get tensor.vulkan(), tensor.is_vulkan
// 2. Registers the VulkanAllocator for device memory
// 3. Operator registrations happen via TORCH_LIBRARY_IMPL in ops/*.cpp
//
// Usage:
//   import torch
//   import torch_vulkan
//   a = torch.randn(64, 64).vulkan()
//   b = torch.randn(64, 64).vulkan()
//   c = torch.mm(a, b)  # dispatches to matmul.spv on GPU

#include "vulkan_allocator.h"
#include "vulkan_context.h"
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>

namespace torch_vulkan {

// DeviceGuard integration — tells PyTorch how to manage Vulkan device context
struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    // Single-device for now (device 0)
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  void setDevice(c10::Device d) const override {
    // Single device — nothing to do
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {}

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1; // Single Vulkan device for now
  }
};

// Register the guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

// Python bindings
void set_shader_dir(const std::string& path) {
  // Set the shader dir on the Kompute-based VulkanContext (used by all the
  // registered ops). The raw VulkanEngine (used only by the not-yet-wired
  // mm_raw path) is constructed lazily and reads its directory from the
  // TORCH_VULKAN_SHADER_DIR env var, which __init__.py points at this same
  // directory -- so we deliberately do NOT touch VulkanEngine::instance()
  // here, to avoid eagerly creating a second Vulkan device at import time.
  VulkanContext::instance().set_shader_dir(path);
}

std::string device_name() {
  return VulkanContext::instance().device_name();
}

bool is_available() {
  try {
    VulkanContext::instance().manager();
    return true;
  } catch (...) {
    return false;
  }
}

// Phase 4: Expose cache statistics to Python for benchmarking.
py::dict cache_stats() {
  auto stats = VulkanContext::instance().cache_stats();
  py::dict d;
  d["algo_hits"] = stats.algo_hits;
  d["algo_misses"] = stats.algo_misses;
  d["seq_reuses"] = stats.seq_reuses;
  d["seq_creates"] = stats.seq_creates;
  return d;
}

void clear_algorithm_cache() {
  VulkanContext::instance().clear_algorithm_cache();
}

PYBIND11_MODULE(_C, m) {
  m.def("_set_shader_dir", &set_shader_dir);
  m.def("_device_name", &device_name);
  m.def("_is_available", &is_available);
  m.def("_cache_stats", &cache_stats,
        "Return algorithm cache hit/miss statistics (Phase 4)");
  m.def("_clear_algorithm_cache", &clear_algorithm_cache,
        "Clear all cached Vulkan algorithms");

  // Rename PrivateUse1 -> "vulkan"
  // This gives us: tensor.vulkan(), tensor.is_vulkan, torch.device("vulkan")
  c10::register_privateuse1_backend("vulkan");

  // Register our allocator for the vulkan device
  c10::SetAllocator(
      c10::DeviceType::PrivateUse1, &VulkanAllocator::instance());
}

} // namespace torch_vulkan
