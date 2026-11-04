#pragma once

#include <cstring>
#include <c10/core/Allocator.h>
#include <kompute/Kompute.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch_vulkan {

// Phase 4: Buffer-reusing allocator.
// When a Vulkan tensor is freed, its underlying Kompute tensor (and Vulkan
// buffer) goes onto a size-bucketed free list. The next allocation of the
// same size reuses the existing buffer, which means:
//   1. The data pointer is the same as before, so
//   2. The algorithm cache in VulkanContext can match on tensor pointers and
//   3. The cached Algorithm (VkPipeline + descriptor sets) is reused.
// This eliminates the 3-5ms per-op overhead of pipeline + descriptor set
// recreation during transformer inference.
class VulkanAllocator final : public c10::Allocator {
public:
  static VulkanAllocator& instance();

  c10::DataPtr allocate(size_t nbytes) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  std::shared_ptr<kp::TensorT<float>> get_kompute_tensor(void* ptr);

private:
  VulkanAllocator() = default;

  static void deleter(void* ptr);

  std::mutex mutex_;

  // Active allocations: data_ptr -> Kompute tensor
  std::unordered_map<void*, std::shared_ptr<kp::TensorT<float>>> tensor_map_;

  // Free list: size_in_floats -> list of reusable Kompute tensors
  // When a tensor is freed, it moves here instead of being destroyed.
  std::unordered_map<size_t, std::vector<std::shared_ptr<kp::TensorT<float>>>> free_list_;
};

} // namespace torch_vulkan
