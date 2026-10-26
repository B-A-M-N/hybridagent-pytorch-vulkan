#pragma once
// Shared registry mapping a torch PrivateUse1 tensor's data_ptr() to the
// VkBuffer that holds its data.
//
// Why this exists: torch-vulkan's VulkanAllocator hands out Kompute tensors
// whose VkBuffer is private (Kompute::Tensor::mPrimaryBuffer). To run our own
// raw-Vulkan dispatches against those tensors -- and to keep them resident in
// VRAM without a host mirror -- we need to recover that VkBuffer. The
// registry lives in this shared header so both the extension library and the
// torch-vulkan build (patched) can see it; a single definition is emitted by
// tvk_registry.cpp.

#include <vulkan/vulkan.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tvk {

class Registry {
public:
  static Registry& instance() {
    static Registry r;
    return r;
  }

  // Record that `key` (a torch data_ptr) is backed by `buf` of `bytes`.
  void put(void* key, VkBuffer buf, VkDeviceSize bytes) {
    std::lock_guard<std::mutex> lk(m_);
    map_[key] = Entry{buf, bytes};
  }

  VkBuffer buffer(void* key) const {
    std::lock_guard<std::mutex> lk(m_);
    auto it = map_.find(key);
    return it == map_.end() ? VK_NULL_HANDLE : it->second.buf;
  }

  VkDeviceSize size(void* key) const {
    std::lock_guard<std::mutex> lk(m_);
    auto it = map_.find(key);
    return it == map_.end() ? 0 : it->second.bytes;
  }

  void erase(void* key) {
    std::lock_guard<std::mutex> lk(m_);
    map_.erase(key);
  }

  uint64_t total_bytes() const {
    std::lock_guard<std::mutex> lk(m_);
    uint64_t t = 0;
    for (auto& kv : map_) t += kv.second.bytes;
    return t;
  }

  size_t count() const {
    std::lock_guard<std::mutex> lk(m_);
    return map_.size();
  }

private:
  Registry() = default;
  struct Entry {
    VkBuffer buf;
    VkDeviceSize bytes;
  };
  std::unordered_map<void*, Entry> map_;
  mutable std::mutex m_;
};

}  // namespace tvk
