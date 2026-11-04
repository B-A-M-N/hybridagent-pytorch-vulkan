#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>

namespace torch_vulkan {

// Singleton managing the Vulkan instance, device, and shader cache.
// Wraps Kompute to hide Vulkan boilerplate.
class VulkanContext {
public:
  static VulkanContext& instance();

  kp::Manager& manager();

  // Load a pre-compiled SPIR-V shader from the shaders/ directory.
  // Returns the raw SPIR-V bytecode. Caches after first load.
  std::vector<uint32_t> load_shader(const std::string& name);

  // Set the directory where .spv files live.
  void set_shader_dir(const std::string& path);

  std::string device_name() const;

  // ---- Algorithm Cache (Phase 4) ----
  // Get a cached algorithm or create a new one. The cache key is built from
  // the shader name plus the raw data pointers of each tensor -- so the same
  // algorithm object is returned when the exact same (shader, buffer set) is
  // dispatched again. Push constants and workgroup are updated in-place on
  // cache hits, avoiding the expensive VkPipeline + descriptor-set rebuild.
  //
  // IMPORTANT: the caller must NOT destroy the returned algorithm.
  // The cache owns it.
  std::shared_ptr<kp::Algorithm> get_or_create_algorithm(
      const std::string& shader_name,
      const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
      const kp::Workgroup& workgroup,
      const std::vector<uint32_t>& push_constants);

  // Get a reusable Sequence from a small pool instead of creating a new one
  // each call. The sequence is cleared before return.
  std::shared_ptr<kp::Sequence> acquire_sequence();

  // Return a sequence to the pool after eval completes.
  void release_sequence(std::shared_ptr<kp::Sequence> seq);

  // Report cache statistics (for benchmarking).
  struct CacheStats {
    uint64_t algo_hits = 0;
    uint64_t algo_misses = 0;
    uint64_t seq_reuses = 0;
    uint64_t seq_creates = 0;
  };
  CacheStats cache_stats() const;

  // Clear all cached algorithms (e.g. when tensors are freed).
  void clear_algorithm_cache();

private:
  VulkanContext();
  ~VulkanContext() = default;
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  // Build the cache key string from shader name + tensor raw data pointers.
  static std::string make_cache_key(
      const std::string& shader_name,
      const std::vector<std::shared_ptr<kp::Tensor>>& tensors);

  std::unique_ptr<kp::Manager> mgr_;
  std::string shader_dir_;
  std::unordered_map<std::string, std::vector<uint32_t>> shader_cache_;

  // Algorithm cache: key -> cached algorithm
  std::unordered_map<std::string, std::shared_ptr<kp::Algorithm>> algo_cache_;

  // Sequence pool: reusable sequences
  std::vector<std::shared_ptr<kp::Sequence>> seq_pool_;

  mutable CacheStats stats_;
  mutable std::mutex mutex_;
};

} // namespace torch_vulkan
