#include "vulkan_context.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace torch_vulkan {

VulkanContext& VulkanContext::instance() {
  static VulkanContext ctx;
  return ctx;
}

VulkanContext::VulkanContext() {
  // Kompute manages Vulkan instance + device selection automatically.
  // Device 0 = first Vulkan-capable GPU found.
  mgr_ = std::make_unique<kp::Manager>(0);

  // Default shader directory: alongside the shared library
  shader_dir_ = "";
}

kp::Manager& VulkanContext::manager() {
  return *mgr_;
}

void VulkanContext::set_shader_dir(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  shader_dir_ = path;
  if (!shader_dir_.empty() && shader_dir_.back() != '/') {
    shader_dir_ += '/';
  }
}

std::vector<uint32_t> VulkanContext::load_shader(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = shader_cache_.find(name);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  std::string path = shader_dir_ + name + ".spv";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to load shader: " + path);
  }

  size_t size = file.tellg();
  file.seekg(0);

  std::vector<uint32_t> spirv(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(spirv.data()), size);

  shader_cache_[name] = spirv;
  return spirv;
}

std::string VulkanContext::device_name() const {
  // Kompute exposes device properties through the manager
  return "Vulkan Device 0";
}

// ---- Algorithm Cache (Phase 4) ----

std::string VulkanContext::make_cache_key(
    const std::string& shader_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
  // Key format: "shader_name:N:ptr0:ptr1:..."
  // Using raw data pointers ensures we match only when the exact same
  // Vulkan buffers are bound. This is safe because allocator-backed tensors
  // have stable pointers for their entire lifetime.
  std::ostringstream oss;
  oss << shader_name << ':' << tensors.size();
  for (const auto& t : tensors) {
    oss << ':' << reinterpret_cast<uintptr_t>(t->rawData());
  }
  return oss.str();
}

std::shared_ptr<kp::Algorithm> VulkanContext::get_or_create_algorithm(
    const std::string& shader_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    const kp::Workgroup& workgroup,
    const std::vector<uint32_t>& push_constants) {

  std::string key = make_cache_key(shader_name, tensors);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = algo_cache_.find(key);
  if (it != algo_cache_.end()) {
    // Cache hit: reuse existing algorithm.
    // Update mutable state: push constants and workgroup.
    auto& algo = it->second;
    if (!push_constants.empty()) {
      algo->setPushConstants(
          const_cast<void*>(static_cast<const void*>(push_constants.data())),
          static_cast<uint32_t>(push_constants.size()),
          static_cast<uint32_t>(sizeof(uint32_t)));
    }
    algo->setWorkgroup(workgroup, tensors.size() ? tensors[0]->size() : 1);
    stats_.algo_hits++;
    return algo;
  }

  // Cache miss: create new algorithm and cache it.
  // Check shader cache inline (load_shader would deadlock since we hold mutex_).
  auto spirv = shader_cache_.count(shader_name)
      ? shader_cache_[shader_name]
      : std::vector<uint32_t>{};

  if (spirv.empty()) {
    std::string path = shader_dir_ + shader_name + ".spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to load shader: " + path);
    }
    size_t size = file.tellg();
    file.seekg(0);
    spirv.resize(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    shader_cache_[shader_name] = spirv;
  }

  auto algo = mgr_->algorithm<float, uint32_t>(
      tensors, spirv, workgroup,
      std::vector<float>{}, push_constants);

  algo_cache_[key] = algo;
  stats_.algo_misses++;
  return algo;
}

std::shared_ptr<kp::Sequence> VulkanContext::acquire_sequence() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!seq_pool_.empty()) {
    auto seq = std::move(seq_pool_.back());
    seq_pool_.pop_back();
    stats_.seq_reuses++;
    return seq;
  }

  stats_.seq_creates++;
  return mgr_->sequence();
}

void VulkanContext::release_sequence(std::shared_ptr<kp::Sequence> seq) {
  // Sequences in Kompute create their command pool without the RESET bit,
  // so the command buffer cannot be reused after recording+eval.
  // Just let the sequence drop. The main perf win is from Algorithm caching.
  // (Intentionally left as no-op.)
}

VulkanContext::CacheStats VulkanContext::cache_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void VulkanContext::clear_algorithm_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  algo_cache_.clear();
}

} // namespace torch_vulkan
