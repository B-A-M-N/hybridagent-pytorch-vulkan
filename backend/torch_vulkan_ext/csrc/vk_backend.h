#pragma once
// Vulkan compute backend for the HybridAgent PyTorch extension.
//
// Self-contained raw-Vulkan implementation: it does NOT use Kompute.
// Kompute was evaluated and rejected for the extension because its
// `TensorT<T>` with TensorTypes::eDevice allocates a device-local primary
// buffer *plus a host-visible staging buffer of the same size*
// (kompute/src/Tensor.cpp, allocateMemoryCreateGPUResources). For resident
// transformer weights that doubles VRAM use for no benefit, since weights never
// need a host mirror after upload.
//
// torch-vulkan is still used for what it does well: PrivateUse1 registration,
// the Vulkan allocator, and the fp32 mm/add/mul/softmax/embedding/layernorm
// kernels. This backend adds the transformer kernels torch-vulkan leaves
// unwired (RoPE, RMSNorm, SiLU gate, KV cache, single-token attention, argmax)
// and manages device memory itself.
//
// No Python.h dependency: the Python layer drives this through ctypes, so it
// links against libvulkan only and needs no Python development headers.

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace tvk {

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

struct Pipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  uint32_t n_bindings = 0;
  uint32_t push_size = 0;
};

class Backend {
public:
  static Backend& instance();

  bool init(uint32_t device_index);
  bool ready() const { return ready_; }
  const std::string& device_name() const { return device_name_; }
  uint64_t vram_total() const;
  uint64_t vram_used() const;
  uint64_t dispatch_count() const { return dispatches_; }
  void reset_counters() { dispatches_ = 0; }

  // Maximum descriptor sets per pipeline pool. The pool is created lazily on
  // first use of a shader, so this must be set before any dispatch.
  void set_max_descriptor_sets(uint32_t n) { max_desc_sets_ = n; }

  // ---- memory ----------------------------------------------------------
  Buffer create_device_buffer(VkDeviceSize size);
  Buffer create_host_buffer(VkDeviceSize size);
  void destroy_buffer(Buffer& b);
  void* map(Buffer& b);
  void unmap(Buffer& b);

  // ---- dispatch --------------------------------------------------------
  // `sync == false` leaves the work queued so several dispatches can be
  // submitted together; call `submit()` (or any syncing dispatch) to drain.
  void dispatch(const std::string& shader,
                const std::vector<VkBuffer>& buffers,
                const void* push, uint32_t push_size,
                uint32_t gx, uint32_t gy, uint32_t gz, bool sync = true);

  // Force the queued work to complete. Called automatically by any dispatch
  // with sync == true.
  void submit();

  // ---- transfer ---------------------------------------------------------
  // Copy `src` (host buffer) -> `dst` (device buffer), synchronously.
  void copy_to_device(Buffer& dst, Buffer& src);
  // Copy `src` (device buffer) -> `dst` (host buffer), synchronously.
  void copy_from_device(Buffer& dst, Buffer& src);

  // ---- shaders ---------------------------------------------------------
  std::vector<uint32_t> load_shader(const std::string& name);
  // Must be called with mutex_ already held.
  std::vector<uint32_t> load_shader_unlocked(const std::string& name);
  void set_shader_dir(const std::string& path) { shader_dir_ = path; }
  const std::string& shader_dir() const { return shader_dir_; }

  // ---- helpers ---------------------------------------------------------
  uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags props);
  void ensure_cmd();

  ~Backend();

private:
  Backend() = default;
  bool ready_ = false;
  std::string device_name_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t qfam_ = 0;
  VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  std::string shader_dir_;
  std::unordered_map<std::string, std::vector<uint32_t>> shader_cache_;
  std::unordered_map<std::string, Pipeline> pipelines_;
  struct Bound {
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::vector<VkBuffer> buffers;
    uint32_t n = 0;
    uint64_t gen = 0;   // allocator generation at bind time
    bool live = false;  // a descriptor set is currently allocated
  };
  std::unordered_map<std::string, Bound> bound_;
  uint64_t dispatches_ = 0;
  uint64_t bytes_ = 0;
  // True when an async dispatch has been submitted but the command buffer has
  // not yet been reset -- it must not be reused until the GPU is done.
  bool cmd_inflight_ = false;
  // Bumped on every buffer free, so a recycled VkBuffer address is never
  // mistaken for the same live tensor when deciding to reuse a descriptor set.
  uint64_t alloc_gen_ = 0;
  // Pool sizing; see set_max_descriptor_sets.
  uint32_t max_desc_sets_ = 4096;
  mutable std::mutex mutex_;
};

}  // namespace tvk
