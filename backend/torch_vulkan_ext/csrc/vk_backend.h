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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace tvk {

// Forward declaration
class Backend;

// Shared allocation owner with reference counting.
// A BufferView is a view into an Allocation (offset + size).
// Only the Allocation owns VkBuffer/VkDeviceMemory and frees them when
// the last reference (from views or pending submissions) drops.
class Allocation {
public:
  Allocation() = default;
  Allocation(VkBuffer buf, VkDeviceMemory mem, VkDeviceSize sz, Backend* be)
    : buffer(buf), memory(mem), size(sz), backend(be) {}

  ~Allocation(); // defined in .cpp after Backend definition

  void add_ref() { refs_.fetch_add(1, std::memory_order_relaxed); }
  void release() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  Backend* backend = nullptr;
  std::atomic<uint32_t> refs_{1};
};

struct BufferView {
  Allocation* alloc = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;

  BufferView() = default;
  BufferView(Allocation* a, VkDeviceSize off, VkDeviceSize sz)
    : alloc(a), offset(off), size(sz) {
    if (alloc) alloc->add_ref();
  }
  BufferView(const BufferView& other) : alloc(other.alloc), offset(other.offset), size(other.size) {
    if (alloc) alloc->add_ref();
  }
  BufferView& operator=(const BufferView& other) {
    if (this != &other) {
      if (alloc) alloc->release();
      alloc = other.alloc;
      offset = other.offset;
      size = other.size;
      if (alloc) alloc->add_ref();
    }
    return *this;
  }
  ~BufferView() { if (alloc) alloc->release(); }

  VkBuffer vk_buffer() const { return alloc ? alloc->buffer : VK_NULL_HANDLE; }
  VkDeviceSize vk_offset() const { return offset; }
  VkDeviceSize vk_size() const { return size; }

  bool operator==(const BufferView& other) const {
    return alloc == other.alloc && offset == other.offset && size == other.size;
  }
  bool operator!=(const BufferView& other) const { return !(*this == other); }
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
  // Allocate a new Allocation (owns VkBuffer + VkDeviceMemory).
  class Allocation* create_device_allocation(VkDeviceSize size);
  class Allocation* create_host_allocation(VkDeviceSize size);
  // Destroy an allocation (called by Allocation dtor, not directly).
  void destroy_allocation(Allocation* a);

  // Create a view into an existing allocation (shared ownership).
  // Returns a new BufferView with shared ownership of the allocation.
  BufferView make_view(Allocation* alloc, VkDeviceSize offset, VkDeviceSize size);

  // Release a view (decrement refcount). Called by BufferView dtor.
  void release_view(Allocation* alloc);

  void* map(Allocation* alloc);
  void unmap(Allocation* alloc);

  // ---- dispatch --------------------------------------------------------
  // `sync == false` leaves the work queued so several dispatches can be
  // submitted together; call `submit()` (or any syncing dispatch) to drain.
  void dispatch(const std::string& shader,
                const std::vector<BufferView>& views,
                const void* push, uint32_t push_size,
                uint32_t gx, uint32_t gy, uint32_t gz, bool sync = true);

  // Force the queued work to complete. Called automatically by any dispatch
  // with sync == true.
  void submit();

  // ---- transfer ---------------------------------------------------------
  // Copy `src` (host buffer) -> `dst` (device buffer), synchronously.
  void copy_to_device(Allocation* dst, Allocation* src);
  // Copy `src` (device buffer) -> `dst` (host buffer), synchronously.
  void copy_from_device(Allocation* dst, Allocation* src);

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
    std::vector<BufferView> views;
    uint64_t gen = 0;   // allocator generation at bind time
    bool live = false;  // a descriptor set is currently allocated
  };
  std::unordered_map<std::string, Bound> bound_;
  uint64_t dispatches_ = 0;
  uint64_t bytes_ = 0;
  bool cmd_inflight_ = false;
  uint64_t alloc_gen_ = 0;
  uint32_t max_desc_sets_ = 4096;
  // RECURSIVE mutex.
  //
  // destroy_allocation() takes mutex_ because it mutates alloc_gen_ and
  // bytes_. But ~Allocation -> destroy_allocation() is reachable from
  // INSIDE a region that already holds mutex_:
  //
  //   dispatch() [holds mutex_]
  //     -> bnd.views = views;        (replaces the cached BufferViews)
  //        -> old ~BufferView -> alloc->release() -> last ref
  //           -> delete Allocation -> ~Allocation
  //              -> destroy_allocation() -> lock_guard(mutex_)  <-- SELF-DEADLOCK
  //
  // The same path exists in submit(), which releases the retained Allocations
  // of completed submissions while holding mutex_.
  //
  // A plain std::mutex deadlocks there. std::recursive_mutex makes the
  // re-entrant acquisition legal. This was a latent hang, not a slow path:
  // it presented as "Vulkan is 4000x slower than CPU" because the call
  // blocked forever instead of returning.
  mutable std::recursive_mutex mutex_;

  // Track allocations retained by in-flight submissions.
  struct Submission {
    std::vector<Allocation*> retained_allocs;
    std::vector<BufferView> views;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool new_cmd = false;
  };
  std::vector<Submission> pending_submissions_;
  // Every live Allocation, so the destructor can free them before the device
  // goes away. Guarded by mutex_.
  std::vector<Allocation*> live_allocs_;
};

}  // namespace tvk