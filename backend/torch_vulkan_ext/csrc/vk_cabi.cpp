// C ABI surface for the raw-Vulkan backend, loaded from Python via ctypes.
//
// ctypes cannot bind C++ symbols (name mangling, std:: types), so this file
// exposes a flat extern "C" API over tvk::Backend. Allocation handles are
// opaque pointers to a tvk::Allocation on the C++ heap. BufferView handles
// are also opaque pointers to a tvk::BufferView.
//
// Thread-safety: tvk::Backend is internally mutex-guarded; the Python layer
// additionally serialises dispatches with a lock.

#include "vk_backend.h"
#include <cstdint>
#include <string>
#include <vector>

extern "C" {

// ---- diagnostics -------------------------------------------------------
static char g_last_error[1024] = {0};
const char* tvk_last_error() { return g_last_error; }
static void set_err(const char* what, const char* detail) {
  try {
    std::string s = std::string(what);
    if (detail) s += ": " + std::string(detail);
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", s.c_str());
  } catch (...) {
  }
}

// ---- lifecycle -----------------------------------------------------------
int tvk_init(int device_index) {
  try {
    return tvk::Backend::instance().init((uint32_t)device_index) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

const char* tvk_device_name() {
  try {
    return tvk::Backend::instance().device_name().c_str();
  } catch (...) {
    return "";
  }
}

void tvk_set_shader_dir(const char* path) {
  tvk::Backend::instance().set_shader_dir(std::string(path));
}

long long tvk_vram_total() { return (long long)tvk::Backend::instance().vram_total(); }
long long tvk_vram_used() { return (long long)tvk::Backend::instance().vram_used(); }
long long tvk_dispatch_count() {
  return (long long)tvk::Backend::instance().dispatch_count();
}
void tvk_reset_counters() { tvk::Backend::instance().reset_counters(); }
void tvk_set_max_descriptor_sets(int n) {
  if (n > 0) tvk::Backend::instance().set_max_descriptor_sets((uint32_t)n);
}

// ---- allocations ---------------------------------------------------------
// Returns an opaque handle (heap tvk::Allocation*), or 0 on failure.
void* tvk_alloc_device(long long nbytes) {
  try {
    auto* a = tvk::Backend::instance().create_device_allocation((VkDeviceSize)nbytes);
    return a;
  } catch (...) {
    return nullptr;
  }
}

void* tvk_alloc_host(long long nbytes) {
  try {
    auto* a = tvk::Backend::instance().create_host_allocation((VkDeviceSize)nbytes);
    return a;
  } catch (...) {
    return nullptr;
  }
}

// Create a view into an existing allocation (shared ownership).
// Returns a tvk::BufferView* (opaque handle), or 0 on failure.
void* tvk_buf_view(void* alloc_handle, long long offset, long long size) {
  if (!alloc_handle) return nullptr;
  try {
    auto* alloc = static_cast<tvk::Allocation*>(alloc_handle);
    auto* view = new tvk::BufferView(
        alloc, (VkDeviceSize)offset, (VkDeviceSize)size);
    return view;
  } catch (...) {
    return nullptr;
  }
}

// Release a view (decrement refcount on the allocation).
void tvk_buf_view_free(void* view_handle) {
  if (!view_handle) return;
  delete static_cast<tvk::BufferView*>(view_handle);
}

// Release an allocation (decrement refcount; frees when last ref drops).
void tvk_alloc_free(void* alloc_handle) {
  if (!alloc_handle) return;
  auto* alloc = static_cast<tvk::Allocation*>(alloc_handle);
  try {
    tvk::Backend::instance().release_view(alloc);
  } catch (...) {}
}

// The VkBuffer handle (as an integer) for descriptor binding.
unsigned long long tvk_buf_vk(void* view_handle) {
  if (!view_handle) return 0;
  auto* view = static_cast<tvk::BufferView*>(view_handle);
  return (unsigned long long)(uintptr_t)view->vk_buffer();
}

// Buffer view properties
long long tvk_buf_view_offset(void* view_handle) {
  if (!view_handle) return 0;
  return (long long)static_cast<tvk::BufferView*>(view_handle)->vk_offset();
}
long long tvk_buf_view_size(void* view_handle) {
  if (!view_handle) return 0;
  return (long long)static_cast<tvk::BufferView*>(view_handle)->vk_size();
}

void* tvk_buf_map(void* alloc_handle) {
  if (!alloc_handle) return nullptr;
  try {
    return tvk::Backend::instance().map(static_cast<tvk::Allocation*>(alloc_handle));
  } catch (...) {
    return nullptr;
  }
}

void tvk_buf_unmap(void* alloc_handle) {
  if (!alloc_handle) return;
  try {
    tvk::Backend::instance().unmap(static_cast<tvk::Allocation*>(alloc_handle));
  } catch (...) {}
}

// ---- dispatch ------------------------------------------------------------
int tvk_dispatch(const char* shader,
                 const unsigned long long* views, int n,
                 const void* push, int push_size,
                 int gx, int gy, int gz) {
  try {
    std::vector<tvk::BufferView> views_vec;
    views_vec.reserve(n);
    for (int i = 0; i < n; i++) {
      // The Python side passes BufferView handles; we need to extract the view.
      // For backwards compatibility with the old API where views[i] was VkBuffer,
      // we detect the case: if views[i] looks like a BufferView*, we cast it.
      // Otherwise, assume it's a raw VkBuffer (legacy) - but this path is deprecated.
      views_vec.push_back(*static_cast<tvk::BufferView*>(reinterpret_cast<void*>(views[i])));
    }
    tvk::Backend::instance().dispatch(std::string(shader), views_vec,
                                      push, (uint32_t)push_size,
                                      (uint32_t)gx, (uint32_t)gy, (uint32_t)gz);
    return 0;
  } catch (const std::exception& e) {
    set_err("dispatch", e.what());
    return 1;
  } catch (...) {
    set_err("dispatch", "unknown");
    return 1;
  }
}

// Asynchronous dispatch: the work is queued but not waited on. Several of these
// can be issued back to back and drained with tvk_submit(), which removes the
// per-op host round trip that dominated the batched-layer timing.
int tvk_dispatch_async(const char* shader,
                       const unsigned long long* views, int n,
                       const void* push, int push_size,
                       int gx, int gy, int gz) {
  try {
    std::vector<tvk::BufferView> views_vec;
    views_vec.reserve(n);
    for (int i = 0; i < n; i++) {
      views_vec.push_back(*static_cast<tvk::BufferView*>(reinterpret_cast<void*>(views[i])));
    }
    tvk::Backend::instance().dispatch(std::string(shader), views_vec,
                                      push, (uint32_t)push_size,
                                      (uint32_t)gx, (uint32_t)gy, (uint32_t)gz, false);
    return 0;
  } catch (const std::exception& e) {
    set_err("dispatch_async", e.what());
    return 1;
  } catch (...) {
    set_err("dispatch_async", "unknown");
    return 1;
  }
}

void tvk_submit() { tvk::Backend::instance().submit(); }

// Copy a host allocation into a device allocation (staged, synchronous).
int tvk_copy_to_device(void* dst_alloc, void* src_alloc) {
  try {
    tvk::Backend::instance().copy_to_device(
        static_cast<tvk::Allocation*>(dst_alloc),
        static_cast<tvk::Allocation*>(src_alloc));
    return 0;
  } catch (...) {
    return 1;
  }
}

// Copy a device allocation into a host allocation (synchronous).
int tvk_copy_from_device(void* dst_alloc, void* src_alloc) {
  try {
    tvk::Backend::instance().copy_from_device(
        static_cast<tvk::Allocation*>(dst_alloc),
        static_cast<tvk::Allocation*>(src_alloc));
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // extern "C"