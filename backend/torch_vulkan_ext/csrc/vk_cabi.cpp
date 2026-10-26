// C ABI surface for the raw-Vulkan backend, loaded from Python via ctypes.
//
// ctypes cannot bind C++ symbols (name mangling, std:: types), so this file
// exposes a flat extern "C" API over tvk::Backend. Buffer handles are passed
// as opaque pointers to a tvk::Buffer allocated on the C++ heap.
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

// ---- buffers -------------------------------------------------------------
// Returns an opaque handle (heap tvk::Buffer*), or 0 on failure.
void* tvk_buf_device(long long nbytes) {
  try {
    auto* b = new tvk::Buffer(tvk::Backend::instance().create_device_buffer(
        (VkDeviceSize)nbytes));
    return b;
  } catch (...) {
    return nullptr;
  }
}

void* tvk_buf_host(long long nbytes) {
  try {
    auto* b = new tvk::Buffer(
        tvk::Backend::instance().create_host_buffer((VkDeviceSize)nbytes));
    return b;
  } catch (...) {
    return nullptr;
  }
}

void tvk_buf_free(void* h) {
  if (!h) return;
  auto* b = static_cast<tvk::Buffer*>(h);
  try {
    tvk::Backend::instance().destroy_buffer(*b);
  } catch (...) {
  }
  delete b;
}

long long tvk_buf_size(void* h) {
  if (!h) return 0;
  return (long long)static_cast<tvk::Buffer*>(h)->size;
}

// The VkBuffer handle (as an integer) for descriptor binding.
unsigned long long tvk_buf_vk(void* h) {
  if (!h) return 0;
  return (unsigned long long)(uintptr_t)static_cast<tvk::Buffer*>(h)->buffer;
}

void* tvk_buf_map(void* h) {
  if (!h) return nullptr;
  try {
    return tvk::Backend::instance().map(*static_cast<tvk::Buffer*>(h));
  } catch (...) {
    return nullptr;
  }
}

void tvk_buf_unmap(void* h) {
  if (!h) return;
  try {
    tvk::Backend::instance().unmap(*static_cast<tvk::Buffer*>(h));
  } catch (...) {
  }
}

// ---- dispatch ------------------------------------------------------------
// `buffers` is an array of tvk_buf_vk() values; `n` is its length.
// `push` is a pointer to the push-constant block, `push_size` its byte length.
// Returns 0 on success, non-zero on failure.
int tvk_dispatch(const char* shader,
                 const unsigned long long* buffers, int n,
                 const void* push, int push_size,
                 int gx, int gy, int gz) {
  try {
    std::vector<VkBuffer> bufs(n);
    for (int i = 0; i < n; i++) {
      bufs[i] = (VkBuffer)(uintptr_t)buffers[i];
    }
    tvk::Backend::instance().dispatch(std::string(shader), bufs,
                                      push, (uint32_t)push_size,
                                      (uint32_t)gx, (uint32_t)gy,
                                      (uint32_t)gz);
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
                        const unsigned long long* buffers, int n,
                        const void* push, int push_size,
                        int gx, int gy, int gz) {
  try {
    std::vector<VkBuffer> bufs(n);
    for (int i = 0; i < n; i++) {
      bufs[i] = (VkBuffer)(uintptr_t)buffers[i];
    }
    tvk::Backend::instance().dispatch(std::string(shader), bufs,
                                      push, (uint32_t)push_size,
                                      (uint32_t)gx, (uint32_t)gy,
                                      (uint32_t)gz, false);
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

// Copy a host buffer into a device buffer (staged, not a raw memmove).
int tvk_copy_to_device(void* dst, void* src) {
  try {
    tvk::Backend::instance().copy_to_device(*static_cast<tvk::Buffer*>(dst),
                                            *static_cast<tvk::Buffer*>(src));
    return 0;
  } catch (...) {
    return 1;
  }
}

// Copy a device buffer into a host buffer (mapped result).
int tvk_copy_from_device(void* dst, void* src) {
  try {
    tvk::Backend::instance().copy_from_device(*static_cast<tvk::Buffer*>(dst),
                                              *static_cast<tvk::Buffer*>(src));
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // extern "C"
