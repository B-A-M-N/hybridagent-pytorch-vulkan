// Test command buffer lifecycle with multiple async dispatches without submit
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <dlfcn.h>

typedef int (*tvk_init_t)(int);
typedef const char* (*tvk_device_name_t)();
typedef void (*tvk_set_shader_dir_t)(const char*);
typedef long long (*tvk_vram_total_t)();
typedef int (*tvk_dispatch_t)(const char*, const unsigned long long*, int, const void*, int, int, int, int);
typedef int (*tvk_dispatch_async_t)(const char*, const unsigned long long*, int, const void*, int, int, int, int);
typedef void (*tvk_submit_t)();
typedef const char* (*tvk_last_error_t)();
typedef void* (*tvk_alloc_device_t)(long long);
typedef void* (*tvk_alloc_host_t)(long long);
typedef void* (*tvk_buf_view_t)(void*, long long, long long);
typedef unsigned long long (*tvk_buf_vk_t)(void*);

int main() {
    std::cout << "=== Command Buffer Lifecycle Test (Multiple Async) ===" << std::endl;

    void* handle = dlopen("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext/lib/tvkvk.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "FAIL: dlopen: " << dlerror() << std::endl;
        return 1;
    }

    tvk_init_t tvk_init = (tvk_init_t)dlsym(handle, "tvk_init");
    tvk_device_name_t tvk_device_name = (tvk_device_name_t)dlsym(handle, "tvk_device_name");
    tvk_set_shader_dir_t tvk_set_shader_dir = (tvk_set_shader_dir_t)dlsym(handle, "tvk_set_shader_dir");
    tvk_vram_total_t tvk_vram_total = (tvk_vram_total_t)dlsym(handle, "tvk_vram_total");
    tvk_dispatch_t tvk_dispatch = (tvk_dispatch_t)dlsym(handle, "tvk_dispatch");
    tvk_dispatch_async_t tvk_dispatch_async = (tvk_dispatch_async_t)dlsym(handle, "tvk_dispatch_async");
    tvk_submit_t tvk_submit = (tvk_submit_t)dlsym(handle, "tvk_submit");
    tvk_last_error_t tvk_last_error = (tvk_last_error_t)dlsym(handle, "tvk_last_error");
    tvk_alloc_device_t tvk_alloc_device = (tvk_alloc_device_t)dlsym(handle, "tvk_alloc_device");
    tvk_alloc_host_t tvk_alloc_host = (tvk_alloc_host_t)dlsym(handle, "tvk_alloc_host");
    tvk_buf_view_t tvk_buf_view = (tvk_buf_view_t)dlsym(handle, "tvk_buf_view");
    tvk_buf_vk_t tvk_buf_vk = (tvk_buf_vk_t)dlsym(handle, "tvk_buf_vk");

    if (!tvk_init || !tvk_device_name || !tvk_set_shader_dir || !tvk_vram_total || !tvk_dispatch) {
        std::cerr << "FAIL: Missing symbols" << std::endl;
        return 1;
    }

    if (!tvk_init(0)) {
        std::cerr << "FAIL: tvk_init: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: Backend initialized" << std::endl;

    tvk_set_shader_dir("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext/csrc/shaders/");

    void* host_alloc = tvk_alloc_host(64);
    void* host_view = tvk_buf_view(host_alloc, 0, 64);

    void* dev_alloc1 = tvk_alloc_device(1024);
    void* dev_alloc2 = tvk_alloc_device(1024);
    void* dev_alloc3 = tvk_alloc_device(1024);
    void* view1 = tvk_buf_view(dev_alloc1, 0, 1024);
    void* view2 = tvk_buf_view(dev_alloc2, 0, 1024);
    void* view3 = tvk_buf_view(dev_alloc3, 0, 1024);

    unsigned long long handles[] = { (unsigned long long)view1, (unsigned long long)view2, (unsigned long long)view3 };

    // Test: Multiple async dispatches BEFORE submit (this tests the command buffer allocation logic)
    std::cout << "\n--- Test: 5 async dispatches without submit ---" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Dispatch " << (i+1) << "..." << std::endl;
        int rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
        if (rc != 0) {
            std::cerr << "FAIL: tvk_dispatch_async " << (i+1) << ": " << tvk_last_error() << std::endl;
            return 1;
        }
        std::cout << "  OK" << std::endl;
    }

    std::cout << "\n--- Submitting all ---" << std::endl;
    tvk_submit();
    std::cout << "OK: tvk_submit succeeded" << std::endl;

    // Test again: More async dispatches
    std::cout << "\n--- Test: Another 5 async dispatches ---" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Dispatch " << (i+1) << "..." << std::endl;
        int rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
        if (rc != 0) {
            std::cerr << "FAIL: tvk_dispatch_async " << (i+1) << ": " << tvk_last_error() << std::endl;
            return 1;
        }
        std::cout << "  OK" << std::endl;
    }

    tvk_submit();
    std::cout << "OK: tvk_submit succeeded" << std::endl;

    // Test: Mix sync and async
    std::cout << "\n--- Test: Sync dispatch after async ---" << std::endl;
    int rc = tvk_dispatch("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch sync: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch sync succeeded" << std::endl;

    std::cout << "\n--- Test: Async after sync ---" << std::endl;
    rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch_async: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch_async succeeded" << std::endl;

    tvk_submit();
    std::cout << "OK: tvk_submit succeeded" << std::endl;

    dlclose(handle);
    std::cout << "\n=== ALL COMMAND BUFFER LIFECYCLE TESTS PASSED ===" << std::endl;
    return 0;
}