// Test HybridAgent extension command buffer lifecycle
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

int main() {
    std::cout << "=== HybridAgent Extension Test ===" << std::endl;

    // Load the library
    void* handle = dlopen("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext/lib/tvkvk.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "FAIL: dlopen: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "OK: Library loaded" << std::endl;

    tvk_init_t tvk_init = (tvk_init_t)dlsym(handle, "tvk_init");
    tvk_device_name_t tvk_device_name = (tvk_device_name_t)dlsym(handle, "tvk_device_name");
    tvk_set_shader_dir_t tvk_set_shader_dir = (tvk_set_shader_dir_t)dlsym(handle, "tvk_set_shader_dir");
    tvk_vram_total_t tvk_vram_total = (tvk_vram_total_t)dlsym(handle, "tvk_vram_total");
    tvk_dispatch_t tvk_dispatch = (tvk_dispatch_t)dlsym(handle, "tvk_dispatch");
    tvk_dispatch_async_t tvk_dispatch_async = (tvk_dispatch_async_t)dlsym(handle, "tvk_dispatch_async");
    tvk_submit_t tvk_submit = (tvk_submit_t)dlsym(handle, "tvk_submit");
    tvk_last_error_t tvk_last_error = (tvk_last_error_t)dlsym(handle, "tvk_last_error");

    if (!tvk_init || !tvk_device_name || !tvk_set_shader_dir || !tvk_vram_total || !tvk_dispatch) {
        std::cerr << "FAIL: Missing symbols" << std::endl;
        return 1;
    }
    std::cout << "OK: Symbols resolved" << std::endl;

    // Initialize
    if (!tvk_init(0)) {
        std::cerr << "FAIL: tvk_init: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: Backend initialized" << std::endl;
    std::cout << "Device: " << tvk_device_name() << std::endl;
    std::cout << "VRAM: " << tvk_vram_total() << std::endl;

    tvk_set_shader_dir("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext/csrc/shaders/");
    std::cout << "OK: Shader dir set" << std::endl;

    // Allocate some test buffers via C ABI
    // We need to use the C ABI functions for allocation
    typedef void* (*tvk_alloc_device_t)(long long);
    typedef void* (*tvk_alloc_host_t)(long long);
    typedef void* (*tvk_buf_view_t)(void*, long long, long long);
    typedef unsigned long long (*tvk_buf_vk_t)(void*);

    tvk_alloc_device_t tvk_alloc_device = (tvk_alloc_device_t)dlsym(handle, "tvk_alloc_device");
    tvk_alloc_host_t tvk_alloc_host = (tvk_alloc_host_t)dlsym(handle, "tvk_alloc_host");
    tvk_buf_view_t tvk_buf_view = (tvk_buf_view_t)dlsym(handle, "tvk_buf_view");
    tvk_buf_vk_t tvk_buf_vk = (tvk_buf_vk_t)dlsym(handle, "tvk_buf_vk");

    if (!tvk_alloc_device || !tvk_alloc_host || !tvk_buf_view || !tvk_buf_vk) {
        std::cerr << "FAIL: Missing allocation symbols" << std::endl;
        return 1;
    }

    // Allocate host buffer for push constants
    void* host_alloc = tvk_alloc_host(64);
    if (!host_alloc) {
        std::cerr << "FAIL: host allocation" << std::endl;
        return 1;
    }
    void* host_view = tvk_buf_view(host_alloc, 0, 64);
    if (!host_view) {
        std::cerr << "FAIL: host view" << std::endl;
        return 1;
    }
    std::cout << "OK: Host allocation created" << std::endl;

    // Allocate device buffers for test
    void* dev_alloc1 = tvk_alloc_device(1024);
    void* dev_alloc2 = tvk_alloc_device(1024);
    void* dev_alloc3 = tvk_alloc_device(1024);
    if (!dev_alloc1 || !dev_alloc2 || !dev_alloc3) {
        std::cerr << "FAIL: device allocation" << std::endl;
        return 1;
    }
    void* view1 = tvk_buf_view(dev_alloc1, 0, 1024);
    void* view2 = tvk_buf_view(dev_alloc2, 0, 1024);
    void* view3 = tvk_buf_view(dev_alloc3, 0, 1024);
    if (!view1 || !view2 || !view3) {
        std::cerr << "FAIL: device views" << std::endl;
        return 1;
    }
    std::cout << "OK: Device allocations created" << std::endl;

    // Test synchronous dispatch with a simple shader
    unsigned long long handles[] = { (unsigned long long)view1, (unsigned long long)view2, (unsigned long long)view3 };
    std::cout << "Calling tvk_dispatch with shader 'rmsnorm'..." << std::endl;
    
    int rc = tvk_dispatch("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch sync: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch sync succeeded" << std::endl;

    // Test async dispatch
    std::cout << "Calling tvk_dispatch_async with shader 'rmsnorm'..." << std::endl;
    rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch_async: " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch_async succeeded" << std::endl;

    // Submit
    std::cout << "Calling tvk_submit..." << std::endl;
    tvk_submit();
    std::cout << "OK: tvk_submit succeeded" << std::endl;

    // Test second async dispatch
    std::cout << "Calling tvk_dispatch_async (2nd) with shader 'rmsnorm'..." << std::endl;
    rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch_async (2nd): " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch_async (2nd) succeeded" << std::endl;

    tvk_submit();
    std::cout << "OK: tvk_submit (2nd) succeeded" << std::endl;

    // Test third async dispatch
    std::cout << "Calling tvk_dispatch_async (3rd) with shader 'rmsnorm'..." << std::endl;
    rc = tvk_dispatch_async("rmsnorm", handles, 3, nullptr, 0, 1, 1, 1);
    if (rc != 0) {
        std::cerr << "FAIL: tvk_dispatch_async (3rd): " << tvk_last_error() << std::endl;
        return 1;
    }
    std::cout << "OK: tvk_dispatch_async (3rd) succeeded" << std::endl;

    tvk_submit();
    std::cout << "OK: tvk_submit (3rd) succeeded" << std::endl;

    dlclose(handle);
    std::cout << "\n=== ALL EXTENSION TESTS PASSED ===" << std::endl;
    return 0;
}