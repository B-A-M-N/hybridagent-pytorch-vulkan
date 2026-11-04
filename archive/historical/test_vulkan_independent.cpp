// Independent Vulkan compute test - completely separate from HybridAgent
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>

int main() {
    std::cout << "=== Independent Vulkan Compute Test ===" << std::endl;

    // 1. Create instance
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "independent_vulkan_test";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance instance;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateInstance" << std::endl;
        return 1;
    }
    std::cout << "OK: Instance created" << std::endl;

    // 2. Enumerate physical devices
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    if (n == 0) {
        std::cerr << "FAIL: no Vulkan physical devices" << std::endl;
        return 1;
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());
    VkPhysicalDevice phys = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    std::cout << "OK: Physical device: " << props.deviceName << std::endl;

    // 3. Find compute queue family
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, qp.data());
    
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qcount; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }
    if (qfam == UINT32_MAX) {
        std::cerr << "FAIL: no compute queue family" << std::endl;
        return 1;
    }
    std::cout << "OK: Compute queue family: " << qfam << std::endl;

    // 4. Create device
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q{};
    q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q.queueFamilyIndex = qfam;
    q.queueCount = 1;
    q.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &q;

    VkDevice device;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateDevice" << std::endl;
        return 1;
    }
    std::cout << "OK: Device created" << std::endl;

    VkQueue queue;
    vkGetDeviceQueue(device, qfam, 0, &queue);

    // 5. Create command pool
    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfam;

    VkCommandPool cmd_pool;
    if (vkCreateCommandPool(device, &cpi, nullptr, &cmd_pool) != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateCommandPool" << std::endl;
        return 1;
    }
    std::cout << "OK: Command pool created" << std::endl;

    // 6. Allocate command buffer
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &ai, &cmd) != VK_SUCCESS) {
        std::cerr << "FAIL: vkAllocateCommandBuffers" << std::endl;
        return 1;
    }
    std::cout << "OK: Command buffer allocated: " << cmd << std::endl;

    // 7. Begin command buffer
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkResult begin_result = vkBeginCommandBuffer(cmd, &bi);
    if (begin_result != VK_SUCCESS) {
        std::cerr << "FAIL: vkBeginCommandBuffer: " << begin_result << std::endl;
        return 1;
    }
    std::cout << "OK: vkBeginCommandBuffer succeeded" << std::endl;

    // 8. End command buffer
    vkEndCommandBuffer(cmd);
    std::cout << "OK: vkEndCommandBuffer succeeded" << std::endl;

    // 9. Submit and wait
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    if (vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cerr << "FAIL: vkQueueSubmit" << std::endl;
        return 1;
    }
    std::cout << "OK: vkQueueSubmit succeeded" << std::endl;

    if (vkQueueWaitIdle(queue) != VK_SUCCESS) {
        std::cerr << "FAIL: vkQueueWaitIdle" << std::endl;
        return 1;
    }
    std::cout << "OK: vkQueueWaitIdle succeeded" << std::endl;

    // 10. Reset command buffer
    VkResult reset_result = vkResetCommandBuffer(cmd, 0);
    if (reset_result != VK_SUCCESS) {
        std::cerr << "FAIL: vkResetCommandBuffer: " << reset_result << std::endl;
        return 1;
    }
    std::cout << "OK: vkResetCommandBuffer succeeded" << std::endl;

    // 11. Begin again
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        std::cerr << "FAIL: vkBeginCommandBuffer (2nd time)" << std::endl;
        return 1;
    }
    std::cout << "OK: vkBeginCommandBuffer (2nd time) succeeded" << std::endl;
    vkEndCommandBuffer(cmd);
    std::cout << "OK: vkEndCommandBuffer (2nd time) succeeded" << std::endl;

    // Cleanup
    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}