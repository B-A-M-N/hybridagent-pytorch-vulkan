#include "vulkan_engine.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace torch_vulkan {

VulkanEngine& VulkanEngine::instance() {
    static VulkanEngine eng;
    return eng;
}

VulkanEngine::VulkanEngine() {
    // The shader directory is normally set at import time via
    // setShaderDir() (driven by torch_vulkan._set_shader_dir in __init__.py).
    // As a fallback for direct/standalone use, honour TORCH_VULKAN_SHADER_DIR
    // so the path is never hardcoded to a developer's machine.
    if (const char* env = std::getenv("TORCH_VULKAN_SHADER_DIR")) {
        shaderDir_ = env;
        if (!shaderDir_.empty() && shaderDir_.back() != '/') {
            shaderDir_ += '/';
        }
    }
    initVulkan();
}

VulkanEngine::~VulkanEngine() {
    vkDeviceWaitIdle(device_);

    for (auto& [name, cp] : pipelineCache_) {
        vkDestroyPipeline(device_, cp.pipeline, nullptr);
        vkDestroyPipelineLayout(device_, cp.layout, nullptr);
        vkDestroyDescriptorSetLayout(device_, cp.descSetLayout, nullptr);
    }
    for (auto& [buf, mem] : bufferMemory_) {
        vkDestroyBuffer(device_, buf, nullptr);
        vkFreeMemory(device_, mem, nullptr);
    }
    vkDestroyDescriptorPool(device_, descPool_, nullptr);
    vkDestroyFence(device_, fence_, nullptr);
    vkDestroyCommandPool(device_, cmdPool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

void VulkanEngine::initVulkan() {
    // Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "torch-vulkan";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &instance_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan instance");

    // Physical device (first one)
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance_, &devCount, nullptr);
    if (devCount == 0) throw std::runtime_error("No Vulkan devices");
    std::vector<VkPhysicalDevice> devices(devCount);
    vkEnumeratePhysicalDevices(instance_, &devCount, devices.data());
    physDevice_ = devices[0];

    // Find compute queue family
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &qfCount, qfProps.data());
    queueFamily_ = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamily_ = i;
            break;
        }
    }
    if (queueFamily_ == UINT32_MAX) throw std::runtime_error("No compute queue");

    // Logical device + queue
    float qPriority = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = queueFamily_;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &qPriority;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    if (vkCreateDevice(physDevice_, &devInfo, nullptr, &device_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create device");
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    // Command pool + reusable command buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuf_);

    // Fence
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device_, &fenceInfo, nullptr, &fence_);

    // Descriptor pool (large enough for all ops)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1024;

    VkDescriptorPoolCreateInfo dpInfo{};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 256;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    dpInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    vkCreateDescriptorPool(device_, &dpInfo, nullptr, &descPool_);
}

uint32_t VulkanEngine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

VkBuffer VulkanEngine::createBuffer(VkDeviceSize size, void** mappedPtr) {
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    vkCreateBuffer(device_, &bufInfo, nullptr, &buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory memory;
    vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(device_, buffer, memory, 0);

    if (mappedPtr) {
        vkMapMemory(device_, memory, 0, size, 0, mappedPtr);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    bufferMemory_[buffer] = memory;
    return buffer;
}

void VulkanEngine::destroyBuffer(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bufferMemory_.find(buffer);
    if (it != bufferMemory_.end()) {
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, it->second, nullptr);
        bufferMemory_.erase(it);
    }
}


static VkDeviceSize roundUpPow2(VkDeviceSize size) {
    VkDeviceSize p = 1;
    while (p < size) p <<= 1;
    return p;
}

VulkanEngine::PooledBuffer VulkanEngine::acquireBuffer(VkDeviceSize size) {
    VkDeviceSize bucket = roundUpPow2(std::max(size, (VkDeviceSize)256));

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& pool = bufferPool_[bucket];
        if (!pool.empty()) {
            auto buf = pool.back();
            pool.pop_back();
            return buf;
        }
    }

    // No pooled buffer available — create new one at bucket size
    void* mapped = nullptr;
    VkBuffer buffer = createBuffer(bucket, &mapped);

    std::lock_guard<std::mutex> lock(mutex_);
    auto mem = bufferMemory_[buffer];
    return PooledBuffer{buffer, mem, mapped, bucket};
}

void VulkanEngine::releaseBuffer(PooledBuffer& buf) {
    std::lock_guard<std::mutex> lock(mutex_);
    bufferPool_[buf.capacity].push_back(buf);
}

std::vector<uint32_t> VulkanEngine::loadShader(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shaderCache_.find(name);
    if (it != shaderCache_.end()) return it->second;

    std::string path = shaderDir_ + name + ".spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Failed to load shader: " + path);

    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);

    shaderCache_[name] = spirv;
    return spirv;
}

VkShaderModule VulkanEngine::createShaderModule(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirv.size() * sizeof(uint32_t);
    info.pCode = spirv.data();

    VkShaderModule module;
    vkCreateShaderModule(device_, &info, nullptr, &module);
    return module;
}

CachedPipeline& VulkanEngine::getOrCreatePipeline(
    const std::string& shaderName, uint32_t numBuffers, uint32_t pushConstantSize) {

    std::string key = shaderName + ":" + std::to_string(numBuffers) + ":" + std::to_string(pushConstantSize);

    auto it = pipelineCache_.find(key);
    if (it != pipelineCache_.end()) return it->second;

    auto spirv = loadShader(shaderName);
    VkShaderModule shaderModule = createShaderModule(spirv);

    // Descriptor set layout: N storage buffers
    std::vector<VkDescriptorSetLayoutBinding> bindings(numBuffers);
    for (uint32_t i = 0; i < numBuffers; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = numBuffers;
    dslInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descSetLayout;
    vkCreateDescriptorSetLayout(device_, &dslInfo, nullptr, &descSetLayout);

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descSetLayout;
    if (pushConstantSize > 0) {
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
    }

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device_, &plInfo, nullptr, &pipelineLayout);

    // Compute pipeline
    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = shaderModule;
    cpInfo.stage.pName = "main";
    cpInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline);

    vkDestroyShaderModule(device_, shaderModule, nullptr);

    CachedPipeline cp{pipeline, pipelineLayout, descSetLayout, numBuffers, pushConstantSize};
    pipelineCache_[key] = cp;
    return pipelineCache_[key];
}

void VulkanEngine::dispatch(
    CachedPipeline& pipeline,
    const std::vector<VkBuffer>& buffers,
    const std::vector<VkDeviceSize>& bufferSizes,
    const void* pushConstants,
    uint32_t pushConstantSize,
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ) {

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = descPool_;
    dsAllocInfo.descriptorSetCount = 1;
    dsAllocInfo.pSetLayouts = &pipeline.descSetLayout;

    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device_, &dsAllocInfo, &descSet);

    // Update descriptor set with buffer bindings
    std::vector<VkDescriptorBufferInfo> bufInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
        bufInfos[i] = {buffers[i], 0, bufferSizes[i]};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Record command buffer
    vkResetCommandBuffer(cmdBuf_, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf_, &beginInfo);

    vkCmdBindPipeline(cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout, 0, 1, &descSet, 0, nullptr);

    if (pushConstantSize > 0) {
        vkCmdPushConstants(cmdBuf_, pipeline.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);
    }

    vkCmdDispatch(cmdBuf_, groupCountX, groupCountY, groupCountZ);

    // Memory barrier for compute
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkEndCommandBuffer(cmdBuf_);

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf_;

    vkResetFences(device_, 1, &fence_);
    vkQueueSubmit(queue_, 1, &submitInfo, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);

    // Free descriptor set for reuse
    vkFreeDescriptorSets(device_, descPool_, 1, &descSet);
}

} // namespace torch_vulkan
