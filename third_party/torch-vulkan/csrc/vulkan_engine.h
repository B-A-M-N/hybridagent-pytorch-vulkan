#pragma once
/**
 * VulkanEngine — Raw Vulkan dispatch, no Kompute ceremony.
 * 
 * Pre-creates pipelines per shader, caches descriptor sets,
 * reuses command buffers. Zero per-call allocation.
 */

#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>

namespace torch_vulkan {

struct CachedPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descSetLayout;
    uint32_t numBuffers;
    uint32_t pushConstantSize;
};

class VulkanEngine {
public:
    static VulkanEngine& instance();

    // Get raw Vulkan handles
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physDevice_; }
    VkQueue computeQueue() const { return queue_; }

    // Get or create a cached compute pipeline for a SPIR-V shader
    CachedPipeline& getOrCreatePipeline(
        const std::string& shaderName,
        uint32_t numBuffers,
        uint32_t pushConstantSize);

    // Dispatch: bind buffers, push constants, dispatch workgroups
    // Uses pre-allocated command buffer from pool
    void dispatch(
        CachedPipeline& pipeline,
        const std::vector<VkBuffer>& buffers,
        const std::vector<VkDeviceSize>& bufferSizes,
        const void* pushConstants,
        uint32_t pushConstantSize,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ);

    // Buffer pool: get a buffer of at least `size` bytes, reuse from pool
    struct PooledBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped;
        VkDeviceSize capacity;
    };
    PooledBuffer acquireBuffer(VkDeviceSize size);
    void releaseBuffer(PooledBuffer& buf);

    // Legacy (non-pooled) — kept for compatibility
    VkBuffer createBuffer(VkDeviceSize size, void** mappedPtr);
    void destroyBuffer(VkBuffer buffer);

    // Load SPIR-V shader
    std::vector<uint32_t> loadShader(const std::string& name);
    void setShaderDir(const std::string& path) { shaderDir_ = path; }

    ~VulkanEngine();

private:
    VulkanEngine();
    void initVulkan();
    VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    VkInstance instance_;
    VkPhysicalDevice physDevice_;
    VkDevice device_;
    VkQueue queue_;
    uint32_t queueFamily_;
    VkCommandPool cmdPool_;
    VkCommandBuffer cmdBuf_;  // Single reusable command buffer
    VkFence fence_;
    VkDescriptorPool descPool_;

    std::string shaderDir_;
    std::unordered_map<std::string, std::vector<uint32_t>> shaderCache_;
    std::unordered_map<std::string, CachedPipeline> pipelineCache_;
    std::unordered_map<VkBuffer, VkDeviceMemory> bufferMemory_;
    // Buffer pool: buckets by size (rounded up to power of 2)
    std::unordered_map<VkDeviceSize, std::vector<PooledBuffer>> bufferPool_;
    std::mutex mutex_;
};

} // namespace torch_vulkan
