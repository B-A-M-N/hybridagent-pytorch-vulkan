#include "vk_backend.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tvk {

Backend& Backend::instance() {
  static Backend b;
  return b;
}

uint32_t Backend::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  throw std::runtime_error("no suitable Vulkan memory type for flags " +
                           std::to_string((uint32_t)props));
}

uint64_t Backend::vram_total() const {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
  // Prefer a device-local heap.
  for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
    if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      return mp.memoryHeaps[h].size;
    }
  }
  return mp.memoryHeapCount ? mp.memoryHeaps[0].size : 0;
}

uint64_t Backend::vram_used() const {
  // Vulkan has no universal used-memory query; report our own allocation
  // accounting, which is the number that matters for this task.
  uint64_t total = 0;
  std::lock_guard<std::mutex> lk(mutex_);
  for (auto& kv : residents_) total += kv.second.size;
  return total;
}

bool Backend::init(uint32_t device_index) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (ready_) return true;

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "torch_vulkan_ext";
  app.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;

  if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
    std::cerr << "[tvk] vkCreateInstance failed\n";
    return false;
  }

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(instance_, &n, nullptr);
  if (n == 0) {
    std::cerr << "[tvk] no Vulkan physical devices\n";
    return false;
  }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(instance_, &n, devs.data());
  if (device_index >= n) device_index = 0;

  phys_ = devs[device_index];

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(phys_, &props);
  device_name_ = props.deviceName;
  std::cout << "[tvk] selected physical device " << device_index << ": "
            << device_name_ << "\n";

  float prio = 1.0f;
  VkDeviceQueueCreateInfo q{};
  q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  q.queueFamilyIndex = qfam_;
  q.queueCount = 1;
  q.pQueuePriorities = &prio;

  // Queue family: prefer one advertising compute.
  uint32_t qcount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, qprops.data());
  qfam_ = 0;
  for (uint32_t i = 0; i < qcount; i++) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam_ = i;
      break;
    }
  }
  q.queueFamilyIndex = qfam_;

  const char* exts[] = {"VK_KHR_16bit_storage", "VK_KHR_8bit_storage",
                        "VK_KHR_shader_float16_int8"};
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &q;
  dci.enabledExtensionCount = 0;   // resolved lazily below
  dci.ppEnabledExtensionNames = nullptr;

  if (vkCreateDevice(phys_, &dci, nullptr, &device_) != VK_SUCCESS) {
    std::cerr << "[tvk] vkCreateDevice failed\n";
    return false;
  }
  vkGetDeviceQueue(device_, qfam_, 0, &queue_);

  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = qfam_;
  if (vkCreateCommandPool(device_, &cpi, nullptr, &cmd_pool_) != VK_SUCCESS) {
    std::cerr << "[tvk] vkCreateCommandPool failed\n";
    return false;
  }

  VkQueryPoolCreateInfo qpi{};
  qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = 64;
  vkCreateQueryPool(device_, &qpi, nullptr, &timestamps_);

  ready_ = true;
  std::cout << "[tvk] backend ready, queue family " << qfam_ << "\n";
  return true;
}

void Backend::ensure_command_buffer() {
  if (cmd_ != VK_NULL_HANDLE) return;
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = cmd_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  vkAllocateCommandBuffers(device_, &ai, &cmd_);
}

VkBuffer Backend::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer b;
  if (vkCreateBuffer(device_, &bi, nullptr, &b) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateBuffer failed for " + std::to_string(size) +
                             " bytes");
  }
  return b;
}

bool Backend::alloc_bind(VkBuffer buf, VkDeviceMemory* mem, VkMemoryPropertyFlags props) {
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, buf, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
  if (vkAllocateMemory(device_, &ai, nullptr, mem) != VK_SUCCESS) return false;
  vkBindBufferMemory(device_, buf, *mem, 0);
  return true;
}

void Backend::submit_and_wait() {
  vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkResetCommandBuffer(cmd_, 0);
}

std::vector<uint32_t> Backend::load_shader(const std::string& name) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = shader_cache_.find(name);
  if (it != shader_cache_.end()) return it->second;
  std::string path = shader_dir_ + name + ".spv";
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    throw std::runtime_error("[tvk] cannot open shader " + path);
  }
  size_t bytes = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> spv(bytes / 4);
  f.read(reinterpret_cast<char*>(spv.data()), bytes);
  shader_cache_[name] = spv;
  return spv;
}

Pipeline* Backend::pipeline(const std::string& shader, uint32_t n_bindings,
                            uint32_t push_size) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = pipelines_.find(shader);
  if (it != pipelines_.end()) return &it->second;

  auto spv = load_shader(shader);

  VkShaderModuleCreateInfo smi{};
  smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smi.codeSize = spv.size() * 4;
  smi.pCode = spv.data();
  VkShaderModule sm;
  if (vkCreateShaderModule(device_, &smi, nullptr, &sm) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateShaderModule failed for " + shader);
  }

  std::vector<VkDescriptorSetLayoutBinding> binds;
  for (uint32_t i = 0; i < n_bindings; i++) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = i;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds.push_back(b);
  }

  VkDescriptorSetLayoutCreateInfo dli{};
  dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dli.bindingCount = n_bindings;
  dli.pBindings = binds.data();
  VkDescriptorSetLayout dsl;
  if (vkCreateDescriptorSetLayout(device_, &dli, nullptr, &dsl) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateDescriptorSetLayout failed for " +
                             shader);
  }

  VkPipelineLayoutCreateInfo pli{};
  pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &dsl;
  VkPushConstantRange pcr{};
  if (push_size) {
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_size;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
  }
  VkPipelineLayout pl;
  if (vkCreatePipelineLayout(device_, &pli, nullptr, &pl) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreatePipelineLayout failed for " + shader);
  }

  // One descriptor pool per pipeline with enough slots for every binding.
  VkDescriptorPoolSize dps{};
  dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dps.descriptorCount = n_bindings * 4;
  VkDescriptorPoolCreateInfo dpi{};
  dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpi.maxSets = 4;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &dps;
  VkDescriptorPool pool;
  if (vkCreateDescriptorPool(device_, &dpi, nullptr, &pool) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateDescriptorPool failed for " + shader);
  }

  VkPipelineCache pc = VK_NULL_HANDLE;
  VkComputePipelineCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = sm;
  cpi.stage.pName = "main";
  cpi.layout = pl;

  Pipeline p;
  if (vkCreateComputePipelines(device_, pc, 1, &cpi, nullptr, &p.pipeline) !=
      VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateComputePipelines failed for " + shader);
  }
  p.layout = pl;
  p.dsl = dsl;
  p.pool = pool;
  p.n_bindings = n_bindings;
  p.push_size = push_size;
  p.used = true;

  auto res = pipelines_.emplace(shader, p);
  return &res.first->second;
}

void Backend::dispatch(const std::string& shader,
                       const std::vector<VkBuffer>& buffers,
                       const std::vector<VkDeviceSize>& sizes,
                       const void* push, uint32_t push_size,
                       uint32_t gx, uint32_t gy, uint32_t gz) {
  std::lock_guard<std::mutex> lk(mutex_);
  Pipeline* p = pipeline(shader, (uint32_t)buffers.size(), push_size);
  ensure_command_buffer();

  // (Re)bind the descriptor set when the buffer list changed.
  auto& bs = bound_sets_[shader];
  bool need = bs.set == VK_NULL_HANDLE || bs.n != buffers.size() ||
              bs.buffers != buffers;
  if (need) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = p->pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p->dsl;
    if (bs.set == VK_NULL_HANDLE) {
      if (vkAllocateDescriptorSets(device_, &ai, &bs.set) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] vkAllocateDescriptorSets failed for " +
                                 shader);
      }
    } else {
      vkResetDescriptorPool(device_, p->pool, 0);
      if (vkAllocateDescriptorSets(device_, &ai, &bs.set) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] vkAllocateDescriptorSets failed for " +
                                 shader);
      }
    }
    std::vector<VkWriteDescriptorSet> w(buffers.size());
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
      infos[i].buffer = buffers[i];
      infos[i].offset = 0;
      infos[i].range = sizes.empty() ? VK_WHOLE_SIZE : sizes[i];
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = bs.set;
      w[i].dstBinding = (uint32_t)i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    bs.buffers = buffers;
    bs.n = (uint32_t)buffers.size();
  }

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd_, &bi);
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1,
                          &bs.set, 0, nullptr);
  if (push_size) {
    vkCmdPushConstants(cmd_, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                       push);
  }
  vkCmdDispatch(cmd_, gx, gy, gz);
  dispatches_++;
  submit_and_wait();
}

Backend::Staging* Backend::acquire_staging(VkDeviceSize size) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = staging_free_.find(size);
  if (it != staging_free_.end() && !it->second.empty()) {
    auto* s = it->second.back();
    it->second.pop_back();
    return s;
  }
  Staging* s = new Staging();
  s->size = size;
  VkBufferUsageFlags dev_usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  s->device_buf = create_buffer(size, dev_usage);
  if (!alloc_bind(s->device_buf, &s->device_mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    throw std::runtime_error("[tvk] staging device alloc failed");
  }
  s->host_buf = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  if (!alloc_bind(s->host_buf, &s->host_mem,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    throw std::runtime_error("[tvk] staging host alloc failed");
  }
  vkMapMemory(device_, s->host_mem, 0, size, 0, &s->mapped);
  allocs_ += 2;
  staging_pool_.push_back(s);
  return s;
}

void Backend::release_staging(Staging* s) {
  if (!s) return;
  std::lock_guard<std::mutex> lk(mutex_);
  staging_free_[s->size].push_back(s);
}

void Backend::copy_to_device(Staging* s, const void* src) {
  std::lock_guard<std::mutex> lk(mutex_);
  ensure_command_buffer();
  memcpy(s->mapped, src, s->size);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = s->size;
  vkCmdCopyBuffer(cmd_, s->host_buf, s->device_buf, 1, &r);
  submit_and_wait();
}

void Backend::copy_from_device(Staging* s, void* dst) {
  std::lock_guard<std::mutex> lk(mutex_);
  ensure_command_buffer();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = s->size;
  vkCmdCopyBuffer(cmd_, s->device_buf, s->host_buf, 1, &r);
  submit_and_wait();
  memcpy(dst, s->mapped, s->size);
}

bool Backend::register_resident(void* ptr, ResidentBuffer* out) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = residents_.find(ptr);
  if (it == residents_.end()) return false;
  *out = it->second;
  return true;
}

const ResidentBuffer* Backend::find_resident(void* ptr) const {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = residents_.find(ptr);
  return it == residents_.end() ? nullptr : &it->second;
}

void Backend::forget_resident(void* ptr) {
  std::lock_guard<std::mutex> lk(mutex_);
  residents_.erase(ptr);
}

Backend::~Backend() {
  if (device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  for (auto& kv : pipelines_) {
    vkDestroyPipeline(device_, kv.second.pipeline, nullptr);
    vkDestroyPipelineLayout(device_, kv.second.layout, nullptr);
    vkDestroyDescriptorSetLayout(device_, kv.second.dsl, nullptr);
    vkDestroyDescriptorPool(device_, kv.second.pool, nullptr);
  }
  for (auto* s : staging_pool_) {
    if (s->mapped) vkUnmapMemory(device_, s->host_mem);
    vkDestroyBuffer(device_, s->device_buf, nullptr);
    vkFreeMemory(device_, s->device_mem, nullptr);
    vkDestroyBuffer(device_, s->host_buf, nullptr);
    vkFreeMemory(device_, s->host_mem, nullptr);
    delete s;
  }
  if (timestamps_) vkDestroyQueryPool(device_, timestamps_, nullptr);
  if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
  vkDestroyDevice(device_, nullptr);
  vkDestroyInstance(instance_, nullptr);
}

}  // namespace tvk
