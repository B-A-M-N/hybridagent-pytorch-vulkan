#include "vk_backend.h"
#include "spirv_reflect.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace tvk {

Backend& Backend::instance() {
  static Backend b;
  return b;
}

uint32_t Backend::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  throw std::runtime_error("[tvk] no Vulkan memory type for flags " +
                           std::to_string((uint32_t)props));
}

uint64_t Backend::vram_total() const {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
  for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
    if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      return mp.memoryHeaps[h].size;
    }
  }
  return mp.memoryHeapCount ? mp.memoryHeaps[0].size : 0;
}

uint64_t Backend::vram_used() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return bytes_;
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
  std::cout << "[tvk] physical device " << device_index << ": " << device_name_
            << " (Vulkan " << VK_VERSION_MAJOR(props.apiVersion) << "."
            << VK_VERSION_MINOR(props.apiVersion) << ")\n";

  uint32_t qcount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qp(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qcount, qp.data());
  qfam_ = 0;
  for (uint32_t i = 0; i < qcount; i++) {
    if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam_ = i; break; }
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo q{};
  q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  q.queueFamilyIndex = qfam_;
  q.queueCount = 1;
  q.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &q;

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

  ready_ = true;
  return true;
}

std::vector<uint32_t> Backend::load_shader_unlocked(const std::string& name) {
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

std::vector<uint32_t> Backend::load_shader(const std::string& name) {
  std::lock_guard<std::mutex> lk(mutex_);
  return load_shader_unlocked(name);
}

Buffer Backend::create_device_buffer(VkDeviceSize size) {
  std::lock_guard<std::mutex> lk(mutex_);
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  Buffer b;
  if (vkCreateBuffer(device_, &bi, nullptr, &b.buffer) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateBuffer failed (" +
                             std::to_string(size) + " bytes)");
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, b.buffer, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(device_, &ai, nullptr, &b.memory) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkAllocateMemory failed (device-local)");
  }
  vkBindBufferMemory(device_, b.buffer, b.memory, 0);
  b.size = size;
  bytes_ += req.size;
  return b;
}

Buffer Backend::create_host_buffer(VkDeviceSize size) {
  std::lock_guard<std::mutex> lk(mutex_);
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  Buffer b;
  if (vkCreateBuffer(device_, &bi, nullptr, &b.buffer) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateBuffer(host) failed");
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, b.buffer, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(device_, &ai, nullptr, &b.memory) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkAllocateMemory failed (host)");
  }
  vkBindBufferMemory(device_, b.buffer, b.memory, 0);
  b.size = size;
  return b;
}

void* Backend::map(Buffer& b) {
  void* p = nullptr;
  vkMapMemory(device_, b.memory, 0, b.size, 0, &p);
  return p;
}

void Backend::unmap(Buffer& b) {
  if (b.memory != VK_NULL_HANDLE) vkUnmapMemory(device_, b.memory);
}

void Backend::destroy_buffer(Buffer& b) {
  std::lock_guard<std::mutex> lk(mutex_);
  alloc_gen_++;
  if (b.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, b.memory, nullptr);
    b.memory = VK_NULL_HANDLE;
  }
  if (b.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, b.buffer, nullptr);
    b.buffer = VK_NULL_HANDLE;
  }
}

void Backend::ensure_cmd() {
  if (cmd_ != VK_NULL_HANDLE) return;
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = cmd_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  vkAllocateCommandBuffers(device_, &ai, &cmd_);
}

void Backend::copy_to_device(Buffer& dst, Buffer& src) {
  std::lock_guard<std::mutex> lk(mutex_);
  ensure_cmd();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = std::min(dst.size, src.size);
  vkCmdCopyBuffer(cmd_, src.buffer, dst.buffer, 1, &r);
  vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkResetCommandBuffer(cmd_, 0);
}

void Backend::copy_from_device(Buffer& dst, Buffer& src) {
  std::lock_guard<std::mutex> lk(mutex_);
  ensure_cmd();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = std::min(dst.size, src.size);
  vkCmdCopyBuffer(cmd_, src.buffer, dst.buffer, 1, &r);
  vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkResetCommandBuffer(cmd_, 0);
}



void Backend::dispatch(const std::string& shader,
                       const std::vector<VkBuffer>& buffers,
                       const void* push, uint32_t push_size,
                       uint32_t gx, uint32_t gy, uint32_t gz, bool sync) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!ready_) throw std::runtime_error("[tvk] backend not initialised");

  Pipeline* p = nullptr;
  {
    auto it = pipelines_.find(shader);
    if (it == pipelines_.end()) {
      auto spv = load_shader_unlocked(shader);

      VkShaderModuleCreateInfo smi{};
      smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      smi.codeSize = spv.size() * 4;
      smi.pCode = spv.data();
      VkShaderModule sm;
      if (vkCreateShaderModule(device_, &smi, nullptr, &sm) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] vkCreateShaderModule failed: " + shader);
      }
      // The shader may declare more bindings than this dispatch passes (e.g.
      // scratch buffers). Allocate for the shader's own count, otherwise RADV
      // returns undefined results instead of an error.
      uint32_t n_decl = (uint32_t)spirv_max_binding(spv);
      if (n_decl == 0u) n_decl = (uint32_t)buffers.size();
      uint32_t n_to_bind = (uint32_t)buffers.size();
      if (n_decl < n_to_bind) n_decl = n_to_bind;

      std::vector<VkDescriptorSetLayoutBinding> binds;
      for (uint32_t i = 0; i < n_decl; i++) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = i;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds.push_back(b);
      }
      VkDescriptorSetLayoutCreateInfo dli{};
      dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      dli.bindingCount = (uint32_t)binds.size();
      dli.pBindings = binds.data();
      VkDescriptorSetLayout dsl;
      if (vkCreateDescriptorSetLayout(device_, &dli, nullptr, &dsl) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] descriptor set layout failed: " + shader);
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
        throw std::runtime_error("[tvk] pipeline layout failed: " + shader);
      }
      VkDescriptorPoolSize dps{};
      // The pool must be able to hold `maxSets` sets, each of which consumes
      // `n_decl` storage-buffer descriptors. Sizing this as n_decl*4 while
      // declaring maxSets=4096 exhausted the pool after 4 sets and produced
      // VK_ERROR_OUT_OF_POOL_MEMORY partway through a forward pass.
      dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      dps.descriptorCount = n_decl * max_desc_sets_;
      VkDescriptorPoolCreateInfo dpi{};
      dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      dpi.maxSets = max_desc_sets_;
      dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
      dpi.poolSizeCount = 1;
      dpi.pPoolSizes = &dps;
      VkDescriptorPool pool;
      if (vkCreateDescriptorPool(device_, &dpi, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] descriptor pool failed: " + shader);
      }
      VkComputePipelineCreateInfo cpi{};
      cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      cpi.stage.module = sm;
      cpi.stage.pName = "main";
      cpi.layout = pl;
      Pipeline np;
      if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                   &np.pipeline) != VK_SUCCESS) {
        throw std::runtime_error("[tvk] compute pipeline failed: " + shader);
      }
      std::cerr << "[tvk] pipeline '" << shader << "' declared bindings="
                << n_decl << " passed=" << n_to_bind << "\n";
      np.layout = pl;
      np.dsl = dsl;
      np.pool = pool;
      np.n_bindings = n_decl;
      np.push_size = push_size;
      auto ins = pipelines_.emplace(shader, np);
      p = &ins.first->second;
    } else {
      p = &it->second;
    }
  }

  auto& bnd = bound_[shader];
  // Rebind when the buffer list changed. The descriptor set is NOT freed and
  // re-allocated: a previously queued (async) dispatch may still reference it,
  // and freeing it under the GPU corrupts results and can lose the context.
  // A new set is allocated instead, so in-flight work keeps its own.
  bool need = bnd.set == VK_NULL_HANDLE || bnd.buffers != buffers ||
              bnd.n != buffers.size();
  if (need) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = p->pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p->dsl;
    if (vkAllocateDescriptorSets(device_, &ai, &bnd.set) != VK_SUCCESS) {
      throw std::runtime_error("[tvk] vkAllocateDescriptorSets failed: " + shader);
    }
    bnd.gen = alloc_gen_;
    bnd.live = true;
    std::vector<VkWriteDescriptorSet> w(buffers.size());
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
      infos[i].buffer = buffers[i];
      infos[i].offset = 0;
      infos[i].range = VK_WHOLE_SIZE;
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = bnd.set;
      w[i].dstBinding = (uint32_t)i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    bnd.buffers = buffers;
    bnd.n = (uint32_t)buffers.size();
    bnd.gen = alloc_gen_;
  }

  ensure_cmd();

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd_, &bi);
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1,
                          &bnd.set, 0, nullptr);
  if (push_size) {
    vkCmdPushConstants(cmd_, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                       push);
  }
  // Flatten a 3-D grid into 1-D. RADV on this driver ignores the Y and Z
  // components of vkCmdDispatch for compute pipelines in a way that silently
  // collapses multi-row grids to a single workgroup, so every dispatch goes out
  // as (gx*gy*gz, 1, 1) and each shader derives its own 2-D/3-D indices from
  // gl_WorkGroupID.x. See dispatch_linear() in vk_cabi.cpp.
  vkCmdDispatch(cmd_, gx * gy * gz, 1, 1);
  vkEndCommandBuffer(cmd_);

  // Submit whatever has accumulated. `sync == true` waits for the queue to go
  // idle so the caller can read results immediately; `sync == false` leaves the
  // work queued so several dispatches can be submitted as one batch.
  //
  // The command buffer may only be RESET once the GPU has finished with it.
  // Resetting it while a previous async submission is still executing is what
  // produced `GPUVM fault` / `context is lost` -- submission is not completion.
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  if (sync) {
    vkQueueWaitIdle(queue_);
    vkResetCommandBuffer(cmd_, 0);
    cmd_inflight_ = false;
    // The queue has drained, so the descriptor sets allocated for this
    // dispatch are no longer referenced by in-flight work. Free them here --
    // otherwise they accumulate one per dispatch until the pool is exhausted,
    // after which vkAllocateDescriptorSets fails and leaves the command buffer
    // mid-record. That exhaustion is the source of the intermittent corruption
    // and the SIGSEGV.
    for (auto& kv : bound_) {
      auto& b = kv.second;
      if (b.live && b.set != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device_, pipelines_[kv.first].pool, 1, &b.set);
        b.set = VK_NULL_HANDLE;
        b.live = false;
        b.buffers.clear();
        b.n = 0;
      }
    }
  } else {
    cmd_inflight_ = true;
  }
  dispatches_++;
}

void Backend::submit() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (device_ == VK_NULL_HANDLE) return;
  if (cmd_inflight_) {
    // The previous async submission must complete before the command buffer can
    // be reset and reused.
    vkQueueWaitIdle(queue_);
    vkResetCommandBuffer(cmd_, 0);
    cmd_inflight_ = false;
  }
  // The queue has drained, so every descriptor set allocated for queued work is
  // now free to reuse. Recycle them instead of leaking a set per dispatch --
  // without this the pool is exhausted after a few hundred dispatches.
  for (auto& kv : bound_) {
    auto& b = kv.second;
    if (b.live && b.set != VK_NULL_HANDLE) {
      vkFreeDescriptorSets(device_, pipelines_[kv.first].pool, 1, &b.set);
      b.set = VK_NULL_HANDLE;
      b.live = false;
      b.buffers.clear();
      b.n = 0;
    }
  }
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
  if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
  vkDestroyDevice(device_, nullptr);
  vkDestroyInstance(instance_, nullptr);
}

}  // namespace tvk
