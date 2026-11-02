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

bool Backend::init(uint32_t device_index) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  if (ready_) {
    // Check if the device is still valid by attempting a trivial operation.
    // If the device was lost (e.g., GPU reset), we need to reinitialize.
    VkCommandBuffer test_cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = cmd_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkResult test_result = vkAllocateCommandBuffers(device_, &ai, &test_cmd);
    if (test_result == VK_ERROR_DEVICE_LOST || test_result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      std::cerr << "[tvk] Device lost, reinitializing..." << std::endl;
      ready_ = false;
    } else if (test_result == VK_SUCCESS) {
      vkFreeCommandBuffers(device_, cmd_pool_, 1, &test_cmd);
      return true;
    }
    // For other errors, fall through to reinitialize
  }

  // Cleanup any existing Vulkan resources before reinitializing
  if (instance_ != VK_NULL_HANDLE) {
    // Destroy device first (destroys command pool, buffers, etc.)
    if (device_ != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(device_);
      vkDestroyDevice(device_, nullptr);
      device_ = VK_NULL_HANDLE;
    }
    if (cmd_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, cmd_pool_, nullptr);
      cmd_pool_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
      vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
    }
    cmd_pool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    phys_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    ready_ = false;
    cmd_pool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    phys_ = VK_NULL_HANDLE;
    cmd_ = VK_NULL_HANDLE;
    cmd_inflight_ = false;
    ready_ = false;
  }

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
            << VK_VERSION_MINOR(props.apiVersion) << ".)\n";

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

Allocation::~Allocation() {
  if (backend && buffer != VK_NULL_HANDLE) {
    try {
      backend->destroy_allocation(this);
    } catch (...) {}
  }
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
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  return bytes_;
}

Allocation* Backend::create_device_allocation(VkDeviceSize size) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buf;
  if (vkCreateBuffer(device_, &bi, nullptr, &buf) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateBuffer failed (" +
                             std::to_string(size) + " bytes)");
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, buf, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory mem;
  if (vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) {
    vkDestroyBuffer(device_, buf, nullptr);
    throw std::runtime_error("[tvk] vkAllocateMemory failed (device-local)");
  }
  vkBindBufferMemory(device_, buf, mem, 0);
  bytes_ += req.size;
  Allocation* a = new Allocation(buf, mem, req.size, this);
  live_allocs_.push_back(a);
  return a;
}

Allocation* Backend::create_host_allocation(VkDeviceSize size) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buf;
  if (vkCreateBuffer(device_, &bi, nullptr, &buf) != VK_SUCCESS) {
    throw std::runtime_error("[tvk] vkCreateBuffer(host) failed");
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device_, buf, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkDeviceMemory mem;
  if (vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) {
    vkDestroyBuffer(device_, buf, nullptr);
    throw std::runtime_error("[tvk] vkAllocateMemory failed (host)");
  }
  vkBindBufferMemory(device_, buf, mem, 0);
  Allocation* a = new Allocation(buf, mem, req.size, this);
  live_allocs_.push_back(a);
  return a;
}

void Backend::destroy_allocation(Allocation* a) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  if (!a) return;
  alloc_gen_++;
  if (a->memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, a->memory, nullptr);
    a->memory = VK_NULL_HANDLE;
  }
  if (a->buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, a->buffer, nullptr);
    a->buffer = VK_NULL_HANDLE;
  }
  if (a->size > 0) bytes_ -= a->size;
  // Unregister so the Backend destructor does not free it a second time.
  // The slot is nulled, not erased: erasing would invalidate an iterator
  // if this runs during the Backend destructor's own loop over live_allocs_.
  for (size_t i = 0; i < live_allocs_.size(); i++) {
    if (live_allocs_[i] == a) {
      live_allocs_[i] = nullptr;
      break;
    }
  }
}

BufferView Backend::make_view(Allocation* alloc, VkDeviceSize offset, VkDeviceSize size) {
  if (!alloc) return BufferView{};
  alloc->add_ref();
  return BufferView(alloc, offset, size);
}

void Backend::release_view(Allocation* alloc) {
  if (alloc) alloc->release();
}

void* Backend::map(Allocation* alloc) {
  if (!alloc) return nullptr;
  void* p = nullptr;
  vkMapMemory(device_, alloc->memory, 0, alloc->size, 0, &p);
  return p;
}

void Backend::unmap(Allocation* alloc) {
  if (alloc && alloc->memory != VK_NULL_HANDLE) {
    vkUnmapMemory(device_, alloc->memory);
  }
}

void Backend::dispatch(const std::string& shader,
                       const std::vector<BufferView>& views,
                       const void* push, uint32_t push_size,
                       uint32_t gx, uint32_t gy, uint32_t gz, bool sync) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
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
      uint32_t n_decl = (uint32_t)spirv_max_binding(spv);
      if (n_decl == 0u) n_decl = (uint32_t)views.size();
      uint32_t n_to_bind = (uint32_t)views.size();
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
                << n_decl << " passed=" << views.size() << "\n";
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
  bool need = bnd.set == VK_NULL_HANDLE || bnd.views != views ||
              views.size() != bnd.views.size();
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
    std::vector<VkWriteDescriptorSet> w(views.size());
    std::vector<VkDescriptorBufferInfo> infos(views.size());
    for (size_t i = 0; i < views.size(); i++) {
      infos[i].buffer = views[i].vk_buffer();
      infos[i].offset = views[i].vk_offset();
      infos[i].range = views[i].vk_size();
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = bnd.set;
      w[i].dstBinding = (uint32_t)i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    bnd.views = views;
    bnd.gen = alloc_gen_;
  }

  // For async dispatches, allocate a new command buffer to avoid reusing
  // the command buffer while the GPU is still executing the previous dispatch.
  bool new_cmd = false;
  if (!sync) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = cmd_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd_) != VK_SUCCESS) {
      throw std::runtime_error("[tvk] vkAllocateCommandBuffers failed");
    }
    std::cerr << "[tvk] Allocated new async command buffer: " << cmd_ << std::endl;
    new_cmd = true;
  } else {
    ensure_cmd();
  }

  VkCommandBuffer cmd = cmd_;

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VkResult begin_result = vkBeginCommandBuffer(cmd, &bi);
  if (begin_result != VK_SUCCESS) {
    std::string err = "[tvk] vkBeginCommandBuffer failed: " + std::to_string(begin_result);
    std::cerr << err << std::endl;
    throw std::runtime_error(err);
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1,
                          &bnd.set, 0, nullptr);
  if (push_size) {
    vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                       push);
  }
  vkCmdDispatch(cmd, gx * gy * gz, 1, 1);
  vkEndCommandBuffer(cmd);

  // Create a submission record that retains all allocations used by this dispatch.
  // These allocations will be released when the GPU completes (in submit() or sync dispatch).
  Submission sub;
  sub.views = views;
  sub.cmd = cmd;
  sub.new_cmd = new_cmd;
  for (const auto& v : views) {
    if (v.alloc) {
      v.alloc->add_ref();
      sub.retained_allocs.push_back(v.alloc);
    }
  }
  sub.set = bnd.set;
  pending_submissions_.push_back(std::move(sub));

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  if (sync) {
    vkQueueWaitIdle(queue_);
    if (new_cmd) {
      vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
      cmd_ = VK_NULL_HANDLE;
    } else {
          // For sync dispatches, reset the command buffer so it can be reused.
          // The command buffer must be reset after vkQueueWaitIdle before it can be reused.
          VkResult reset_result = vkResetCommandBuffer(cmd_, 0);
          if (reset_result != VK_SUCCESS) {
            std::string err = "[tvk] vkResetCommandBuffer failed: " + std::to_string(reset_result);
            std::cerr << err << std::endl;
            throw std::runtime_error(err);
          }
          std::cerr << "[tvk] Reset command buffer: " << cmd_ << std::endl;
          // Keep cmd_ valid after reset - it's now ready for reuse.
          // For sync dispatches, DON'T free descriptor sets.
          // Keep them for reuse in the next dispatch with the same shader and bindings.
          // Only mark that no command is in flight.
          cmd_inflight_ = false;
        }
  } else {
    cmd_inflight_ = true;
  }
  dispatches_++;
}

void Backend::submit() {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  if (device_ == VK_NULL_HANDLE) return;
  if (cmd_inflight_) {
    vkQueueWaitIdle(queue_);
    VkResult reset_result = vkResetCommandBuffer(cmd_, 0);
    if (reset_result != VK_SUCCESS) {
      std::string err = "[tvk] vkResetCommandBuffer failed: " + std::to_string(reset_result);
      std::cerr << err << std::endl;
      throw std::runtime_error(err);
    }
    std::cerr << "[tvk] Reset command buffer in submit: " << cmd_ << std::endl;
    // Keep cmd_ valid after reset - it's now ready for reuse.
    cmd_inflight_ = false;
  }
  // NOTE: We DON'T free descriptor sets here - they are cached per shader for reuse.
  // Only free command buffers that were allocated for async dispatches.
  for (auto& sub : pending_submissions_) {
    for (auto* alloc : sub.retained_allocs) {
      if (alloc) alloc->release();
    }
    if (sub.new_cmd && sub.cmd != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, cmd_pool_, 1, &sub.cmd);
      if (sub.cmd == cmd_) {
        cmd_ = VK_NULL_HANDLE;
      }
    }
  }
  pending_submissions_.clear();
}

void Backend::copy_to_device(Allocation* dst, Allocation* src) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  ensure_cmd();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = std::min(dst->size, src->size);
  vkCmdCopyBuffer(cmd_, src->buffer, dst->buffer, 1, &r);
  vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkResetCommandBuffer(cmd_, 0);
}

void Backend::copy_from_device(Allocation* dst, Allocation* src) {
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  ensure_cmd();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  VkBufferCopy r{};
  r.srcOffset = 0;
  r.dstOffset = 0;
  r.size = std::min(dst->size, src->size);
  vkCmdCopyBuffer(cmd_, src->buffer, dst->buffer, 1, &r);
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
  std::lock_guard<std::recursive_mutex> lk(mutex_);
  return load_shader_unlocked(name);
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

void Backend::ensure_cmd() {
  // If a command buffer is in-flight (async dispatch submitted but not yet submitted),
  // we must allocate a new one to avoid using a buffer that's still in-flight.
  if (cmd_ != VK_NULL_HANDLE && !cmd_inflight_) {
    std::cerr << "[tvk] Reusing command buffer: " << cmd_ << std::endl;
    return;
  }
  if (cmd_ != VK_NULL_HANDLE) {
    std::cerr << "[tvk] Command buffer in-flight, allocating new one: " << cmd_ << std::endl;
  }
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = cmd_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkResult alloc_result = vkAllocateCommandBuffers(device_, &ai, &cmd_);
  if (alloc_result != VK_SUCCESS) {
    std::string err = "[tvk] vkAllocateCommandBuffers failed: " + std::to_string(alloc_result);
    std::cerr << err << std::endl;
    throw std::runtime_error(err);
  }
  std::cerr << "[tvk] Allocated new command buffer: " << cmd_ << std::endl;
}

Backend::~Backend() {
  if (device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  // Free every still-live Allocation BEFORE destroying the device.
  //
  // ~Allocation -> destroy_allocation() calls vkFreeMemory/vkDestroyBuffer
  // on device_. If an Allocation outlives the Backend (the usual case: the
  // Python objects are garbage-collected after the backend singleton is torn
  // down at interpreter exit), that call runs against a DESTROYED device and
  // the driver reports:
  //     ERROR: vkFreeMemory: Invalid device [VUID-vkFreeMemory-device-parameter]
  // Freeing them here, while the device is still valid, removes that error
  // and the resulting leak.
  for (size_t i = 0; i < live_allocs_.size(); i++) {
    Allocation* alloc = live_allocs_[i];
    if (!alloc) continue;
    // Stop the destructor from calling back into destroy_allocation().
    alloc->backend = nullptr;
    // Release the Vulkan resources while the device is still valid.
    //
    // NOTE: do NOT `delete alloc` here. The Allocation is reference counted and
    // Python may still hold BufferView references to it. Deleting it now would
    // let the last view's release() call `delete this` a second time, which
    // is a double-free (heap corruption). The object is freed by its own
    // refcount when the last view goes away.
    if (alloc->memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, alloc->memory, nullptr);
      alloc->memory = VK_NULL_HANDLE;
    }
    if (alloc->buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, alloc->buffer, nullptr);
      alloc->buffer = VK_NULL_HANDLE;
    }
    if (alloc->size > 0) bytes_ -= alloc->size;
    alloc->size = 0;
  }
  live_allocs_.clear();
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