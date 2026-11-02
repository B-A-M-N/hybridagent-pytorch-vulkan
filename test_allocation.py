import torch
import torch_vulkan_ext as tvk

# Initialize Vulkan first
tvk.init(0)

# Test Allocation and BufferView
a = tvk.Allocation(4096, 'device')
print('Allocation type:', type(a))
print('Allocation size:', a.size)

v = a.view(0, 4096)
print('BufferView type:', type(v))
print('BufferView size:', v.size)
print('BufferView _alloc:', v._alloc)
print('BufferView _h:', v._h)