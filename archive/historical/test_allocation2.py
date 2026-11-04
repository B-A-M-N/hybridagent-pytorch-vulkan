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

# Test view with offset and size
v2 = a.view(1024, 2048)
print('BufferView2 size:', v2.size)
print('BufferView2 _alloc:', v2._alloc)
print('BufferView2 _h:', v2._h)

# Test view with different offsets
for offset in [0, 512, 1024, 2048, 3072]:
    try:
        v_test = a.view(offset, 512)
        print(f'View offset={offset}, size=512: OK, _h={v_test._h}')
    except Exception as e:
        print(f'View offset={offset}, size=512: FAILED - {e}')

# Test various sizes
for size in [256, 512, 1024, 2048, 4096]:
    try:
        v_test = a.view(0, size)
        print(f'View offset=0, size={size}: OK, _h={v_test._h}')
    except Exception as e:
        print(f'View offset=0, size={size}: FAILED - {e}')