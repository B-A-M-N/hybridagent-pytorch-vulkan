import torch
import torch_vulkan_ext as tvk
from p2_gpu_kv_test import u32

# Initialize Vulkan first
tvk.init(0)

# Test the _grow scenario
# Initial cap = 257
k_elements = 1 * 8 * 257 * 128  # 263168
total_bytes = k_elements * 4  # 1052672

a = tvk.Allocation(total_bytes, "device")
print(f'Initial Allocation size: {a.size} bytes')

# Test _grow scenario: new_cap = 257 + 1 = 258 (or doubled)
new_cap = 258
k_elements = 1 * 8 * new_cap * 128
total_bytes_new = k_elements * 4
print(f'New capacity: {new_cap}, bytes: {total_bytes_new}')

new_a = tvk.Allocation(total_bytes_new, "device")
print(f'New Allocation size: {new_a.size} bytes')

# Copy old data
old_cap = 257
old_k_elements = 1 * 8 * old_cap * 128
print(f'Old elements: {old_k_elements}')

# Dispatch slice
tvk.dispatch("slice", [a.view(), new_a.view()],
            u32(old_k_elements) + u32(0),
            (old_k_elements + 255) // 256, 1, 1, sync=False)
tvk.submit()

# Now try to use the new allocation with offset
offset = 4096
size = 4096
try:
    v = new_a.view(offset, size)
    print(f'View on new allocation offset={offset}, size={size}: OK')
except Exception as e:
    print(f'View offset={offset}, size={size}: FAILED - {e}')

# Test the exact case from the error
# Step 2: valid_len = 2, k_write_offset = 2 * 8 * 128 = 2048 elements = 8192 bytes
offset = 8192
size = 4096
try:
    v = new_a.view(offset, size)
    print(f'View offset={offset}, size={size}: OK')
except Exception as e:
    print(f'View offset={offset}, size={size}: FAILED - {e}')