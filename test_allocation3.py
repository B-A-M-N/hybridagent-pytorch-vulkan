import torch
import torch_vulkan_ext as tvk

# Initialize Vulkan first
tvk.init(0)

# Reproduce the exact case from the GPU KV cache
# num_kv_heads = 8, head_dim = 128, KVD = 1024
# cap = 257, so total elements = 1 * 8 * 257 * 128 = 263168
# total bytes = 263168 * 4 = 1052672

k_elements = 1 * 8 * 257 * 128  # 263168
total_bytes = k_elements * 4  # 1052672

a = tvk.Allocation(total_bytes, "device")
print(f'Allocation size: {a.size} bytes')

# Test the exact case that fails
# Step 1: valid_len = 1
# k_write_offset = 1 * 8 * 128 = 1024 elements
# k_write_offset * 4 = 4096 bytes
# k_elements = S * KVD = 1 * 1024 = 1024 elements = 4096 bytes

offset = 4096  # bytes
size = 4096    # bytes

try:
    v = a.view(offset, size)
    print(f'View offset={offset}, size={size}: OK, _h={v._h}')
except Exception as e:
    print(f'View offset={offset}, size={size}: FAILED - {e}')

# Test various valid cases
print("\n--- Testing various offsets and sizes ---")
test_cases = [
    (0, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (102400, 4096),
    (100000, 4096),
    (0, total_bytes),
    (0, 4096),
    (4096, 4096),
]

for offset, size in test_cases:
    try:
        v = a.view(offset, size)
        print(f'View offset={offset}, size={size}: OK')
    except Exception as e:
        print(f'View offset={offset}, size={size}: FAILED - {e}')

# Also test the exact case from step 2
print("\n--- Step 2 case ---")
# Step 2: valid_len = 2
# k_write_offset = 2 * 8 * 128 = 2048 elements = 8192 bytes
offset = 8192
size = 4096
try:
    v = a.view(offset, size)
    print(f'View offset={offset}, size={size}: OK')
except Exception as e:
    print(f'View offset={offset}, size={size}: FAILED - {e}')