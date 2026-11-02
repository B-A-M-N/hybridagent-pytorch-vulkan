#!/usr/bin/env python3
"""Test Vulkan device after cleanup."""
import torch_vulkan_ext as tvk
import struct
import gc

print("=== Fresh Vulkan Device Test ===")

# Force garbage collection first
gc.collect()

# Try init
print("Test 1: tvk.init(0)")
try:
    tvk.init(0)
    print("  SUCCESS: tvk.init(0) OK")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try to create a buffer
print("Test 2: VkBuffer allocation")
try:
    buf = tvk.VkBuffer(1024, "device")
    print(f"  SUCCESS: VkBuffer OK ({buf.size} bytes)")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try a simple dispatch
print("Test 3: Simple dispatch")
try:
    import struct
    push = struct.pack("<I", 1)
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
    tvk.submit()
    print("  SUCCESS: Dispatch + submit OK")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test multiple allocations
print("Test 4: Multiple allocations")
try:
    for i in range(5):
        buf = tvk.VkBuffer(1024 * 1024, "device")
    print("  SUCCESS: 5x 1MB allocations OK")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== All Vulkan tests passed - Device is functional ===")