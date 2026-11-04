#!/usr/bin/env python3
"""Test command buffer lifecycle."""
import torch_vulkan_ext as tvk
import struct

print("=== Command Buffer Lifecycle Test ===")

tvk.init(0)
print("Init OK")

# Test 1: First dispatch (sync)
print("\n1. First sync dispatch")
buf = tvk.VkBuffer(1024, "device")
push = struct.pack("<I", 1)
try:
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1, True)
    print("  Sync dispatch OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Second sync dispatch (reusing cmd_)
print("\n2. Second sync dispatch (reusing cmd_)")
buf2 = tvk.VkBuffer(1024, "device")
try:
    tvk.dispatch("rmsnorm", [buf2], b"\x01\x00\x00\x00", 1, 1, 1, True)
    print("  Second sync dispatch OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Async then submit
print("\n3. Async then submit")
buf3 = tvk.VkBuffer(1024, "device")
try:
    tvk.dispatch("rmsnorm", [buf3], b"\x01\x00\x00\x00", 1, 1, 1, False)
    tvk.submit()
    print("  Async then submit OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== All tests passed ===")