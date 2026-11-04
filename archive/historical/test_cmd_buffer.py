#!/usr/bin/env python3
"""Detailed Vulkan command buffer lifecycle test."""
import torch_vulkan_ext as tvk
import struct

print("=== Detailed Vulkan Command Buffer Test ===")

tvk.init(0)
print("Init OK")

# Test 1: Buffer creation
buf = tvk.VkBuffer(1024, "device")
print(f"Buffer OK: {buf.size} bytes")

# Test 2: Check dispatch without sync
print("\nTest 2: Async dispatch")
try:
    push = struct.pack("<I", 1)
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1, False)  # async
    print("  Async dispatch OK")
    tvk.submit()
    print("  Submit OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check dispatch with sync
print("\nTest 3: Sync dispatch")
try:
    push = struct.pack("<I", 1)
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1, True)  # sync
    print("  Sync dispatch OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test complete ===")