#!/usr/bin/env python3
"""Simple Vulkan device test."""
import torch_vulkan_ext as tvk
import struct

print("Test 1: tvk.init(0)")
tvk.init(0)
print("  OK")

print("Test 2: VkBuffer allocation")
buf = tvk.VkBuffer(1024, "device")
print(f"  Buffer size: {buf.size} bytes")

print("Test 3: Simple dispatch")
push = struct.pack("<I", 1)
tvk.dispatch("rmsnorm", [buf.buf], b"\x01\x00\x00\x00", 1, 1, 1)
tvk.submit()
print("  Dispatch OK")

print("Test 3b: Multiple allocations")
for i in range(5):
    buf = tvk.VkBuffer(1024 * 1024, "device")
print("  Multiple allocations OK")

print("\nAll tests passed - Vulkan is functional")