#!/usr/bin/env python3
"""Test Vulkan device loss recovery."""
import torch_vulkan_ext as tvk
import struct

tvk.init(0)
print("Init OK")

# Test 1: First dispatch (sync)
buf = tvk.VkBuffer(1024, 'device')
push = struct.pack('<I', 1)
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1, True)
print("First sync dispatch OK")

# Test 2: Second dispatch (sync) - this should trigger device loss detection
buf2 = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf2], b'\x01\x00\x00\x00', 1, 1, 1, True)
print("Second sync dispatch OK")

# Test 3: Third dispatch
buf3 = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf3], b'\x01\x00\x00\x00', 1, 1, 1, True)
print("Third sync dispatch OK")

print("All tests passed!")