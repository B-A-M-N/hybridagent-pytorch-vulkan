#!/usr/bin/env python3
"""Test Vulkan command buffer lifecycle."""
import torch_vulkan_ext as tvk
import struct

print("Testing Vulkan command buffer lifecycle...")
tvk.init(0)
print("Init OK")

# Test 1: Basic buffer creation
buf = tvk.VkBuffer(1024, 'device')
print(f"Buffer OK: size={buf.size}")

# Test 2: Sync dispatch
push = struct.pack('<I', 1)
buf = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1, True)
print("Sync dispatch OK")

# Test 2: Another sync dispatch
buf2 = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf2], b'\x01\x00\x00\x00', 1, 1, 1, True)
print("Second sync dispatch OK")

# Test 3: Async dispatch
buf3 = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf3], b'\x01\x00\x00\x00', 1, 1, 1, False)
tvk.submit()
print("Async dispatch OK")

print("All tests passed!")