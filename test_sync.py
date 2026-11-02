#!/usr/bin/env python3
"""Minimal test with sync dispatch."""
import torch_vulkan_ext as tvk
import struct

print("Testing sync dispatch...")
tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, 'device')
print('Buffer OK')

push = struct.pack('<I', 1)
buf = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1, True)
print('Sync dispatch OK')

print('Vulkan functional')