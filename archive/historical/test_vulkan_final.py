#!/usr/bin/env python3
"""Test Vulkan dispatch with correct API."""
import torch_vulkan_ext as tvk
import struct

tvk.init(0)
buf = tvk.VkBuffer(1024, 'device')
print("Buffer OK")

push = struct.pack('<I', 1)  # M=1
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1)
tvk.submit()
print('Dispatch OK')
print('Vulkan is functional')