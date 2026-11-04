#!/usr/bin/env python3
"""Test async dispatch without sync."""
import torch_vulkan_ext as tvk
import struct

tvk.init(0)
print('Init OK')

# Test just dispatch without sync
push = struct.pack('<I', 1)
buf = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1, False)
tvk.submit()
print('Async dispatch OK')