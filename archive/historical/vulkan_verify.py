#!/usr/bin/env python3
"""Verify loaded Vulkan extension binary."""
import torch_vulkan_ext as tvk
import os

print("Shared object path:", tvk.__file__)
print("File exists:", os.path.exists(tvk.__file__))

import stat
st = os.stat(tvk.__file__)
print("File size:", st.st_size)
import time
print("Modified:", time.ctime(st.st_mtime))

import torch_vulkan_ext as tvk
tvk.init(0)
print("Init OK")
print("Last error:", tvk.last_error())

buf = tvk.VkBuffer(1024, 'device')
print("Buffer OK")

import struct
push = struct.pack("<I", 1)
buf = tvk.VkBuffer(1024, 'device')
tvk.dispatch('rmsnorm', [buf], b'\x01\x00\x00\x00', 1, 1, 1)
print("Dispatch OK")