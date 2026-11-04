#!/usr/bin/env python3
"""Test dispatch with proper API."""
import torch_vulkan_ext as tvk
import struct

print("Testing dispatch...")
tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, "device")
print(f"Buffer OK: {buf.size} bytes")

# Test dispatch
push = struct.pack("<I", 1)
tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
tvk.submit()
print("Dispatch OK")

print("Vulkan functional")