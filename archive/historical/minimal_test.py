#!/usr/bin/env python3
"""Minimal Vulkan test - just init and buffer."""
import torch_vulkan_ext as tvk

print("Testing minimal Vulkan...")
tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, "device")
print(f"Buffer OK: {buf.size} bytes")

print("Vulkan functional")