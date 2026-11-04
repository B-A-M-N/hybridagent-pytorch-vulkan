#!/usr/bin/env python3
"""Test Vulkan basic functionality."""
import torch_vulkan_ext as tvk

print("Testing Vulkan basic functionality...")
tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, 'device')
print(f"Buffer OK: size={buf.size}")

print("Vulkan functional")