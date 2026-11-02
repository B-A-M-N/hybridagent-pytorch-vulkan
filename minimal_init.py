#!/usr/bin/env python3
"""Minimal Vulkan test - just init, no dispatch."""
import torch_vulkan_ext as tvk

print("=== Minimal Vulkan Test ===")

tvk.init(0)
print("Init OK")

print("Vulkan init successful")