#!/usr/bin/env python3
"""Test with Vulkan validation layers enabled."""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["VK_LAYER_PATH"] = "/usr/share/vulkan/explicit_layer.d"
os.environ["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
os.environ["VK_LOADER_LAYERS_ENABLE"] = "VK_LAYER_KHRONOS_validation"

import torch_vulkan_ext as tvk
import struct

print("=== Vulkan with Validation Layers ===")

tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, "device")
print(f"Buffer OK: {buf.size} bytes")

push = struct.pack("<I", 1)
tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
tvk.submit()
print("Dispatch OK")

print("Vulkan functional")