#!/usr/bin/env python3
"""Test different Vulkan device indices."""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["MESA_VK_DEVICE_SELECT"] = "1002:73ff"
os.environ["RADV_PERFTEST"] = "aco"

import torch_vulkan_ext as tvk
import struct

print("=== Testing Different Device Indices ===")

for device_idx in range(3):
    print(f"\n=== Testing device index {device_idx} ===")
    try:
        tvk.init(device_idx)
        print(f"  Init({device_idx}) OK")

        buf = tvk.VkBuffer(1024, "device")
        print(f"  Buffer OK: {buf.size} bytes")

        push = struct.pack("<I", 1)
        tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
        tvk.submit()
        print("  Dispatch OK")
        print(f"  Device {device_idx} FUNCTIONAL")
        break
    except Exception as e:
        print(f"  Device {device_idx} FAIL: {e}")

print("\n=== Test complete ===")