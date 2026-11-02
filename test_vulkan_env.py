#!/usr/bin/env python3
"""Test Vulkan with explicit device selection."""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["MESA_VK_DEVICE_SELECT"] = "1002:73ff"
os.environ["RADV_PERFTEST"] = "aco"

import torch_vulkan_ext as tvk
import struct

print("=== Vulkan Device Selection Test ===")

try:
    tvk.init(0)
    print("Init OK")

    buf = tvk.VkBuffer(1024, "device")
    print(f"Buffer OK: {buf.size} bytes")

    push = struct.pack("<I", 1)
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
    tvk.submit()
    print("Dispatch OK")
    print("Vulkan functional")

except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()