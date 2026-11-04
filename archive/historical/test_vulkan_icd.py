#!/usr/bin/env python3
"""Test Vulkan with explicit ICD selection."""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
os.environ["MESA_VK_DEVICE_SELECT"] = "1002:73ff"
os.environ["RADV_PERFTEST"] = "aco"

import torch_vulkan_ext as tvk

print("=== Test with explicit ICD selection ===")
try:
    tvk.init(0)
    print("Init OK")
except Exception as e:
    print(f"Init FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Vulkan init successful")