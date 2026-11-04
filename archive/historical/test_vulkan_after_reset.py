#!/usr/bin/env python3
"""Test Vulkan after GPU reset."""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"

import torch_vulkan_ext as tvk

print("Testing Vulkan after GPU reset...")
try:
    tvk.init(0)
    print("OK")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)