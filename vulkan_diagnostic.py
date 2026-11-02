#!/usr/bin/env python3
"""Independent Vulkan diagnostic - capture every VkResult."""
import os
import sys
import struct

print("=== Independent Vulkan Diagnostic ===")

# Clean environment
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.json"
os.environ["MESA_VK_DEVICE_SELECT"] = "1002:73ff"
os.environ["RADV_PERFTEST"] = "aco"

# Step 1: Check vulkaninfo
print("1. vulkaninfo --summary")
import subprocess
result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
print(f"  Exit code: {result.returncode}")
if result.returncode == 0:
    for line in result.stdout.split('\n')[:40]:
        print(f"  {line}")
else:
    print(f"  stderr: {result.stderr[:500]}")

# Step 2: Check DRM nodes
print("\n2. DRM render nodes")
try:
    for node in os.listdir("/dev/dri/"):
        print(f"  /dev/dri/{node}")
except Exception as e:
    print(f"  Error: {e}")

# Step 3: Vulkan via torch_vulkan_ext with detailed error capture
print("\n3. torch_vulkan_ext initialization with error capture")
import torch_vulkan_ext as tvk

print("  Step 3a: tvk.init(0)")
try:
    tvk.init(0)
    print("  tvk.init(0) SUCCESS")
except Exception as e:
    print(f"  tvk.init(0) FAILED: {type(e).__name__}: {e}")
    # Get last error
    last_err = tvk.last_error()
    print(f"  Last error: '{last_err}'")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  Step 3b: VkBuffer allocation")
try:
    buf = tvk.VkBuffer(1024, "device")
    print(f"  VkBuffer OK: {buf.size} bytes")
except Exception as e:
    print(f"  VkBuffer FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 3c: Dispatch + submit")
try:
    push = struct.pack("<I", 1)
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
    tvk.submit()
    print("  Dispatch+submit OK")
except Exception as e:
    print(f"  Dispatch FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 3d: Multiple dispatches")
try:
    for i in range(5):
        buf = tvk.VkBuffer(1024, "device")
        tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
        tvk.submit()
    print("  Multiple dispatches OK")
except Exception as e:
    print(f"  Multiple dispatches FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL TESTS PASSED - Vulkan functional ===")