#!/usr/bin/env python3
"""Independent Vulkan initialization probe - fresh process, capture every VkResult."""
import os
import sys
import subprocess

# Clean environment
env = os.environ.copy()
env.update({
    "OMP_NUM_THREADS": "1",
    "PYTHONPATH": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext",
    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    "MESA_VK_DEVICE_SELECT": "1002:73ff",
    "RADV_PERFTEST": "aco",
})

test_code = '''
import torch_vulkan_ext as tvk
import struct

print("=== Independent Vulkan Probe ===")
print("Step 1: tvk.init(0)")
try:
    tvk.init(0)
    print("  tvk.init(0) SUCCESS")
except Exception as e:
    print(f"  tvk.init(0) FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("Step 2: VkBuffer allocation")
try:
    buf = tvk.VkBuffer(1024, "device")
    print(f"  VkBuffer OK: {buf.size} bytes")
except Exception as e:
    print(f"  VkBuffer FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("Step 3: Dispatch + submit")
try:
    push = struct.pack("<I", 1)
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\\x01\\x00\\x00\\x00", 1, 1, 1)
    tvk.submit()
    print("  Dispatch+submit OK")
except Exception as e:
    print(f"  Dispatch FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("Step 4: Multiple dispatches")
try:
    for i in range(5):
        buf = tvk.VkBuffer(1024, "device")
        tvk.dispatch("rmsnorm", [buf], b"\\x01\\x00\\x00\\x00", 1, 1, 1)
        tvk.submit()
    print("  Multiple dispatches OK")
except Exception as e:
    print(f"  Multiple dispatches FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\\n=== ALL TESTS PASSED - Vulkan functional ===")
'''

# Write test to file
with open('/tmp/vulkan_probe.py', 'w') as f:
    f.write(test_code)

env = os.environ.copy()
env.update({
    "OMP_NUM_THREADS": "1",
    "PYTHONPATH": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext",
    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    "MESA_VK_DEVICE_SELECT": "1002:73ff",
    "RADV_PERFTEST": "aco",
})

result = subprocess.run(
    ["/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/.venv/bin/python", "/tmp/vulkan_probe.py"],
    capture_output=True, text=True, timeout=30, env=env
)

print(f"Exit code: {result.returncode}")
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)