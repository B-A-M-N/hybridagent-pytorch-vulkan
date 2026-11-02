#!/usr/bin/env python3
"""Post-reboot Vulkan health check - fresh process probe."""
import os
import sys
import struct
import subprocess

print("=== POST-REBOOT VULKAN HEALTH CHECK ===")

# Clean environment
env = os.environ.copy()
env.update({
    "OMP_NUM_THREADS": "1",
    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json",
    "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.json",
    "MESA_VK_DEVICE_SELECT": "1002:73ff",
    "RADV_PERFTEST": "aco",
})

# Step 1: vulkaninfo
print("1. vulkaninfo --summary")
result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
print(f"  Exit code: {result.returncode}")
if result.returncode == 0:
    for line in result.stdout.split('\n')[:30]:
        print(f"  {line}")
else:
    print(f"  stderr: {result.stderr[:500]}")

# Step 2: DRM render nodes
print("\n2. DRM render nodes")
try:
    for node in os.listdir("/dev/dri/"):
        print(f"  /dev/dri/{node}")
except Exception as e:
    print(f"  Error: {e}")

# Step 3: Check DRM processes
print("\n3. Current GPU processes")
try:
    for node in os.listdir("/dev/dri/"):
        if node.startswith("renderD") or node.startswith("card"):
            try:
                result = subprocess.run(["lsof", f"/dev/dri/{node}"], capture_output=True, text=True, timeout=5)
                if result.stdout.strip():
                    print(f"  /dev/dri/{node}:")
                    for line in result.stdout.strip().split('\n')[:5]:
                        print(f"    {line}")
            except:
                pass
except Exception as e:
    print(f"  Error: {e}")

# Step 4: Kernel logs
print("\n4. Recent AMDGPU kernel logs")
result = subprocess.run(["journalctl", "-k", "-o", "short", "-n", "50"], capture_output=True, text=True, timeout=10)
for line in result.stdout.split('\n')[-20:]:
    if any(kw in line.lower() for kw in ['amdgpu', 'radv', 'vulkan', 'gpu', 'drm', 'vk']):
        print(f"  {line}")

# Step 5: Minimal Vulkan test
print("\n5. Minimal Vulkan test in clean process")
test_code = '''
import os
import torch_vulkan_ext as tvk
import struct

print("  tvk.init(0)")
tvk.init(0)
print("  Init OK")

buf = tvk.VkBuffer(1024, "device")
print(f"  Buffer OK: {buf.size} bytes")

push = struct.pack("<I", 1)
tvk.dispatch("rmsnorm", [buf], b"\\x01\\x00\\x00\\x00", 1, 1, 1)
tvk.submit()
print("  Dispatch+submit OK")

print("VULKAN FUNCTIONAL")
'''

with open('/tmp/vulkan_probe.py', 'w') as f:
    f.write(test_code)

result = subprocess.run(
    ["/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/.venv/bin/python", "/tmp/vulkan_probe.py"],
    capture_output=True, text=True, timeout=30, env=env
)

print(f"  Exit code: {result.returncode}")
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

# Step 6: Repeated init/cleanup test
print("\n6. Repeated init/cleanup test")
test_code2 = '''
import torch_vulkan_ext as tvk
import struct

for i in range(3):
    print(f"  Iteration {i+1}: tvk.init(0)")
    tvk.init(0)
    buf = tvk.VkBuffer(1024, "device")
    push = struct.pack("<I", 1)
    tvk.dispatch("rmsnorm", [buf], b"\\x01\\x00\\x00\\x00", 1, 1, 1)
    tvk.submit()
    print(f"  Iteration {i+1} OK")
print("REPEATED INIT TEST PASSED")
'''

with open('/tmp/vulkan_repeated.py', 'w') as f:
    f.write(test_code2)

result = subprocess.run(
    ["/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/.venv/bin/python", "/tmp/vulkan_repeated.py"],
    capture_output=True, text=True, timeout=30, env=env
)

print(f"  Exit code: {result.returncode}")
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

print("\n=== POST-REBOOT VULKAN HEALTH CHECK COMPLETE ===")