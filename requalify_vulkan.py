#!/usr/bin/env python3
"""Clean-process Vulkan device requalification probe."""
import subprocess
import sys
import os

def probe_vulkan_clean():
    """Probe Vulkan in a completely fresh process with clean environment."""
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext",
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
    })
    
    print("=== VULKAN CLEAN-PROCESS REQUALIFICATION ===\n")
    
    # Test 1: vulkaninfo
    print("1. vulkaninfo --summary")
    result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
    print(f"   Exit code: {result.returncode}")
    if result.returncode == 0:
        for line in result.stdout.split('\n')[:60]:
            print(f"   {line}")
    else:
        print(f"   stderr: {result.stderr[:500]}")
    
    # Test 2: DRM render nodes
    print("\n2. DRM render nodes")
    try:
        for node in os.listdir("/dev/dri/"):
            print(f"   /dev/dri/{node}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Clean process Vulkan init
    print("\n3. Clean-process Vulkan init test")
    test_code = '''
import torch_vulkan_ext as tvk
import struct

tvk.init(0)
print("Init OK")

buf = tvk.VkBuffer(1024, "device")
print(f"Buffer OK: {buf.size} bytes")

push = struct.pack("<I", 1)
tvk.dispatch("rmsnorm", [buf], b"\\x01\\x00\\x00\\x00", 1, 1, 1)
tvk.submit()
print("Dispatch OK")

print("Vulkan functional")
'''
    
    result = subprocess.run([
        sys.executable, "-c", test_code
    ], capture_output=True, text=True, timeout=30, env=env)
    
    print(f"   Exit code: {result.returncode}")
    print(f"   stdout:\n{result.stdout}")
    if result.stderr:
        print(f"   stderr:\n{result.stderr}")

if __name__ == "__main__":
    probe_vulkan_clean()