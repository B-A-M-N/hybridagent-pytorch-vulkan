#!/usr/bin/env python3
"""Clean-process Vulkan availability probe - no stale state."""
import subprocess
import sys
import os

def probe_vulkan():
    """Probe Vulkan in a completely fresh process."""
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent:/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext",
    })
    
    # Test 1: vulkaninfo
    print("=== Test 1: vulkaninfo --summary ===")
    result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=30)
    print(f"Exit code: {result.returncode}")
    if result.returncode == 0:
        for line in result.stdout.split('\n')[:50]:
            print(f"  {line}")
    else:
        print(f"  stderr: {result.stderr[:500]}")
    
    # Test 2: Minimal Vulkan init via torch_vulkan_ext
    print("\n=== Test 2: torch_vulkan_ext init ===")
    test_code = """
import torch_vulkan_ext as tvk
try:
    tvk.init(0)
    print("SUCCESS: tvk.init(0) OK")
    # Get physical device info
    # Try to create a small buffer
    buf = tvk.VkBuffer(1024, "device")
    print(f"SUCCESS: VkBuffer allocation OK ({buf.size} bytes)")
    # Try a simple dispatch
    tvk.dispatch("matmul_t_bias", [buf.buf, buf.buf, buf.buf], b"\\x01\\x00\\x00\\x00\\x01\\x00\\x00\\x00", 1, 1, 1)
    tvk.submit()
    print("SUCCESS: Dispatch + submit OK")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
import traceback
traceback.print_exc()
"""
    result = subprocess.run([sys.executable, "-c", test_code], 
                           capture_output=True, text=True, timeout=60, env=env)
    print(f"Exit code: {result.returncode}")
    print(f"stdout:\n{result.stdout}")
    if result.stderr:
        print(f"stderr:\n{result.stderr}")

    # Test 3: Check DRM render nodes
    print("\n=== Test 3: DRM render nodes ===")
    try:
        for node in os.listdir("/dev/dri/"):
            print(f"  /dev/dri/{node}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 4: Check kernel logs for AMDGPU
    print("\n=== Test 4: Recent kernel logs ===")
    result = subprocess.run(["dmesg", "-T", "-l", "err,warn"], capture_output=True, text=True, timeout=10)
    for line in result.stdout.split('\n')[-20:]:
        if 'amdgpu' in line.lower() or 'radv' in line.lower() or 'vulkan' in line.lower() or 'gpu' in line.lower():
            print(f"  {line}")

if __name__ == "__main__":
    probe_vulkan()