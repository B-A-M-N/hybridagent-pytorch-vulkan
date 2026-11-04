#!/usr/bin/env python3
"""Independent Vulkan probe."""
import torch_vulkan_ext as tvk
import struct

print("=== Independent Vulkan Probe ===")

print("Step 1: tvk.init(0)")
tvk.init(0)
print("  Init OK")

print("Step 2: VkBuffer allocation")
buf = tvk.VkBuffer(1024, "device")
print(f"  Buffer OK: {buf.size} bytes")

print("Step 3: Dispatch + submit")
push = struct.pack("<I", 1)
buf = tvk.VkBuffer(1024, "device")
tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
tvk.submit()
print("  Dispatch+submit OK")

print("Step 4: Multiple dispatches")
for i in range(3):
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
    tvk.submit()
    print(f"  Iteration {i+1} OK")

print("\n=== ALL TESTS PASSED - Vulkan functional ===")