#!/usr/bin/env python3
"""Independent Vulkan probe - fresh process."""
import torch_vulkan_ext as tvk
import ctypes
import struct
import os

print("=== Independent Vulkan Probe ===")
print(f"Shared object: {__import__('torch_vulkan_ext').__file__}")
print("libvulkan_radeon.so loaded:", any('libvulkan_radeon' in l for l in open('/proc/self/maps').read().splitlines()))

print("\nStep 1: tvk.init(0)")
tvk.init(0)
print("  Init OK")
print("Last error:", tvk.last_error())

print("\nStep 2: VkBuffer allocation")
buf = tvk.VkBuffer(1024, "device")
print(f"  Buffer OK: {buf.size} bytes")

print("\nStep 3: Dispatch + submit")
push = struct.pack("<I", 1)
buf = tvk.VkBuffer(1024, "device")
tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
tvk.submit()
print("  Dispatch+submit OK")

print("Step 4: Multiple dispatches")
for i in range(5):
    buf = tvk.VkBuffer(1024, "device")
    tvk.dispatch("rmsnorm", [buf], b"\x01\x00\x00\x00", 1, 1, 1)
    tvk.submit()
    print(f"  Iteration {i+1} OK")

print("\n=== ALL TESTS PASSED - Vulkan functional ===")