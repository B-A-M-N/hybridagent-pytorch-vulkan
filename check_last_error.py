#!/usr/bin/env python3
"""Check tvk.last_error() after init."""
import torch_vulkan_ext as tvk

print("Testing tvk.init(0) with last_error capture...")
tvk.init(0)
print("Init OK")

# Check last error
last_err = tvk.last_error()
print(f"Last error: '{last_err}'")