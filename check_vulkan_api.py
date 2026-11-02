#!/usr/bin/env python3
"""Check Vulkan buffer API."""
import torch_vulkan_ext as tvk
import inspect

tvk.init(0)
buf = tvk.VkBuffer(1024, "device")
print("Buffer type:", type(buf))
print("Buffer dir:", [x for x in dir(buf) if not x.startswith('_')])
print("Buffer size:", buf.size)

# Check dispatch signature
print("dispatch signature:", inspect.signature(tvk.dispatch))