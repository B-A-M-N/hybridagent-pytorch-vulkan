#!/usr/bin/env python3
"""Check if the source fix is loaded."""
import backend.hybrid_offload as hf
import inspect

src = inspect.getsource(hf.VulkanDecoderLayer.forward)

if "cached_k = past_key_values.key_cache[self.layer_idx]" in src:
    print("FIX IS LOADED")
else:
    print("FIX NOT LOADED - using cached version")
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "past_key_values is None" in line or "cached_k" in line:
            print(f"  {line.strip()}")