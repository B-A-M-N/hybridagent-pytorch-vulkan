#!/usr/bin/env python3
"""Test INT8 model with 1 GPU layer."""
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
import torch
import gc

print("=== Testing INT8 model with 1 GPU layer ===")

hb = create_hybrid_backbone_with_int8(1)
print('SUCCESS: INT8 model with 1 GPU layer created')

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    hb(input_ids=ids, use_cache=True)  # warmup
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if t0:
        t0.record()
    out = hb(input_ids=ids, use_cache=True)
    if t1:
        t1.record()
        torch.cuda.synchronize()
        print(f"Forward pass: {t0.elapsed_time(t1):.2f} ms")
    else:
        import time
        t0 = time.perf_counter()
        out = hb(input_ids=ids, use_cache=True)
        t1 = time.perf_counter()
        print(f"Forward pass: {(t1-t0)*1000:.2f} ms")

logits = hb.backbone.lm_head(out.last_hidden_state) if hasattr(hb, 'backbone') else out
print(f'Forward OK: logits shape {logits.shape}')