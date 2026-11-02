#!/usr/bin/env python3
"""Test direct INT8 loader."""
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
import torch

hb = create_hybrid_backbone_with_int8(0)
print('SUCCESS: CPU INT8 model created')

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    out = hb(input_ids=ids, use_cache=True)
print(f'Forward pass OK: logits shape {out.logits.shape}')