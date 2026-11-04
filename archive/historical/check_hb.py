#!/usr/bin/env python3
"""Check HybridBackbone structure."""
import torch
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8

hb = create_hybrid_backbone_with_int8(5)
print(type(hb))
print('Attributes:', [a for a in dir(hb) if not a.startswith('_')])
print('Has lm_head:', hasattr(hb, 'lm_head'))
print('Has backbone:', hasattr(hb, 'backbone'))
print('Has hybridlink:', hasattr(hb, 'hybridlink'))