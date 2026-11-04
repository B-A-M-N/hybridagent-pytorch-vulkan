#!/usr/bin/env python3
"""Run BF16 baseline for S=1024."""
import torch
import time
import gc
import torch_vulkan_ext as tvk
from hybrid_agent import load_hybrid_agent
from backend.hybrid_offload import HybridBackbone, Placement

tvk.init(0)
torch.manual_seed(0)

model, tok = load_hybrid_agent(
    '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
    dtype=torch.bfloat16, device='cpu'
)
qm = model.backbone.model
hb = HybridBackbone(qm, Placement.first_n(0, 36))
del model
gc.collect()

ids = torch.randint(0, 1000, (1, 1024))
with torch.no_grad():
    hb(input_ids=ids, use_cache=True)
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        with torch.no_grad():
            hb(input_ids=ids, use_cache=True)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"RESULT {ts[1]*1000:.2f}")