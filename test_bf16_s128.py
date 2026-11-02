#!/usr/bin/env python3
"""Run BF16 baseline for S=128."""
import torch
import torch_vulkan_ext as tvk
from hybrid_agent import load_hybrid_agent
from backend.hybrid_offload import HybridBackbone, Placement
import time
import gc

tvk.init(0)
torch.manual_seed(0)

# Load original BF16 model
print('Loading original BF16 model...')
model, tok = load_hybrid_agent(
    '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
    dtype=torch.bfloat16, device='cpu'
)
print('Model loaded')

# Create HybridBackbone with BF16
qm = model.backbone.model
hb = HybridBackbone(qm, Placement.first_n(0, 36))
print('HybridBackbone created with 0 GPU layers')

del model
gc.collect()

# Benchmark
ids = torch.randint(0, 1000, (1, 128))
with torch.no_grad():
    hb(input_ids=ids, use_cache=True)  # warmup
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        with torch.no_grad():
            hb(input_ids=ids, use_cache=True)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    result = ts[len(ts) // 2] * 1000
    print(f"RESULT {result:.2f}")