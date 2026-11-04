"""Measure per-config throughput without the slow all-Vulkan path."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import HybridBackbone, Placement  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def run(bb, ids):
    t0 = time.time()
    with torch.no_grad():
        out = bb(input_ids=ids, use_cache=True, output_hidden_states=True)
    return out, time.time() - t0


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    print(f"tokens: {ids.shape[1]}, layers: {len(qm.layers)}\n")

    configs = [
        ("all-CPU", Placement.all_cpu()),
        ("8 layers Vulkan", Placement.first_n(8, 36)),
        ("4 layers Vulkan", Placement.first_n(4, 36)),
        ("1 layer Vulkan", Placement.first_n(1, 36)),
    ]

    results = {}
    for name, pl in configs:
        bb = HybridBackbone(qm, pl)
        # warmup
        run(bb, ids)
        ts = []
        for _ in range(3):
            out, dt = run(bb, ids)
            ts.append(dt)
        ts.sort()
        results[name] = (ts[1], out)
        print(f"{name:18s} median {ts[1]:6.2f}s  (cache {out.past_key_values.get_seq_length()})",
              flush=True)

    # compare heterogeneous vs CPU
    _, out_cpu = results["all-CPU"]
    for name in ("1 layer Vulkan", "4 layers Vulkan", "8 layers Vulkan"):
        _, out = results[name]
        err = float((out.last_hidden_state.float()
                     - out_cpu.last_hidden_state.float()).abs().max())
        print(f"{name:18s} vs CPU: max_abs {err:.3e}")


if __name__ == "__main__":
    main()
