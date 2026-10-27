"""
Whole-model sweep to locate the optimal GPU layer count per sequence length.

This is the ONLY trustworthy way to choose a partition, because the cost of a
CPU<->GPU boundary is not visible when a layer is measured in isolation.

Method: time the full 36-layer model at several GPU-layer counts, then
report the measured optimum. The VRAM ceiling is computed independently from
the weight and activation sizes.

Each measurement runs in a fresh process so that a VRAM-pressure failure at
one partition point cannot contaminate the next.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
VRAM = 8.57e9
TOTAL_LAYERS = 36


def run_one(seq_len: int, gpu_layers: int, reps: int = 2):
    """Measure the full model with `gpu_layers` on the GPU. Returns seconds."""
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    tvk.init(0)
    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = torch.randint(0, 1000, (1, seq_len))
    if gpu_layers == 0:
        m = lm
    else:
        m = HybridBackbone(qm, Placement.first_n(gpu_layers, TOTAL_LAYERS))

    with torch.no_grad():
        m(input_ids=ids, use_cache=True)          # warmup
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                m(input_ids=ids, use_cache=True)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.driver:
        secs = run_one(args.seq, args.layers, args.reps)
        print(f"RESULT {secs}")
        return

    # driver: one subprocess per partition point
    seq = args.seq
    counts = [0, 1, 2, 3, 4, 5, 6]
    results = []
    print(f"=== whole-model sweep at S={seq} ===", flush=True)
    for g in counts:
        p = subprocess.run(
            [sys.executable, __file__, "--seq", str(seq), "--layers", str(g),
             "--reps", str(args.reps)],
            capture_output=True, text=True, timeout=1800,
            env={**os.environ,
                 "PYTHONPATH": f"{ROOT}:{ROOT}/third_party/HybridAgent:"
                               f"{ROOT}/backend/torch_vulkan_ext",
                 "OMP_NUM_THREADS": "12"})
        secs = None
        err = None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                secs = float(line.split()[1])
        if secs is None:
            err = (p.stderr or p.stdout)[-300:]
            print(f"  gpu_layers={g}: FAILED ({err.strip()[:120]})", flush=True)
        else:
            print(f"  gpu_layers={g}: {secs:9.3f} s", flush=True)
        results.append({"gpu_layers": g, "seconds": secs, "error": err})

    ok = [r for r in results if r["seconds"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["seconds"])
        cpu = next((r["seconds"] for r in results if r["gpu_layers"] == 0), None)
        print(f"\n  best: {best['gpu_layers']} GPU layers -> {best['seconds']:.3f} s")
        if cpu:
            print(f"  vs all-CPU {cpu:.3f} s -> "
                  f"{cpu/best['seconds']:.3f}x "
                  f"({'GPU WINS' if best['seconds'] < cpu else 'CPU WINS'})")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"seq_len": seq, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
