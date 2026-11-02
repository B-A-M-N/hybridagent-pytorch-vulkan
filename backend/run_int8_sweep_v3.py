#!/usr/bin/env python3
"""Single-process INT8+Vulkan sweep with layer-by-layer quantization to minimize peak memory."""
import argparse
import json
import os
import sys
import time
import gc

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch
import torch_vulkan_ext as tvk
from backend.memory_budget import check_memory_budget, estimate_config_memory, get_memory_budget
from backend.model_access import get_qwen3_model
from backend.quantized_loader import create_hybrid_backbone_with_int8


def run_config(seq_len: int, gpu_layers: int, reps: int = 2) -> float:
    """Run a single sweep configuration and return latency in ms."""
    import gc
    
    hb = create_hybrid_backbone_with_int8(gpu_layers)
    gc.collect()
    
    ids = torch.randint(0, 1000, (1, seq_len))
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)  # warmup
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                hb(input_ids=ids, use_cache=True)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=12)
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    layer_counts = [int(x) for x in args.layers.split(",")]
    
    budget = get_memory_budget()
    print(f"=== Memory Budget ===")
    print(f"  Total RAM: {budget.total_ram_gb:.1f} GB")
    print(f"  Available RAM: {budget.available_ram_gb:.1f} GB")
    
    seq_len = args.seq
    reps = args.reps
    
    results = []
    print(f"\n=== INT8+Vulkan sweep at S={seq_len} ===", flush=True)
    
    for gpu_layers in [0, 1, 2, 3, 4, 5, 6]:
        est_mem = estimate_config_memory(gpu_layers)
        can_run, status = check_memory_budget(f"S={args.seq} GPU={gpu_layers}", estimate_config_memory(gpu_layers))
        print(f"\n  GPU={gpu_layers}: est {estimate_config_memory(gpu_layers):.1f}GB - {status}")
        
        if not can_run:
            results.append({"gpu_layers": gpu_layers, "ms": None, "error": "INSUFFICIENT_MEMORY", "status": "SKIPPED"})
            continue
        
        print(f"  Running GPU={gpu_layers}...", flush=True)
        try:
            ms = run_config(seq_len, gpu_layers, reps)
            results.append({"gpu_layers": gpu_layers, "ms": ms, "error": None})
            print(f"  GPU={gpu_layers}: {ms:.2f} ms")
        except Exception as e:
            results.append({"gpu_layers": gpu_layers, "ms": None, "error": str(e)})
            print(f"  FAILED: {e}")
        
        gc.collect()
        time.sleep(2)
    
    # Summary
    print(f"\n=== SWEEP SUMMARY S={args.seq} ===")
    ok = [r for r in results if r["ms"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["ms"])
        cpu = next((r["ms"] for r in results if r["gpu_layers"] == 0), None)
        print(f"  best: {best['gpu_layers']} GPU layers -> {best['ms']:.2f} ms")
        if cpu:
            speedup = cpu / best['ms']
            print(f"  vs all-CPU {cpu:.2f} ms -> {speedup:.3f}x {'GPU WINS' if best['ms'] < cpu else 'CPU WINS'}")
    
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"seq_len": args.seq, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    import json
    import gc
    import time
    import torch
    import torch_vulkan_ext as tvk
    
    tvk.init(0)
    main()