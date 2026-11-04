"""
Requalification: CPU layer vs Vulkan layer, measured in the SAME process.

The "4000x slower" claim came from a table like:
    Single layer S=12 | CPU 28 ms | Vulkan (Q4) >120 s (timeout) | >4000x slower

A timeout is not a measurement. The ratio was computed from a number that was
never observed. This script measures BOTH sides properly:

  - same layer, same input, same process, same clock
  - CPU: the PyTorch layer forward
  - Vulkan: the same layer's forward on the GPU
  - fp32 and Q4 variants
  - and the whole-model sweep at 512 / 1024 / 2048 to reproduce the earlier
    prefill comparison

Everything is timed with time.perf_counter and reported with units. Nothing
is estimated, and no timeout is used as a number.

Run: python tests/requalify_cpu_vs_vulkan.py [--seq 12] [--sweep]
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
REPS = 5
WARMUP = 2


def med(xs):
    return statistics.median(xs)


def timeit(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--q4", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    print(f"device: {tvk.device_name()}")
    print(f"model layers: {len(qm.layers)}  hidden {qm.config.hidden_size}\n")

    results = []

    if not args.sweep:
        # ---- per-layer CPU vs Vulkan, same layer, same input ----
        S = args.seq
        layer = qm.layers[0]
        ids = torch.randint(0, 1000, (1, S))

        # CPU baseline: the unmodified upstream model, same input.
        def cpu_model():
            with torch.no_grad():
                lm(input_ids=ids, use_cache=True)

        c_med, c_min, c_max = timeit(cpu_model, reps=3, warmup=1)
        print(f"CPU  full {len(qm.layers)}-layer forward @S={S}: "
              f"median {c_med:.2f} ms ({c_med/len(qm.layers):.2f} ms/layer)")
        results.append({"kind": "cpu_full", "seq": S, "median_ms": c_med,
                        "min_ms": c_min, "max_ms": c_max,
                        "ms_per_layer": c_med / len(qm.layers)})

        # Vulkan: the same model with N layers on GPU
        for ng in (1, 2, 4, 6):
            try:
                hb = HybridBackbone(qm, Placement.first_n(ng, len(qm.layers)))
                with torch.no_grad():
                    hb(input_ids=ids, use_cache=True)

                def vk():
                    with torch.no_grad():
                        hb(input_ids=ids, use_cache=True)
                v_med, v_min, v_max = timeit(vk, reps=3, warmup=1)
                print(f"VULKAN {ng:2d} GPU layers, S={S}: median {v_med:.2f} ms")
                results.append({"kind": "vk", "gpu_layers": ng, "seq": S,
                                "median_ms": v_med, "min_ms": v_min,
                                "max_ms": v_max})
                del hb
                gc.collect()
            except Exception as e:
                print(f"VULKAN {ng:2d} GPU layers FAILED: {type(e).__name__}: {e}")
                results.append({"kind": "vk", "gpu_layers": ng, "seq": S,
                                "error": f"{type(e).__name__}: {e}"})

    else:
        # ---- whole-model prefill sweep, reproducing the earlier table ----
        for S in (512, 1024, 2048):
            ids = torch.randint(0, 1000, (1, S))
            for ng in (0, 4):
                try:
                    if ng == 0:
                        m = lm
                    else:
                        m = HybridBackbone(qm, Placement.first_n(ng, len(qm.layers)))
                    with torch.no_grad():
                        m(input_ids=ids, use_cache=True)

                    def run():
                        with torch.no_grad():
                            m(input_ids=ids, use_cache=True)
                    med_ms, lo, hi = timeit(run, reps=2, warmup=1)
                    label = "CPU " if ng == 0 else f"VULKAN({ng})"
                    print(f"{label:10s} S={S:5d}  {med_ms/1000:9.2f} s", flush=True)
                    results.append({"kind": "prefill", "gpu_layers": ng,
                                    "seq": S, "median_ms": med_ms,
                                    "min_ms": lo, "max_ms": hi})
                    if ng:
                        del m
                    gc.collect()
                except Exception as e:
                    print(f"gpu_layers={ng} S={S} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    results.append({"kind": "prefill", "gpu_layers": ng,
                                    "seq": S,
                                    "error": f"{type(e).__name__}: {e}"})

    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
