"""
Placement profiler: measure the marginal cost of a GPU layer and pick a partition.

Why a sweep and not a model
---------------------------
A single GPU layer measured in ISOLATION is faster than a CPU layer at every
sequence length (42 vs 49 ms at S=12). But the whole model gets 28.8 ms SLOWER
per GPU layer at that same length, because each offloaded layer introduces a
CPU<->GPU boundary that the isolated measurement cannot see.

So the only trustworthy input is a whole-model sweep: time the model at several
GPU-layer counts and regress T(g) against g. The slope is the marginal cost of
one GPU layer, and its SIGN tells you whether offloading helps at all.

  S=12   slope = +28.79 ms/layer   -> offloading LOSES, choose 0 GPU layers
  S=512  slope = -248.70 ms/layer  -> offloading WINS, choose as many as fit

That sign flip is the whole reason a fixed `first_n(n)` was wrong.

Usage
-----
    PYTHONPATH=backend/torch_vulkan_ext:third_party/HybridAgent:. \\
        .venv/bin/python -m backend.profile_placement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
VRAM_TOTAL = 8.57e9          # RX 6600 XT
TOTAL_LAYERS = 36


def timeit(fn, reps: int = 3) -> float:
    fn()                                     # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def marginal_cost(points: list[tuple[int, float]]) -> float:
    """Least-squares slope of T(g) against g. Negative means offloading wins."""
    n = len(points)
    if n < 2:
        return float("nan")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else float("nan")


def marginal_cost_ci(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Slope with a 95% confidence interval, from the spread of the points.

    This matters because the effect being measured can be smaller than the
    run-to-run variance. At S=12 the five sweep points span 0.120 s while the
    fitted slope was only ~0.029 s -- so the sign was never determined, and an
    earlier version of this code reported it confidently anyway.

    Returns (slope, lower, upper). If the interval straddles zero the sign is
    not established and callers must say so.
    """
    import statistics
    n = len(points)
    if n < 3:
        s = marginal_cost(points)
        return s, s, s

    slope = marginal_cost(points)
    xs = [p[0] for p in points]
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return slope, slope, slope

    # Residual standard error of the fit.
    residuals = []
    for g, t in points:
        pred = (sum(p[1] for p in points) / n) + slope * (g - mean_x)
        residuals.append(t - pred)
    s2 = sum(r * r for r in residuals) / (n - 2)
    se = (s2 / sxx) ** 0.5 if s2 > 0 else 0.0

    # ~2.78 is t(0.975, 3) for n=5; good enough for a screening estimate.
    tcrit = 2.776 if n == 5 else 2.0
    return slope, slope - tcrit * se, slope + tcrit * se


def slope_is_established(points: list[tuple[int, float]]) -> bool:
    """True only if the slope's confidence interval excludes zero."""
    _, lo, hi = marginal_cost_ci(points)
    if lo != lo or hi != hi:          # NaN
        return False
    return (lo < 0 < hi) is False


def sweep(qm, lm, seq_lens, layer_counts, reps=3):
    """Time the whole model at each (seq_len, gpu_layers) combination."""
    results = []
    for S in seq_lens:
        ids = torch.randint(0, 1000, (1, S))
        for ng in layer_counts:
            from backend.hybrid_offload import HybridBackbone, Placement
            if ng == 0:
                model = lm
            else:
                model = HybridBackbone(qm, Placement.first_n(ng, len(qm.layers)))
            with torch.no_grad():
                t = timeit(lambda: model(input_ids=ids, use_cache=True), reps)
            results.append({"seq_len": S, "gpu_layers": ng, "seconds": t})
            print(f"  S={S:5d}  gpu_layers={ng:2d}  {t:9.3f}s", flush=True)
    return results


def recommend(results, vram_total=VRAM_TOTAL, total_layers=TOTAL_LAYERS,
              reserve_fraction=0.30):
    """Turn a sweep into a placement recommendation per sequence length.

    The marginal cost (slope of T vs gpu_layers) is the only trustworthy signal:
    a single GPU layer measured in isolation is faster than a CPU layer at every
    length, yet the whole model can get slower per GPU layer because of the
    CPU<->GPU boundary. Only the sweep sees the boundary.
    """
    out = []
    by_seq = {}
    for r in results:
        by_seq.setdefault(r["seq_len"], []).append((r["gpu_layers"], r["seconds"]))

    for S, pts in sorted(by_seq.items()):
        pts.sort()
        slope = marginal_cost(pts)
        base = dict(pts).get(0)
        s, lo, hi = marginal_cost_ci(pts)
        established = slope_is_established(pts)
        out.append({
            "seq_len": S,
            "marginal_cost_ms": slope * 1000,
            "ci95_ms": [lo * 1000, hi * 1000],
            "sign_established": bool(established),
            "offload_helps": (bool(slope < 0) if established else None),
            "verdict": ("offloading HELPS" if (established and slope < 0)
                        else "offloading HURTS" if (established and slope > 0)
                        else "INDETERMINATE (CI straddles zero)"),
            "all_cpu_seconds": base,
            "points": [{"gpu_layers": g, "seconds": t} for g, t in pts],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="12,512")
    ap.add_argument("--layer-counts", default="0,1,2,4,6")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="reports/placement_profile.json")
    args = ap.parse_args()

    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    layer_counts = [int(x) for x in args.layer_counts.split(",")]

    print(f"VRAM {VRAM_TOTAL/1e9:.2f} GB, {TOTAL_LAYERS} layers")
    print(f"sweeping seq_lens={seq_lens} layer_counts={layer_counts}\n")

    results = sweep(qm, lm, seq_lens, layer_counts, args.reps)

    print("\n=== recommendation ===")
    for rec in recommend(results):
        sign = "HELPS" if rec["offload_helps"] else "HURTS"
        print(f"  S={rec['seq_len']:5d}: marginal {rec['marginal_cost_ms']:+8.2f} ms"
              f"/layer  -> offloading {sign}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "recommendation": recommend(results)},
                  f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
