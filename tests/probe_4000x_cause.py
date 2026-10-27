"""
Decisive test for the "Vulkan is 4,000x slower than CPU" claim -- ISOLATED runs.

What the first version of this probe established:
  * Allocation is NOT the cause: a fresh 201 MB device allocation costs
    0.04 ms, not 250 ms.
  * The first version leaked buffers until the RADV context was lost, which
    made later rows garbage. Every case here therefore runs in its own
    subprocess with its own fresh Vulkan context, and every buffer is
    explicitly freed before the next case.

The claim under test: "Vulkan is ~4,000x slower than CPU on this hardware."

The original anomalous numbers (logs/ops_cost.log) were:
    fused_gate_up  [12,4096]->[12,12288]   258.59 ms
    matmul q_proj  [12,4096]->[12, 4096]   112.50 ms
    matmul down    [12,12288]->[12,4096]   340.86 ms
    rmsnorm        [12,4096]                 0.37 ms
    attention2     cap=12                    0.96 ms
    --------------------------------------------
    sum                                 883.28 ms

But the measured GPU layer forward at S=12 is 29.64 ms (logs/layer_lat.log).
The per-op sum exceeds the whole layer by 29.8x, so the per-op numbers cannot
all be real. This probe measures each op with:
    - weights RESIDENT on the device (the real model's behaviour)
    - weights re-uploaded per call (the bad harness' behaviour)
and splits out upload / compute / alloc.

Run:  python tests/probe_4000x_cause.py --case <name>   (one case per process)
      python tests/probe_4000x_cause.py --all            (driver, spawns cases)
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402

REPS = 10
WARMUP = 3

# (label, op, M, K, N)
CASES = [
    ("gate_up  12x4096 -> 12x12288", "fused_gate_up", 12, 4096, 12288),
    ("q_proj   12x4096 -> 12x4096",  "fused_gate_up", 12, 4096, 4096),
    ("down     12x12288 -> 12x4096", "fused_gate_up", 12, 12288, 4096),
    ("gate_up  512x4096 -> 512x12288", "fused_gate_up", 512, 4096, 12288),
]


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
    return med(ts)


def run_case(label, op, M, K, N):
    """One case, one process. Frees every buffer before returning."""
    torch.manual_seed(0)
    tvk.init(0)
    w = torch.randn(N, K) * 0.02
    a = torch.randn(M, K)
    wbytes = w.numel() * 4

    out = {"label": label, "M": M, "K": K, "N": N, "wMB": wbytes / 1e6}

    # ---- resident: weights already on device, measure compute only ----
    wa = tvk.Allocation(wbytes, "device"); wv = wa.view()
    aa = tvk.Allocation(a.numel() * 4, "device"); av = aa.view()
    oa = tvk.Allocation(M * N * 4, "device"); ov = oa.view()
    wv.upload(w); av.upload(a)

    def compute():
        tvk.fused_gate_up(av, wv, ov, M, K, N)
    out["compute"] = timeit(compute)

    # upload cost with the buffer already resident
    out["upload"] = timeit(lambda: wv.upload(w))

    # allocation cost
    def alloc():
        x = tvk.Allocation(wbytes, "device"); x.view(); del x
    out["alloc"] = timeit(alloc)

    # CPU reference
    def cpu():
        return a @ w.T
    out["cpu"] = timeit(cpu)

    # fresh-per-call (the bad harness shape) -- done LAST so it cannot
    # contaminate the measurements above with VRAM pressure
    def fresh():
        a1 = tvk.Allocation(a.numel() * 4, "device"); v1 = a1.view(); v1.upload(a)
        a2 = tvk.Allocation(wbytes, "device"); v2 = a2.view(); v2.upload(w)
        a3 = tvk.Allocation(M * N * 4, "device"); v3 = a3.view()
        tvk.fused_gate_up(v1, v2, v3, M, K, N)
        v1.free(); v2.free(); v3.free()
        a1.free(); a2.free(); a3.free()
    out["fresh"] = timeit(fresh, reps=5, warmup=2)

    del wa, wv, aa, av, oa, ov
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.case is not None:
        label, op, M, K, N = CASES[args.case]
        r = run_case(label, op, M, K, N)
        for k, v in r.items():
            print(f"{k}={v}")
        return

    # driver: each case in its own process
    results = []
    for i, c in enumerate(CASES):
        p = subprocess.run(
            [sys.executable, __file__, "--case", str(i)],
            capture_output=True, text=True,
            env={**os.environ,
                 "PYTHONPATH": f"{ROOT}:{ROOT}/backend/torch_vulkan_ext",
                 "OMP_NUM_THREADS": "12"})
        if p.returncode != 0:
            print(f"case {i} FAILED:\n{p.stderr[-2000:]}")
            continue
        d = dict(line.split("=", 1) for line in p.stdout.strip().splitlines() if "=" in line)
        for k in ("M", "K", "N"):
            d[k] = int(d[k])
        for k in ("wMB", "compute", "upload", "alloc", "cpu", "fresh"):
            d[k] = float(d[k])
        results.append(d)

    print(f"{'case':32s} {'wMB':>6} | {'alloc':>7} {'upload':>8} {'COMPUTE':>8} "
          f"{'CPU':>8} | {'fresh':>8} {'vk/cpu':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:32s} {r['wMB']:6.1f} | {r['alloc']:7.3f} "
              f"{r['upload']:8.3f} {r['compute']:8.3f} {r['cpu']:8.3f} | "
              f"{r['fresh']:8.3f} {r['compute']/max(r['cpu'],1e-9):7.2f}")
    print()
    print("RECONCILIATION")
    print("  measured GPU layer forward @S=12 : 29.64 ms  (logs/layer_lat.log)")
    print("  sum of OLD per-op numbers         : 883.28 ms (logs/ops_cost.log)")
    print("  overstatement                     : 29.8x")
    for r in results:
        print(f"  {r['label']:32s} old-claim vs compute: "
              f"upload/fresh dominates: fresh={r['fresh']:.1f}ms "
              f"compute={r['compute']:.2f}ms")


if __name__ == "__main__":
    main()
