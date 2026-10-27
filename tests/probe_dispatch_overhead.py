"""
Controlled test of the CLAIMED root cause of the "4000x slowdown".

The previous report attributed the slowdown to "Descriptor Set Reallocation Per
Dispatch" in vk_backend.cpp, and recommended rewriting descriptor-set
management. That attribution was made WITHOUT profiling evidence -- it was
inferred from reading the source. The user explicitly asked:

    "Do not attribute a 4,000x slowdown to small tiles or missing fusion
     without profiling evidence."

This probe tests that attribution with a parameter-matched control.

Design
------
The dispatch path in vk_backend.cpp reallocates a descriptor set only when the
buffer bindings CHANGE:

    bool need = bnd.set == VK_NULL_HANDLE || bnd.views != views || ...
    if (need) { vkAllocateDescriptorSets(...); vkUpdateDescriptorSets(...); }

So we can hold EVERYTHING constant (same shader, same op, same dispatch count,
same queue) and vary only whether the bindings change:

  A. STABLE bindings  - same input/output buffers every iteration
                       -> descriptor set is reused (cache hit)
  B. CHANGING bindings- a different input buffer each iteration
                       -> descriptor set reallocated every dispatch (cache miss)

If the descriptor-set hypothesis is right, B is much slower than A, and
(B - A) is the per-dispatch descriptor cost.

If A is also slow, the overhead is NOT descriptor sets -- it is command
recording, submission, or the sync wait, and the previous report's fix would
NOT help. That is the outcome this probe is designed to detect.

A third arm isolates the cost of the synchronous wait:
  C. STABLE bindings, async dispatch, N dispatches then one submit
     -> removes the per-dispatch vkQueueWaitIdle entirely
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402

REPS = 7
WARMUP = 2
N_DISPATCH = 50


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


def main():
    torch.manual_seed(0)
    tvk.init(0)
    print(f"device: {tvk.device_name()}\n")

    # A trivial op so that per-dispatch overhead is visible:
    # rmsnorm on 12x4096 is ~0.37 ms of real work but ~20-30 dispatches/layer
    # is the real cost driver.
    M, K = 12, 4096
    x = torch.randn(M, K)
    w = torch.ones(K) * 0.02

    xa = tvk.Allocation(x.numel() * 4, "device"); xv = xa.view(); xv.upload(x)
    wa = tvk.Allocation(w.numel() * 4, "device"); wv = wa.view(); wv.upload(w)
    oa = tvk.Allocation(x.numel() * 4, "device"); ov = oa.view()

    # A pool of distinct input buffers for the CHANGING-bindings arm.
    # All are pre-allocated, so arm B does not pay allocation cost --
    # it pays ONLY the descriptor update/realloc.
    pool = []
    for _ in range(N_DISPATCH):
        a = tvk.Allocation(x.numel() * 4, "device")
        v = a.view()
        v.upload(x)
        pool.append((a, v))

    def arm_stable(sync=True, n=N_DISPATCH):
        def go():
            for _ in range(n):
                tvk.rmsnorm(xv, wv, ov, M, K, 1e-6, sync=sync)
            if not sync:
                tvk.submit()
        return go

    def arm_changing(sync=True, n=N_DISPATCH):
        def go():
            for i in range(n):
                av = pool[i % len(pool)][1]
                tvk.rmsnorm(av, wv, ov, M, K, 1e-6, sync=sync)
            if not sync:
                tvk.submit()
        return go

    results = {}

    results["A stable/sync"] = timeit(arm_stable(sync=True))
    results["B changing/sync"] = timeit(arm_changing(sync=True))
    results["C stable/async"] = timeit(arm_stable(sync=False))
    results["D changing/async"] = timeit(arm_changing(sync=False))

    print(f"per-dispatch cost, rmsnorm [{M},{K}], {N_DISPATCH} dispatches per rep")
    print("-" * 66)
    for k, v in results.items():
        print(f"  {k:16s} {v:9.3f} ms/rep   {v/N_DISPATCH:8.4f} ms/dispatch")
    print()

    a = results["A stable/sync"] / N_DISPATCH
    b = results["B changing/sync"] / N_DISPATCH
    c = results["C stable/async"] / N_DISPATCH
    d = results["D changing/async"] / N_DISPATCH

    print("DECOMPOSITION (ms/dispatch)")
    print(f"  fixed cost, bindings reused      (A)           {a:8.4f}")
    print(f"  fixed cost, bindings change      (B)           {b:8.4f}")
    print(f"  descriptor-update cost           (B - A)       {b-a:+8.4f}")
    print(f"  sync-wait cost                   (A - C)       {a-c:+8.4f}")
    print(f"  async fixed cost                 (C)           {c:8.4f}")
    print(f"  async descriptor cost            (D - C)       {d-c:+8.4f}")
    print()

    # The real op, for scale: the actual gate_up with resident weights
    Nw = 12288
    wg = torch.randn(Nw, K) * 0.02
    wga = tvk.Allocation(wg.numel() * 4, "device"); wgv = wga.view(); wgv.upload(wg)
    oga = tvk.Allocation(M * Nw * 4, "device"); ogv = oga.view()

    def real_op():
        tvk.fused_gate_up(xv, wgv, ogv, M, K, Nw)
    t_real = timeit(real_op, reps=10)
    print(f"real op (gate_up, resident weights)  {t_real:8.3f} ms/call")
    print(f"  vs measured whole-layer forward   {29.64:8.3f} ms")
    print()

    print("VERDICT")
    if (b - a) > 0.5 * a:
        print("  Descriptor-set reallocation DOMINATES -> the previous report's")
        print("  root cause is supported by measurement.")
    else:
        print("  Descriptor-set reallocation does NOT dominate.")
        print("  The overhead is in the fixed per-dispatch path (recording,")
        print("  submission, or the sync wait), NOT descriptor management.")
    print(f"  A layer issues ~20-30 dispatches -> {20*a:.2f} to {30*a:.2f} ms of")
    print(f"  pure per-dispatch overhead at S=12 (measured layer: 29.64 ms).")


if __name__ == "__main__":
    main()
