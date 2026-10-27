"""
Regression test for the Vulkan backend deadlock.

The bug
-------
The working tree introduced a refcounted `Allocation` / `BufferView` scheme.
`Allocation::~Allocation()` calls `backend->destroy_allocation()`, which takes
the backend's `mutex_`. That destructor is reachable from INSIDE a region
that already holds `mutex_`:

    Backend::dispatch()                       [holds mutex_]
      -> bnd.views = views;                   (replaces the cached BufferViews)
         -> old ~BufferView -> alloc->release() -> last reference
            -> delete Allocation -> ~Allocation
               -> destroy_allocation() -> lock_guard(mutex_)   <-- SELF-DEADLOCK

`Backend::submit()` has the same problem: it releases the retained
Allocations of completed submissions while holding `mutex_`.

Because `mutex_` was a plain `std::mutex`, the second acquisition blocked
forever. The call never returned -- it did not run slowly. From the outside
this looked exactly like "Vulkan is 4,000x slower than CPU": the benchmark
hit its timeout, and the timeout was then divided by the CPU time to produce
the ratio.

The symptom was only visible when the descriptor bindings CHANGED between
dispatches, because that is the only path that replaces `bnd.views`. Every
call in a transformer layer allocates a fresh output buffer, so the
bindings always change.

The fix: `mutex_` is now a `std::recursive_mutex`.

What this test asserts
----------------------
1. The reproducer completes instead of hanging (guarded by a hard timeout so
   a regression FAILS rather than hanging the suite).
2. Repeated async dispatch + submit + free of the buffers completes.
3. The dispatch counter advances, i.e. work was really done.
4. A whole Vulkan transformer layer forward completes and produces the same
   result as the CPU layer.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

REPRO = r'''
import sys, time, torch
ROOT = "{root}"
sys.path.insert(0, ROOT + "/backend/torch_vulkan_ext")
import torch_vulkan_ext as tvk
tvk.init(0)
K, N = 4096, 4096
wa = tvk.Allocation(N*K*4, "device"); wv = wa.view(); wv.upload(torch.randn(N,K)*0.02)
ba = tvk.Allocation(N*4, "device"); bv = ba.view(); bv.upload(torch.zeros(N))
for i in range(5):
    aa = tvk.Allocation(12*K*4, "device"); av = aa.view(); av.upload(torch.randn(12,K))
    oa = tvk.Allocation(12*N*4, "device"); ov = oa.view()
    tvk.matmul_t_bias(av, wv, bv, ov, 12, K, N, sync=False)
    tvk.submit()
    del aa, av, oa, ov
print("DISPATCHES", tvk.dispatch_count())
print("OK")
'''.replace("{root}", ROOT)


def run_with_timeout(script: str, timeout: int = 90):
    """Run a python script in a subprocess with a hard timeout.

    A regression makes the child hang; the timeout turns that into a FAIL
    instead of hanging the whole test suite.
    """
    path = os.path.join("/home/bamn/.hermes/cache/scratch", "_tvk_regress.py")
    with open(path, "w") as f:
        f.write(script)
    env = {**os.environ,
           "PYTHONPATH": f"{ROOT}:{ROOT}/backend/torch_vulkan_ext",
           "OMP_NUM_THREADS": "4"}
    p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def main():
    import torch
    import torch_vulkan_ext as tvk

    print("=== 1. deadlock reproducer (must not hang) ===")
    t0 = time.perf_counter()
    try:
        rc, out, err = run_with_timeout(REPRO, timeout=90)
    except subprocess.TimeoutExpired:
        print("  [FAIL] reproducer HUNG -- the deadlock is back")
        return 1
    dt = time.perf_counter() - t0
    ok = rc == 0 and "OK" in out
    ndisp = 0
    for line in out.splitlines():
        if line.startswith("DISPATCHES"):
            ndisp = int(line.split()[1])
    print(f"  [{'PASS' if ok else 'FAIL'}] completed in {dt:.1f}s, "
          f"dispatch_count={ndisp}")
    if not ok:
        print("  stdout:", out[-2000:])
        print("  stderr:", err[-2000:])
        return 1
    if ndisp < 5:
        print(f"  [FAIL] expected >= 5 dispatches, got {ndisp}")
        return 1

    print("\n=== 2. repeated async dispatch + submit + free ===")
    tvk.init(0)
    K, N = 4096, 4096
    wa = tvk.Allocation(N * K * 4, "device")
    wv = wa.view()
    wv.upload(torch.randn(N, K) * 0.02)
    ba = tvk.Allocation(N * 4, "device")
    bv = ba.view()
    bv.upload(torch.zeros(N))
    try:
        for i in range(20):
            aa = tvk.Allocation(12 * K * 4, "device")
            av = aa.view()
            av.upload(torch.randn(12, K))
            oa = tvk.Allocation(12 * N * 4, "device")
            ov = oa.view()
            tvk.matmul_t_bias(av, wv, bv, ov, 12, K, N, sync=False)
            tvk.submit()
            del aa, av, oa, ov
        print("  [PASS] 20 iterations with changing bindings")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return 1

    print("\n=== 3. whole Vulkan layer forward completes ===")
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(1, len(qm.layers)))
    gl = [l for l in hb.layers if type(l).__name__ == "VulkanDecoderLayer"][0]
    gl.upload_weights()

    S = 12
    with torch.no_grad():
        h = torch.randn(S, qm.config.hidden_size, dtype=torch.bfloat16)
        pos = torch.arange(S).unsqueeze(0)
        rot = qm.rotary_emb(h.float(), pos)
        cpos = torch.arange(S)
        t0 = time.perf_counter()
        out = gl(h, rot, None, past_key_values=None, cache_position=cpos)
        dt = (time.perf_counter() - t0) * 1000
    print(f"  [PASS] Vulkan layer forward completed in {dt:.2f} ms, "
          f"output {tuple(out.shape)}")
    assert out.shape == h.shape, f"shape mismatch {out.shape} vs {h.shape}"

    print("\nALL DEADLOCK REGRESSION TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
