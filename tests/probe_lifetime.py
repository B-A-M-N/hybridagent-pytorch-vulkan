"""
Prove the async GPUVM fault is a buffer-lifetime bug, and prove the fix.

Hypothesis (from review): a queued dispatch references a VkBuffer that Python
frees when its _Buf wrapper goes out of scope. Submission is not completion, so
the GPU reads unmapped VA -> GCVM_L2_PROTECTION_FAULT -> context lost.

This test:
  1. reproduces the fault with the naive pattern (drop the wrapper, keep the
     queue in flight);
  2. shows the same workload is stable when the buffer is held until submit();
  3. confirms results are correct in the held case.
"""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402

M, K, N = 12, 4096, 4096


def main():
    torch.manual_seed(0)
    tvk.init(0)
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.02
    ref = x @ w.t()

    xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(N * 4, "device"); bb.upload(torch.zeros(N))
    ob = tvk.VkBuffer(M * N * 4, "host")

    # --- A: hold the buffers across the queued region (correct pattern)
    tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
    tvk.submit()          # drain before reading
    got = ob.download((M, N))
    err = float((got - ref).abs().max())
    print(f"A held-buffers async : err {err:.3e}")
    assert err < 1e-2, "held-buffer async path is wrong"

    # --- B: simulate the layer's pattern -- a temporary _Buf is created and
    #        dropped while the queue is still in flight.
    class _Buf:
        __slots__ = ("buf", "shape")

        def __init__(self, t):
            t = t.detach().float().contiguous()
            self.buf = tvk.VkBuffer(max(1, t.numel()) * 4, "device")
            self.buf.upload(t)
            self.shape = tuple(t.shape)

    faults = 0
    for i in range(30):
        # temporary buffer created, queued, then dropped immediately
        tmp = _Buf(torch.randn(M, K))
        tvk.matmul_t_bias(tmp.buf, wb, bb, ob, M, K, N, sync=False)
        del tmp                      # freed before submit: the unsafe pattern
        try:
            tvk.submit()
        except Exception as e:
            faults += 1
            print(f"  iter {i}: submit failed: {str(e)[:60]}")
            break
    print(f"B dropped-temp async : {faults} faults in 30 iters")

    # --- C: same workload with the temp held until submit()
    ok = 0
    for i in range(30):
        tmp = _Buf(torch.randn(M, K))
        tvk.matmul_t_bias(tmp.buf, wb, bb, ob, M, K, N, sync=False)
        tvk.submit()
        del tmp
        ok += 1
    print(f"C held-until-submit : {ok}/30 clean")


if __name__ == "__main__":
    main()
