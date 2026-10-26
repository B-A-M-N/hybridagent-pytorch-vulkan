"""Test whether attention2 is correct for the LAYER's calling pattern.

The layer calls attention2 with n_q = S (one query position per head, stacked).
The operator test calls it with n_q = 1 (decode shape).

If attention2 is correct at n_q=1 but wrong at n_q=S, that is exactly the
regression: the layer's prefill path is broken while decode works.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128


def cpu_attn(q, k, v, mask, scaling):
    """q [NH,S,HD], k/v [NKV,S,HD] -> [NH,S,HD] with GQA + causal mask."""
    kh = k.repeat_interleave(NH // NKV, dim=0)
    vh = v.repeat_interleave(NH // NKV, dim=0)
    attn = torch.matmul(q, kh.transpose(-1, -2)) * scaling
    if mask is not None:
        attn = attn + mask
    return torch.matmul(torch.softmax(attn, dim=-1), vh)


def run(S, cap, label):
    torch.manual_seed(0)
    scaling = 1.0 / (HD ** 0.5)
    q = torch.randn(NH, S, HD)
    k = torch.randn(NKV, cap, HD)
    v = torch.randn(NKV, cap, HD)
    # causal mask over [S, cap]: query i (abs pos i) sees keys <= i
    allow = torch.arange(cap).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, NH, S, cap).masked_fill(~allow, float("-inf"))

    ref = cpu_attn(q, k, v, mask, scaling)

    kb = _Buf(k.contiguous())
    vb = _Buf(v.contiguous())
    mb = _Buf(mask.contiguous())
    pb = _Buf.out((NH, S, cap))
    ob = _Buf.out((NH, S, HD))
    tvk.attention2(_Buf(q.contiguous()).buf, kb.buf, vb.buf, mb.buf, pb.buf,
                   ob.buf, 0, cap, NH, NH // NKV, HD, cap, scaling)
    got = ob.cpu()
    err = float((got - ref).abs().max())
    print(f"{label}: S={S} cap={cap} err={err:.3e}  "
          f"norm ref={float(ref.norm()):.3f} vk={float(got.norm()):.3f}")
    return err


def main():
    tvk.init(0)
    print("=== attention2: decode shape vs prefill shape ===")
    run(1, 1, "decode  ")
    run(1, 32, "decode  ")
    run(4, 4, "prefill ")
    run(12, 12, "prefill ")
    run(12, 32, "prefill ")


if __name__ == "__main__":
    main()
