"""Check whether q/k/v arriving at the attention kernel are correct at S=529.

The attention weights are wrong (err 0.998) but rowsums are 1.0. Either the
inputs are wrong, or the score computation is. This test feeds KNOWN q/k/v
directly to attention2 at S=529 and checks the weights against a CPU reference.
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
SCALING = 1.0 / (HD ** 0.5)


def main():
    tvk.init(0)
    torch.manual_seed(0)

    for S in (2, 12, 128, 512, 529):
        q = torch.randn(NH, S, HD)
        k = torch.randn(NKV, S, HD)
        v = torch.randn(NKV, S, HD)
        allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
        mask = torch.zeros(NH, S, S).masked_fill(~allow, float("-inf"))

        kh = k.repeat_interleave(NH // NKV, dim=0)
        vh = v.repeat_interleave(NH // NKV, dim=0)
        ref_w = torch.softmax(
            torch.matmul(q, kh.transpose(-1, -2)) * SCALING + mask, dim=-1)
        ref_o = torch.matmul(ref_w, vh)

        qb = _Buf(q.contiguous())
        kb = _Buf(k.contiguous())
        vb = _Buf(v.contiguous())
        mb = _Buf(mask.contiguous())
        pb = _Buf.out((NH, S, S))
        ob = _Buf.out((NH, S, HD))
        tvk.attention2(qb.buf, kb.buf, vb.buf, mb.buf, pb.buf, ob.buf,
                       0, S, NH, NH // NKV, HD, S, SCALING)
        got_w = pb.cpu()
        got_o = ob.cpu()

        we = float((got_w - ref_w).abs().max())
        oe = float((got_o - ref_o).abs().max())
        print(f"S={S:4d}  probs err={we:.3e}  out err={oe:.3e}  "
              f"rowsum={float(got_w.sum(-1).mean()):.6f}")


if __name__ == "__main__":
    main()
