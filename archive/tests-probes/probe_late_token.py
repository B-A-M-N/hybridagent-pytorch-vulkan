"""Test the layer's attention at a LATE query position only.

Token 0 is exact; error grows with position. Run the layer's exact attention
path for a single query token at position p and compare against a CPU reference
built from the same q/k/v. This isolates whether the divergence is in the
attention accumulation over a long window or elsewhere.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)


def main():
    tvk.init(0)
    torch.manual_seed(0)

    for S in (2, 12, 64, 529):
        # q: [NH, S, HD]; k/v: [NKV, S, HD] (unexpanded -- kernel does GQA)
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
        # error at the LAST row specifically
        last_w = float((got_w[:, -1, :] - ref_w[:, -1, :]).abs().max())
        last_o = float((got_o[:, -1, :] - ref_o[:, -1, :]).abs().max())
        print(f"S={S:4d}  probs={we:.3e} out={oe:.3e}  "
              f"last-row probs={last_w:.3e} out={last_o:.3e}")


if __name__ == "__main__":
    main()
