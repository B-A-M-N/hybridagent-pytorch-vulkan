"""Compare the RoPE output of the kernel against a per-token CPU reference."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
HD_ = NH * HD
KVD = NKV * HD
S = 12


def main():
    torch.manual_seed(0)
    tvk.init(0)
    q = torch.randn(S, NH, HD)
    k = torch.randn(S, NKV, HD)
    v = torch.randn(S, NKV, HD)
    qkv = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    qkvb = _Buf(qkv)
    tvk.rope_qkv(qkvb.buf, 0, NH, NKV, HD, n_tok=S)
    out = qkvb.cpu()

    inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.arange(S).float(), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HD // 2

    def rope(t):
        # t: [S, H, hd]; c/s are [S, hd] -> broadcast over H
        return t * c.unsqueeze(1) + torch.cat(
            [-t[..., half:], t[..., :half]], dim=-1) * s.unsqueeze(1)

    exp_q = rope(q).reshape(-1)
    exp_k = rope(k).reshape(-1)
    got_q = out[:S * HD_]
    got_k = out[S * HD_:S * HD_ + S * KVD]
    got_v = out[S * HD_ + S * KVD:]
    print(f"Q err: {float((got_q - exp_q).abs().max()):.3e}")
    print(f"K err: {float((got_k - exp_k).abs().max()):.3e}")
    print(f"V err: {float((got_v - v.reshape(-1)).abs().max()):.3e}")


if __name__ == "__main__":
    main()
