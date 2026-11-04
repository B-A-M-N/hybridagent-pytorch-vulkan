"""Verify RoPE at long sequence length.

Layer-0 cosine is 0.933 at S=529 (0.93 at S=12 was acceptable). attention2 is
now proven exact to 1e-7 at every length, so the remaining gap is elsewhere.
RoPE is the prime suspect: rope_qkv applies rotations using pos_start + token
index, and an error there would grow with sequence length.
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


def cpu_rope_qkv(qkv, pos_start, n_heads, n_kv_heads, head_dim, n_tok):
    """Reference: Qwen3 interleaved rotate-half RoPE on a packed [Q|K|V] buffer."""
    q_len = n_heads * head_dim
    kv_len = n_kv_heads * head_dim
    out = qkv.clone()
    nhalf = head_dim // 2
    inv = 1000000.0 ** (-2.0 * torch.arange(nhalf).float() / head_dim)
    for t in range(n_tok):
        for h in range(n_heads + n_kv_heads):
            base = (t * q_len if h < n_heads
                    else n_tok * q_len + t * kv_len) + (h % n_heads if h < n_heads
                                                        else (h - n_heads)) * head_dim
            ang = (pos_start + t) * inv
            c, s = torch.cos(ang), torch.sin(ang)
            x0 = qkv[base:base + nhalf].clone()
            x1 = qkv[base + nhalf:base + head_dim].clone()
            out[base:base + nhalf] = x0 * c - x1 * s
            out[base + nhalf:base + head_dim] = x1 * c + x0 * s
    return out


def main():
    tvk.init(0)
    torch.manual_seed(0)
    for S in (1, 2, 12, 529):
        q_len, kv_len = NH * HD, NKV * HD
        qkv = torch.randn(S * (q_len + 2 * kv_len))
        ref = cpu_rope_qkv(qkv, 0, NH, NKV, HD, S)

        qb = _Buf(qkv.clone())
        tvk.rope_qkv(qb.buf, 0, NH, NKV, HD, n_tok=S)
        got = qb.cpu()

        err = float((got - ref).abs().max())
        print(f"S={S:4d}  rope_qkv err={err:.3e}")


if __name__ == "__main__":
    main()
