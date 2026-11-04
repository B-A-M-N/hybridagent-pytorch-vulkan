"""Verify k_norm and q_norm row handling at S=529 directly.

k_norm err (5.188e-04) is ~16x q_norm err (3.147e-05) at S=529. That ratio
equals (n_heads/n_kv_heads)^2, which would suggest k_norm is normalizing over
the wrong row extent. This test checks both paths against an explicit reference
that normalizes each head_dim vector independently.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

S = 529


def ref_norm(x, w, eps=1e-6):
    """Normalize the LAST dimension independently per row."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def main():
    tvk.init(0)
    torch.manual_seed(0)

    for label, n_heads, w_shape in (("q_norm", 32, 128), ("k_norm", 8, 128)):
        # raw projection [S, n_heads, 128]
        raw = torch.randn(S, n_heads, w_shape)
        w = torch.randn(w_shape)
        ref = ref_norm(raw, w)

        # the layer's path: reshape(S*n_heads, 128) then rmsnorm
        rows = raw.reshape(S * n_heads, w_shape)
        xb = _Buf(rows)
        wb = _Buf(w)
        ob = _Buf.out((rows.shape[0], w_shape))
        tvk.rmsnorm(xb.buf, wb.buf, ob.buf, rows.shape[0], w_shape, 1e-6)
        got = ob.cpu().reshape(S, n_heads, w_shape)

        err = float((got - ref).abs().max())
        print(f"{label}: rows={rows.shape[0]} err={err:.3e}")

        # also check the ALTERNATIVE interpretation: normalize over n_heads*w_shape
        # (which is what would happen if rs were the full row width)
        ref_wrong = ref_norm(raw.reshape(S, n_heads * w_shape), w.repeat(n_heads))
        ref_wrong = ref_wrong.reshape(S, n_heads, w_shape)
        print(f"   vs wrong-row-width ref: {float((got - ref_wrong).abs().max()):.3e}")


if __name__ == "__main__":
    main()
