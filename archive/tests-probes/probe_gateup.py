"""Verify the fused_gate_up weight layout."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

I, H = 12288, 4096
S = 4


def main():
    torch.manual_seed(0)
    tvk.init(0)
    gate_w = torch.randn(I, H) * 0.02
    up_w = torch.randn(I, H) * 0.02
    x = torch.randn(S, H)

    ref = torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())

    # layer's stacking: cat([gate, up], dim=0) -> [2I, H], then .t() -> [H, 2I]
    wgu = torch.cat([gate_w, up_w], dim=0).t().contiguous()
    print("wgu shape:", tuple(wgu.shape), "(expect [H, 2I] =", (H, 2 * I), ")")
    gu = _Buf.out((S, I))
    tvk.fused_gate_up(_Buf(x).buf, _Buf(wgu).buf, gu.buf, S, H, I)
    got = gu.cpu()
    print(f"err: {float((got - ref).abs().max()):.3e}")

    # alternative: [H, 2I] with up first
    wgu2 = torch.cat([up_w, gate_w], dim=0).t().contiguous()
    gu2 = _Buf.out((S, I))
    tvk.fused_gate_up(_Buf(x).buf, _Buf(wgu2).buf, gu2.buf, S, H, I)
    print(f"err (up-first): {float((gu2.cpu() - ref).abs().max()):.3e}")


if __name__ == "__main__":
    main()
