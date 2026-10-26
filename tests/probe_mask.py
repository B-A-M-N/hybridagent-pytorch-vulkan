"""Compare the attention mask built by the layer against a CPU causal mask."""
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
    # replicate the layer's mask construction
    pos_start = 0
    n_q = S
    cap = S
    first_pos = pos_start
    q_pos = torch.arange(first_pos, first_pos + S)
    k_pos = torch.arange(cap)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)      # [S, cap]
    mask = torch.zeros(1, NH, n_q, cap)
    mask = mask.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
    print("mask shape:", tuple(mask.shape))
    print("allow[0] (query 0):", allow[0].int().tolist())
    print("allow[11] (query 11):", allow[11].int().tolist())
    print("mask[0,0,0,:5]:", mask[0, 0, 0, :5].tolist())
    print("mask[0,0,11,:14]:", mask[0, 0, 11, :14].tolist())

    # Now run attention with this mask and compare to a CPU softmax
    q = torch.randn(S, NH, HD)
    k = torch.randn(S, NKV, HD)
    v = torch.randn(S, NKV, HD)
    krep = k.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    vrep = v.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)
    sc = HD ** -0.5
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc
    scores = scores + mask[0]
    aw = torch.softmax(scores, dim=-1)
    attn_cpu = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)

    kcache = k.permute(1, 0, 2).contiguous()
    vcache = v.permute(1, 0, 2).contiguous()
    q_in = q.permute(1, 0, 2).contiguous()
    probs = tvk.VkBuffer(max(1, NH * S * S) * 4, "device")
    ob = tvk.VkBuffer(max(1, NH * S * HD) * 4, "host")
    tvk.attention2(_Buf(q_in).buf, _Buf(kcache).buf, _Buf(vcache).buf,
                   _Buf(mask).buf, probs, ob, 0, S, NH, NH // NKV, HD, S, sc)
    attn_vk = ob.download((NH, S, HD)).permute(1, 0, 2).reshape(S, HD_)
    print(f"attn err: {float((attn_vk - attn_cpu).abs().max()):.3e}")


if __name__ == "__main__":
    main()
