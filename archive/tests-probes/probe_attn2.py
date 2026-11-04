"""Diff the two attention calls: passing probe vs assembled layer."""
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
    kcache = k.permute(1, 0, 2).contiguous()   # [nkv, S, hd]
    vcache = v.permute(1, 0, 2).contiguous()
    q_in = q.permute(1, 0, 2).contiguous()     # [nh, S, hd]
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, NH, S, S).masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
    sc = HD ** -0.5

    # CPU reference
    krep = k.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    vrep = v.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc + mask[0]
    aw = torch.softmax(scores, dim=-1)
    attn_cpu = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)

    probs = tvk.VkBuffer(max(1, NH * S * S) * 4, "device")
    ob = tvk.VkBuffer(max(1, NH * S * HD) * 4, "host")
    tvk.attention2(_Buf(q_in).buf, _Buf(kcache).buf, _Buf(vcache).buf,
                   _Buf(mask).buf, probs, ob, 0, S, NH, NH // NKV, HD, S, sc)
    attn_vk = ob.download((NH, S, HD)).permute(1, 0, 2).reshape(S, HD_)
    print(f"attn err: {float((attn_vk - attn_cpu).abs().max()):.3e}")
    print(f"  q_in shape {tuple(q_in.shape)}  n_q derived = "
          f"{(q_in.numel()//4)//(NH*HD)}")
    print(f"  cpu[0,:3] {attn_cpu[0,:3].tolist()}")
    print(f"  vk [0,:3] {attn_vk[0,:3].tolist()}")


if __name__ == "__main__":
    main()
