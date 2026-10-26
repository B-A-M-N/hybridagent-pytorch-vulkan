"""Isolate the attention stage: offload layer vs CPU, same q/k/v."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
H, I, NH, NKV, HD = 4096, 12288, 32, 8, 128
HD_ = NH * HD
KVD = NKV * HD


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    pos = S - 1
    print(f"S={S} pos={pos}")

    # CPU reference attention over the full window
    x = qm.embed_tokens(ids)
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

    h = rms(x.float()) * w["input_layernorm.weight"]
    q = (h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD)
    k = (h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD)
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    q = rms(q) * w["self_attn.q_norm.weight"]
    k = rms(k) * w["self_attn.k_norm.weight"]

    # RoPE
    inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.tensor([float(pos)]), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HD // 2

    def rope(t):
        return t * c + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * s

    q, k = rope(q), rope(k)

    # CPU attention with causal mask
    krep = k.permute(1, 0, 2)          # [nkv, S, hd]
    vrep = v.permute(1, 0, 2)
    krep = krep.repeat_interleave(NH // NKV, dim=0)   # [nh, S, hd]
    vrep = vrep.repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)            # [nh, S, hd]
    sc = HD ** -0.5
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc
    qp = torch.arange(S)
    allow = torch.arange(S).unsqueeze(0) <= qp.unsqueeze(1)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    aw = torch.softmax(scores, dim=-1)
    attn_cpu = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)

    # Vulkan attention via the kernel
    kcache = k.permute(1, 0, 2).contiguous()   # [nkv, S, hd]
    vcache = v.permute(1, 0, 2).contiguous()
    q_in = q.permute(1, 0, 2).contiguous()     # [nh, S, hd]
    allow_vk = torch.arange(S).unsqueeze(0) <= qp.unsqueeze(1)
    mask = torch.zeros(1, NH, S, S).masked_fill(~allow_vk.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = tvk.VkBuffer(max(1, NH * S * S) * 4, "device")
    ob = tvk.VkBuffer(max(1, NH * S * HD) * 4, "host")
    tvk.attention2(_Buf(q_in).buf, _Buf(kcache).buf, _Buf(vcache).buf,
                   _Buf(mask).buf, probs, ob, 0, S, NH, NH // NKV, HD, S, sc)
    attn_vk = ob.download((NH, S, HD)).permute(1, 0, 2).reshape(S, HD_)

    print(f"attention err: {float((attn_vk - attn_cpu).abs().max()):.3e}")
    print(f"  cpu[0,:3] {attn_cpu[0,:3].tolist()}")
    print(f"  vk [0,:3] {attn_vk[0,:3].tolist()}")


if __name__ == "__main__":
    main()
