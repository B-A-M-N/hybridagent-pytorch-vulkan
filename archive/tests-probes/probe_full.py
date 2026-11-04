"""Full-layer diff with per-stage prints, comparing offload vs CPU reference."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

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
    cfg = qm.config
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    pos = S - 1
    print(f"S={S} pos={pos}")

    x = qm.embed_tokens(ids)
    w = {k: v.to(torch.bfloat16) for k, v in qm.layers[0].state_dict().items()}

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

    h = rms(x.float()) * w["input_layernorm.weight"]
    q = (h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD)
    k = (h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD)
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    q = rms(q) * w["self_attn.q_norm.weight"]
    k = rms(k) * w["self_attn.k_norm.weight"]

    inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.tensor([float(pos)]), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HD // 2

    def rope(t):
        return t * c + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * s

    q, k = rope(q), rope(k)

    # CPU attention
    krep = k.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    vrep = v.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)
    sc = HD ** -0.5
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc
    qp = torch.arange(S)
    allow = torch.arange(S).unsqueeze(0) <= qp.unsqueeze(1)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    aw = torch.softmax(scores, dim=-1)
    attn_cpu = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)
    out_cpu = attn_cpu @ w["self_attn.o_proj.weight"].t()
    h_cpu = x + out_cpu
    residual = h_cpu
    h2 = rms(h_cpu.float()) * w["post_attention_layernorm.weight"]
    mlp_cpu = (torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].t())
               * (h2 @ w["mlp.up_proj.weight"].t())) @ w["mlp.down_proj.weight"].t()
    final_cpu = residual + mlp_cpu
    print(f"cpu final[0,:3] = {final_cpu[0,:3].tolist()}")

    # Vulkan full layer
    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    pe = qm.rotary_emb(x, torch.arange(S).unsqueeze(0))
    with torch.no_grad():
        final_vk = layer(x, position_embeddings=pe, attention_mask=None,
                         past_key_values=None,
                         cache_position=torch.arange(S))
    print(f"vk  final[0,:3] = {final_vk[0,0,:3].tolist()}")
    print(f"layer err: {float((final_vk.float() - final_cpu).abs().max()):.3e}")


if __name__ == "__main__":
    main()
