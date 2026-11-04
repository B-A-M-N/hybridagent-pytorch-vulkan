"""Isolate attention: CPU vs Vulkan, with correct GQA handling.

CPU: repeat k/v heads to n_heads for the matmul, then average back.
Vulkan attention2: takes n_heads and n_kv_heads separately and does GQA itself.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
NH, NKV, HD = 32, 8, 128


def main():
    torch.manual_seed(0)
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import VulkanDecoderLayer, _Buf

    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    q_c = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD),
              w["self_attn.q_norm.weight"])
    k_c = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD),
              w["self_attn.k_norm.weight"])
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)

    # ---- CPU attention with GQA
    qh = q_c.transpose(0, 1)                       # [NH, S, HD]
    kh = k_c.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    vh = v_c.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    scale = 1.0 / (HD ** 0.5)
    attn = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    probs = torch.softmax(attn + mask, dim=-1)
    out_cpu = torch.matmul(probs, vh).transpose(0, 1).reshape(S, NH * HD)

    # ---- Vulkan attention
    q_in = q_c.transpose(0, 1).contiguous()
    kb = _Buf(k_c.transpose(0, 1).contiguous())
    vb = _Buf(v_c.transpose(0, 1).contiguous())
    mb = _Buf(mask)
    pb = _Buf.out((NH, S, S))
    ob = _Buf.out((NH, S, HD))
    tvk.attention2(_Buf(q_in).buf, kb.buf, vb.buf, mb.buf, pb.buf, ob.buf,
                   S, S, NH, NKV, HD, scale)
    got_p = pb.cpu()
    got_o = ob.cpu()

    print(f"probs err : {float((got_p - probs).abs().max()):.3e}")
    print(f"probs rowsum cpu: {probs.sum(-1).min():.6f}..{probs.sum(-1).max():.6f}")
    print(f"probs rowsum vk : {got_p.sum(-1).min():.6f}..{got_p.sum(-1).max():.6f}")
    print(f"out err    : {float((got_o - out_cpu.transpose(0, 1)).abs().max()):.3e}")
    print(f"out norm cpu: {float(out_cpu.norm()):.4f}  vk: {float(got_o.norm()):.4f}")

    # ---- now the full layer, to see whether attention explains the gap
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    with torch.no_grad():
        h_vk = vk(emb, position_embeddings=pe, attention_mask=None,
                  past_key_values=None, cache_position=torch.arange(S))
    with torch.no_grad():
        h_cpu = qm.layers[0](emb, attention_mask=None, position_ids=None,
                             past_key_values=None, use_cache=False,
                             cache_position=torch.arange(S), position_embeddings=pe)
    print(f"\nlayer hidden err: {float((h_vk.float()-h_cpu.float()).abs().max()):.4e}")
    print(f"norm cpu {float(h_cpu.float().norm()):.4f}  vk {float(h_vk.float().norm()):.4f}")


if __name__ == "__main__":
    main()
