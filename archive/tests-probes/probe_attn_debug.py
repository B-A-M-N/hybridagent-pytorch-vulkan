"""Isolate the attention stage: compare CPU vs Vulkan attention output only.

If attention is the culprit, the residual stream before attention will match
and the post-attention hidden will diverge.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


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
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    # ---- CPU reference, stage by stage
    h = rms(emb.float(), w["input_layernorm.weight"])
    q_c = (h @ w["self_attn.q_proj.weight"].t()).view(S, 32, 128)
    k_c = (h @ w["self_attn.k_proj.weight"].t()).view(S, 8, 128)
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, 8, 128)
    q_c = rms(q_c, w["self_attn.q_norm.weight"])
    k_c = rms(k_c, w["self_attn.k_norm.weight"])

    # ---- Vulkan stage by stage
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vk.upload_weights()

    def v_rms(t, wb):
        rows = t.numel() // t.shape[-1]
        xb = _Buf(t.reshape(rows, t.shape[-1]))
        ob = _Buf.out((rows, t.shape[-1]))
        tvk.rmsnorm(xb.buf, wb.buf, ob.buf, rows, t.shape[-1], 1e-6)
        return ob.cpu().reshape(t.shape)

    def v_lin(t, wb, bb, out_f):
        M = t.numel() // t.shape[-1]
        K = t.shape[-1]
        xb = _Buf(t.reshape(M, K))
        ob = _Buf.out((M, out_f))
        tvk.matmul_t_bias(xb.buf, wb.buf, bb.buf, ob.buf, M, K, out_f)
        return ob.cpu().reshape(t.shape[:-1] + (out_f,))

    h_vk = v_rms(emb.float(), vk._w["input_ln"])
    print(f"h      err: {float((h_vk - h).abs().max()):.3e}")

    q_vk = v_lin(h_vk, vk._w["q"], vk._zero_hd, 4096).view(S, 32, 128)
    print(f"q_proj err: {float((q_vk - q_c).abs().max()):.3e}")

    q_vk2 = v_rms(q_vk.cpu().reshape(S * 32, 128), vk._w["q_norm"])
    print(f"q_norm err: {float((q_vk2.reshape(S,32,128) - q_c).abs().max()):.3e}")

    k_vk = v_lin(h_vk, vk._w["k"], vk._zero_kvd, 1024).view(S, 8, 128)
    k_vk2 = v_rms(k_vk.cpu().reshape(S * 8, 128), vk._w["k_norm"])
    print(f"k_norm err: {float((k_vk2.reshape(S,8,128) - k_c).abs().max()):.3e}")

    v_vk = v_lin(h_vk, vk._w["v"], vk._zero_kvd, 1024).view(S, 8, 128)
    print(f"v_proj err: {float((v_vk - v_c).abs().max()):.3e}")

    # ---- attention: CPU
    qh = q_c.transpose(0, 1)   # [32, S, 128]
    kh = k_c.transpose(0, 1)   # [8, S, 128]
    vh = v_c.transpose(0, 1)
    scale = 1.0 / (128 ** 0.5)
    attn = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    attn = attn + mask
    probs = torch.softmax(attn, dim=-1)
    out_cpu = torch.matmul(probs, vh)          # [32, S, 128]
    out_cpu = out_cpu.transpose(0, 1).reshape(S, 4096)

    print(f"\nCPU attn probs rowsum: {probs.sum(-1).min():.6f}..{probs.sum(-1).max():.6f}")
    print(f"CPU attn out norm    : {float(out_cpu.norm()):.4f}")

    # ---- attention: Vulkan
    q_in = q_vk2.reshape(S, 32, 128).transpose(0, 1).contiguous()
    mask_buf = _Buf(torch.triu(torch.full((S, S), float("-inf")), diagonal=1))
    probs_b = _Buf.out((32, S, S))
    ob = _Buf.out((32, S, 128))
    kb = _Buf(k_c.transpose(0, 1).contiguous())
    vb = _Buf(v_c.transpose(0, 1).contiguous())
    tvk.attention2(_Buf(q_in).buf, kb.buf, vb.buf, mask_buf.buf, probs_b.buf, ob.buf,
                   S, S, 32, 8, 128, scale)
    got_p = probs_b.cpu()
    got_o = ob.cpu()
    print(f"VK attn probs rowsum : {got_p.sum(-1).min():.6f}..{got_p.sum(-1).max():.6f}")
    print(f"VK attn probs err    : {float((got_p - probs).abs().max()):.3e}")
    print(f"VK attn out norm     : {float(got_o.norm()):.4f}")
    print(f"VK attn out err      : {float((got_o - out_cpu.transpose(0,1)).abs().max()):.3e}")


if __name__ == "__main__":
    main()
