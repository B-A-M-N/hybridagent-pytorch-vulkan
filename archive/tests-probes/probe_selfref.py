"""Compare the layer's output against a reference built from ITS OWN inputs.

Every stage verifies exact in isolation, so build a CPU reference from the same
hidden_states and position_embeddings the layer received, then compare. This
removes any question of whether the inputs differ.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer, _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import SYSTEM_PROMPT, load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "How many trailing zeros does 100! have?"}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    print(f"S={S}")

    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))

    # ---- CPU reference, same inputs, fp32 throughout
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h0 = emb.float()
    h = rms(h0, w["input_layernorm.weight"])
    q = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD),
            w["self_attn.q_norm.weight"])
    k = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD),
            w["self_attn.k_norm.weight"])
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)

    # RoPE (interleaved, rotate-half) -- same as rope_qkv.comp
    qkv = torch.cat([q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1)], dim=-1)
    nhalf = HD // 2
    inv = 1000000.0 ** (-2.0 * torch.arange(nhalf).float() / HD)
    for t in range(S):
        ang = float(t) * inv
        c, s = torch.cos(ang), torch.sin(ang)
        # Q heads are packed per token at t*NH*HD; K heads at t*(NH+KVD+KVD) + NH*HD
        for hh in range(NH + NKV):
            base = ((0 if hh < NH else NH * HD)
                    + (hh if hh < NH else hh - NH) * HD)
            x0 = qkv[t, base:base + nhalf].clone()
            x1 = qkv[t, base + nhalf:base + HD].clone()
            qkv[t, base:base + nhalf] = x0 * c - x1 * s
            qkv[t, base + nhalf:base + HD] = x1 * c + x0 * s
    q_r = qkv[:, :NH * HD].reshape(S, NH, HD)
    k_r = qkv[:, NH * HD:NH * HD + NKV * HD].reshape(S, NKV, HD)
    v_r = qkv[:, NH * HD + NKV * HD:].reshape(S, NKV, HD)

    qh = q_r.transpose(0, 1)
    kh = k_r.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    vh = v_r.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, NH, S, S).masked_fill(~allow, float("-inf"))
    attn_w = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * SCALING + mask,
                           dim=-1)
    attn_o = torch.matmul(attn_w, vh).transpose(0, 1).reshape(S, NH * HD)
    o_out = attn_o @ w["self_attn.o_proj.weight"].t()
    res1 = h0 + o_out
    h2 = rms(res1, w["post_attention_layernorm.weight"])
    gu = torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].t()) * \
        (h2 @ w["mlp.up_proj.weight"].t())
    res2 = res1 + (gu @ w["mlp.down_proj.weight"].t())

    # ---- Vulkan layer, same inputs
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    with torch.no_grad():
        got = vk(emb, position_embeddings=pe, attention_mask=None,
                 past_key_values=None, cache_position=torch.arange(S))

    print(f"\nCPU ref norm : {float(res2.norm()):.4f}")
    print(f"VK    norm   : {float(got.float().norm()):.4f}")
    print(f"cos          : {float(torch.nn.functional.cosine_similarity(res2.flatten(), got.float().flatten(), dim=0)):.6f}")
    print(f"maxdiff      : {float((res2 - got.float()).abs().max()):.4f}")

    # per-token
    d = (res2 - got.float()).abs().amax(dim=-1).reshape(-1)
    print(f"per-token: first3={[round(float(x),4) for x in d[:3]]} "
          f"last3={[round(float(x),4) for x in d[-3:]]}")
    print(f"  argmax token: {int(d.argmax())}")


if __name__ == "__main__":
    main()
