"""Is the K stored in the cache correctly RoPE'd?

Prefill logits are correct (cosine 0.999964) but the cached K is wrong (err 1.0
at layer 0) while cached V is right (err 9.8e-04). V does not go through RoPE;
K does. So the K rotation is the suspect.

Compare the layer's cached K against a reference computed by applying the stock
`apply_rotary_pos_emb` to the same pre-norm K.
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


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))

    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    k_raw = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD),
                w["self_attn.k_norm.weight"])

    # stock RoPE on K: [B, n_kv, S, hd]
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    k4 = k_raw.transpose(0, 1).unsqueeze(0)          # [1, n_kv, S, hd]
    _, k_roped = apply_rotary_pos_emb(torch.zeros_like(k4), k4, pe[0], pe[1])
    k_ref = k_roped[0].transpose(0, 1)               # [S, n_kv, hd]

    # Vulkan: pack Q|K|V and run rope_qkv
    q_raw = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD),
                w["self_attn.q_norm.weight"])
    v_raw = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    qkv = torch.cat([q_raw.reshape(-1), k_raw.reshape(-1), v_raw.reshape(-1)])
    qb = _Buf(qkv.clone())
    tvk.rope_qkv(qb.buf, 0, NH, NKV, HD, n_tok=S)
    got = qb.cpu()
    k_vk = got[S * NH * HD:S * NH * HD + S * NKV * HD].view(S, NKV, HD)

    print(f"S={S}")
    print(f"K err (vulkan vs stock rope): {float((k_vk - k_ref).abs().max()):.3e}")
    print(f"K rms ref={float(k_ref.pow(2).mean().sqrt()):.4f} "
          f"vk={float(k_vk.pow(2).mean().sqrt()):.4f}")

    # per-token error, to see which positions are wrong
    per_tok = (k_vk - k_ref).abs().amax(dim=(1, 2))
    print(f"per-token K err: {[round(float(x),4) for x in per_tok[:6]]}")
    print(f"                 ... {[round(float(x),4) for x in per_tok[-4:]]}")


if __name__ == "__main__":
    main()
