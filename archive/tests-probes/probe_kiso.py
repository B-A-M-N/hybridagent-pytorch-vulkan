"""Isolate the K path inside the Vulkan layer vs the stock layer.

V is exact, K is wrong, layout is identical (no permutation, no shift). So the
difference is in what K the layer computes before caching.

The stock layer does:
    key_states = k_norm(k_proj(hidden).view(shape)).transpose(1,2)
    query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)
    cache.update(key_states, value_states)

The Vulkan layer does the same mathematically, but via kernels. Capture the
Vulkan layer's K immediately before the cache write and compare it against the
stock layer's key_states at the same point.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))

    # ---- capture the stock layer's internal key_states via a hook ----------
    captured = {}

    def hook(module, args, kwargs=None):
        return None

    # Monkey-patch the stock attention to record its key_states.
    import transformers.models.qwen3.modeling_qwen3 as m
    orig_apply = m.apply_rotary_pos_emb

    def spy(q, k, cos, sin):
        captured['q'] = q.detach().clone()
        captured['k'] = k.detach().clone()
        return orig_apply(q, k, cos, sin)

    m.apply_rotary_pos_emb = spy
    try:
        from transformers.cache_utils import DynamicCache
        sc = DynamicCache()
        with torch.no_grad():
            qm.layers[0](emb, position_embeddings=pe, attention_mask=None,
                         past_key_values=sc, use_cache=True,
                         cache_position=torch.arange(S))
    finally:
        m.apply_rotary_pos_emb = orig_apply

    k_stock = captured['k']            # [B, n_kv, S, hd], POST-rope
    print(f"captured stock K: {tuple(k_stock.shape)}  "
          f"rms={float(k_stock.float().pow(2).mean().sqrt()):.4f}")

    # ---- what the Vulkan layer caches -------------------------------------
    vk = VulkanDecoderLayer(qm.config, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vc = DynamicCache()
    with torch.no_grad():
        vk(emb, position_embeddings=pe, attention_mask=None,
           past_key_values=vc, use_cache=True, cache_position=torch.arange(S))
    k_vk = list(vc)[0][0][0]           # [n_kv, S, hd]

    ks = k_stock[0]                    # [n_kv, S, hd]
    print(f"vk cached K:    {tuple(k_vk.shape)}  "
          f"rms={float(k_vk.float().pow(2).mean().sqrt()):.4f}")
    print(f"\nK err (stock post-rope vs vk cached): "
          f"{float((ks.float() - k_vk.float()).abs().max()):.4e}")

    # ---- also compare the PRE-rope projection ------------------------------
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    k_raw = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, 8, 128),
                w["self_attn.k_norm.weight"])
    k_raw4 = k_raw.transpose(0, 1).unsqueeze(0)
    _, k_roped = orig_apply(torch.zeros_like(k_raw4), k_raw4, pe[0], pe[1])
    k_ref = k_roped[0]
    print(f"\nreference K (manual): {tuple(k_ref.shape)}  "
          f"rms={float(k_ref.float().pow(2).mean().sqrt()):.4f}")
    print(f"stock captured vs manual ref: "
          f"{float((ks.float() - k_ref.float()).abs().max()):.4e}")
    print(f"vk cached    vs manual ref: "
          f"{float((k_vk.float() - k_ref.float()).abs().max()):.4e}")


if __name__ == "__main__":
    main()
