"""Capture the K the Vulkan layer writes to the cache vs the K it attends with.

Both tensors are produced in the SAME forward, so if they differ the corruption
is inside the layer. If they agree, the corruption is in the cache write itself.

This patches `DynamicCache.update` to record exactly what each layer hands it,
and separately records the k_c the layer passes to attention2.
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
NH, NKV, HD = 32, 8, 128


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    cp = torch.arange(S)

    # ---- record what each layer hands to the cache ------------------------
    written = {}
    orig_update = DynamicCache.update

    def spy_update(self, key_states, value_states, layer_idx, *a, **kw):
        written[layer_idx] = (key_states.detach().clone(),
                              value_states.detach().clone())
        return orig_update(self, key_states, value_states, layer_idx, *a, **kw)

    DynamicCache.update = spy_update

    # ---- record what the vulkan layer passes to attention2 ---------------
    attn_k = {}
    orig_attn = tvk.attention2

    def spy_attn(q, k, v, mask, probs, out, *a, **kw):
        # k is [n_kv, cap, hd] as the kernel sees it
        attn_k['k'] = k.download((NKV, a[5], HD))
        attn_k['args'] = a
        return orig_attn(q, k, v, mask, probs, out, *a, **kw)

    vk = VulkanDecoderLayer(qm.config, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)

    tvk.attention2 = spy_attn
    try:
        vc = DynamicCache()
        with torch.no_grad():
            vk(emb, position_embeddings=pe, attention_mask=None,
               past_key_values=vc, use_cache=True, cache_position=cp)
    finally:
        tvk.attention2 = orig_attn
        DynamicCache.update = orig_update

    print(f"S={S}")
    print(f"layers that wrote to the cache: {sorted(written.keys())}")

    if 0 in written:
        wk, wv = written[0]
        print(f"\ncache write  K: {tuple(wk.shape)}  dtype {wk.dtype}")
        print(f"cache write  V: {tuple(wv.shape)}  dtype {wv.dtype}")
        print(f"attn input   K: {tuple(attn_k['k'].shape)}")

        # the cache write is [1, n_kv, S, hd]; attn input is [n_kv, cap, hd]
        wk_flat = wk[0]                     # [n_kv, S, hd]
        ak = attn_k['k']
        n = min(wk_flat.shape[1], ak.shape[1])
        d = float((wk_flat[:, :n, :].float() - ak[:, :n, :].float()).abs().max())
        print(f"\nK written-to-cache vs K used-in-attention (first {n} slots): {d:.4e}")

        # and what actually landed in the cache
        ck = list(vc)[0][0][0]
        d2 = float((wk_flat[:, :n, :].float() - ck[:, :n, :].float()).abs().max())
        print(f"K written vs K now in cache:                              {d2:.4e}")
    else:
        print("layer 0 did not write to the cache")


if __name__ == "__main__":
    main()
