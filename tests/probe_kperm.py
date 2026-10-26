"""Is the cached K a head permutation, or an offset shift?

Single-layer cache: V exact (9.8e-04), K wrong (1.0) with the same RMS and a
uniform per-token error. That pattern suggests the K heads are written in a
different order, or from a different offset in the packed buffer.

Compares every stock KV head against every Vulkan KV head to detect a
permutation, and also checks the raw (pre-permute) layout.
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
NKV, HD = 8, 128


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    cp = torch.arange(S)

    sc = DynamicCache()
    with torch.no_grad():
        qm.layers[0](emb, position_embeddings=pe, attention_mask=None,
                     past_key_values=sc, use_cache=True, cache_position=cp)
    sk = list(sc)[0][0][0]        # [n_kv, S, hd]

    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vc = DynamicCache()
    with torch.no_grad():
        vk(emb, position_embeddings=pe, attention_mask=None,
           past_key_values=vc, use_cache=True, cache_position=cp)
    vk_ = list(vc)[0][0][0]       # [n_kv, S, hd]

    print(f"S={S}  K shape {tuple(sk.shape)}")

    # ---- permutation test -------------------------------------------------
    print("\nbest-matching vk head for each stock head (err):")
    perm = []
    for i in range(NKV):
        errs = [float((sk[i] - vk_[j]).abs().max()) for j in range(NKV)]
        best = min(range(NKV), key=lambda j: errs[j])
        perm.append(best)
        print(f"  stock head {i} -> vk head {best}  (err {errs[best]:.4e}, "
              f"identity err {errs[i]:.4e})")
    print(f"\npermutation: {perm}")
    print(f"is identity: {perm == list(range(NKV))}")

    # ---- offset test: is it a token shift? --------------------------------
    print("\ntoken-shift test (stock K[t] vs vk K[t+d]):")
    for d in (-2, -1, 0, 1, 2):
        if d == 0:
            continue
        lo, hi = max(0, -d), min(S, S - d)
        if hi <= lo:
            continue
        e = float((sk[:, lo:hi, :] - vk_[:, lo + d:hi + d, :]).abs().max())
        print(f"  shift {d:+d}: err {e:.4e}")


if __name__ == "__main__":
    main()
