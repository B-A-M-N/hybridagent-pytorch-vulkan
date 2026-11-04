"""Compare the K/V the Vulkan layer WRITES at step 1 against the stock layer's.

Attention itself is healthy (rowsum 1.0 on every call). Greedy decode diverges
at step 1 and then the Vulkan output repeats, which means the cache being
attended over is wrong.

Capture the exact K/V tensors each layer writes into the cache on step 1 and
compare them against an independent fp32 reference computed from the same
hidden state.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import HybridBackbone, Placement  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
NKV, HD = 8, 128


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]

    # ---- stock: prefill then step 1 --------------------------------------
    sc = DynamicCache()
    with torch.no_grad():
        o = lm(input_ids=ids, past_key_values=sc, use_cache=True)
    nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        o2 = lm(input_ids=nxt, past_key_values=sc, use_cache=True)
    sk = list(sc)[0]
    stock_k = sk[0][0]        # [n_kv, L, hd]  (L = 13)
    stock_v = sk[1][0]

    # ---- vulkan: same -----------------------------------------------------
    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))
    vc = DynamicCache()
    with torch.no_grad():
        v = bb(input_ids=ids, past_key_values=vc, use_cache=True)
    vn = lm.lm_head(v.last_hidden_state[:, -1, :]).argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        v2 = bb(input_ids=vn, past_key_values=vc, use_cache=True)
    vk_ = list(vc)[0]
    vk_k = vk_[0][0]
    vk_v = vk_[1][0]

    print(f"stock K {tuple(stock_k.shape)}  vk K {tuple(vk_k.shape)}")

    # ---- compare the NEW slot only (index -1) ----------------------------
    sk_new = stock_k[:, -1, :]
    vk_new = vk_k[:, -1, :]
    print(f"\nNEW slot K err: {float((sk_new.float()-vk_new.float()).abs().max()):.4e}")
    print(f"NEW slot V err: {float((stock_v[:,-1,:].float()-vk_v[:,-1,:].float()).abs().max()):.4e}")

    # ---- and the WHOLE cache ---------------------------------------------
    print(f"whole K err: {float((stock_k.float()-vk_k.float()).abs().max()):.4e}")
    print(f"whole V err: {float((stock_v.float()-vk_v.float()).abs().max()):.4e}")

    # ---- per-token K error across the cache ------------------------------
    kt = (stock_k.float() - vk_k.float()).abs().amax(dim=(0, 2))
    print(f"\nK err per cache position: {[round(float(x),3) for x in kt]}")

    # ---- is the new slot's RoPE angle right? -----------------------------
    # reference: recompute K for the new token from the same hidden state
    print(f"\nstock new-slot K rms: {float(stock_k[:,-1,:].float().pow(2).mean().sqrt()):.4f}")
    print(f"vk    new-slot K rms: {float(vk_k[:,-1,:].float().pow(2).mean().sqrt()):.4f}")
    print(f"stock new-slot V rms: {float(stock_v[:,-1,:].float().pow(2).mean().sqrt()):.4f}")
    print(f"vk    new-slot V rms: {float(vk_v[:,-1,:].float().pow(2).mean().sqrt()):.4f}")


if __name__ == "__main__":
    main()
