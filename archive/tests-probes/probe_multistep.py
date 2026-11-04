"""Multi-step greedy decode: stock vs Vulkan backbone, token by token.

A single step is close (cosine 0.87) but the engine produces garbage. Greedy
decode compounds: if each step's argmax occasionally differs, the trajectories
diverge completely.

This runs N steps of greedy decoding on both backbones and reports:
  * per-step top-1 agreement
  * per-step cosine
  * the first step where the top-1 token differs
  * the cache divergence at that step
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
N_STEPS = 12


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    print(f"prompt: {ids.shape[1]} tokens, {N_STEPS} greedy steps")

    # ---- stock ------------------------------------------------------------
    sc = DynamicCache()
    with torch.no_grad():
        out = lm(input_ids=ids, past_key_values=sc, use_cache=True)
    stock_tokens = []
    cur = out.logits[:, -1, :]
    for _ in range(N_STEPS):
        nxt = cur.argmax(dim=-1, keepdim=True)
        stock_tokens.append(int(nxt[0, 0]))
        with torch.no_grad():
            out = lm(input_ids=nxt, past_key_values=sc, use_cache=True)
        cur = out.logits[:, -1, :]

    # ---- vulkan -----------------------------------------------------------
    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))
    vc = DynamicCache()
    with torch.no_grad():
        vout = bb(input_ids=ids, past_key_values=vc, use_cache=True)
    vk_tokens = []
    vcur = lm.lm_head(vout.last_hidden_state[:, -1, :])
    vk_logits = [vcur]
    for _ in range(N_STEPS):
        nxt = vcur.argmax(dim=-1, keepdim=True)
        vk_tokens.append(int(nxt[0, 0]))
        with torch.no_grad():
            vout = bb(input_ids=nxt, past_key_values=vc, use_cache=True)
        vcur = lm.lm_head(vout.last_hidden_state[:, -1, :])
        vk_logits.append(vcur)

    # ---- compare ----------------------------------------------------------
    print(f"\n{'step':>5} {'stock':>12} {'vulkan':>12} {'agree':>7}")
    print("-" * 40)
    first_diff = None
    for i in range(N_STEPS):
        s = tok.decode([stock_tokens[i]])
        v = tok.decode([vk_tokens[i]])
        ok = stock_tokens[i] == vk_tokens[i]
        if not ok and first_diff is None:
            first_diff = i
        print(f"{i:>5} {s!r:>12} {v!r:>12} {'yes' if ok else 'NO':>7}")

    print(f"\nfirst divergence at step: {first_diff}")
    print(f"stock  text: {tok.decode(stock_tokens)!r}")
    print(f"vulkan text: {tok.decode(vk_tokens)!r}")

    # cache divergence at the end
    sk = list(sc)
    vk_ = list(vc)
    print(f"\nfinal cache K err per layer (first 6):")
    for i in range(6):
        if sk[i][0].shape == vk_[i][0].shape:
            print(f"  layer {i}: {float((sk[i][0].float()-vk_[i][0].float()).abs().max()):.4e}")


if __name__ == "__main__":
    main()
