"""Test the Vulkan backbone's INCREMENTAL decoding path with a KV cache.

The D3 comparison was a single prefill and passed (cosine 0.999918). The engine
produces garbage, and the engine uses a persistent DynamicCache with
`use_cache=True` and one-token-at-a-time stepping. This isolates that path.

Compares, at each step:
  * the logits from a Vulkan-backed backbone against the stock backbone
  * the hidden state
  * the KV cache contents
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


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone          # Qwen3ForCausalLM -- has logits

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    print(f"prompt tokens: {ids.shape[1]}")

    # ---- stock backbone, prefill + incremental ----------------------------
    with torch.no_grad():
        out = lm(input_ids=ids, use_cache=True)
    stock_cache = out.past_key_values
    stock_logits = out.logits[:, -1, :]
    print(f"stock prefill cache len: {stock_cache.get_seq_length()}")
    print(f"stock prefill top1: {tok.decode([int(stock_logits.argmax())])!r}")

    # next token, incremental
    nxt = stock_logits.argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        out2 = lm(input_ids=nxt, past_key_values=stock_cache, use_cache=True)
    stock_logits2 = out2.logits[:, -1, :]
    print(f"stock step-2 cache len: {out2.past_key_values.get_seq_length()}")
    print(f"stock step-2 top1: {tok.decode([int(stock_logits2.argmax())])!r}")

    # ---- Vulkan backbone, same sequence ----------------------------------
    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))
    with torch.no_grad():
        vout = bb(input_ids=ids, use_cache=True)
    vk_cache = vout.past_key_values
    # HybridBackbone is the inner model: apply lm_head to get logits.
    vk_logits = lm.lm_head(vout.last_hidden_state[:, -1, :])
    print(f"\nvk prefill cache len: {vk_cache.get_seq_length()}")
    print(f"vk prefill top1: {tok.decode([int(vk_logits.argmax())])!r}")

    c = float(torch.nn.functional.cosine_similarity(
        stock_logits.float().flatten(), vk_logits.float().flatten(), dim=0))
    print(f"prefill logits cosine: {c:.6f}")
    print(f"prefill logits maxdiff: {float((stock_logits.float()-vk_logits.float()).abs().max()):.3e}")

    # incremental step on the Vulkan side
    vnxt = vk_logits.argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        vout2 = bb(input_ids=vnxt, past_key_values=vk_cache, use_cache=True)
    vk_logits2 = lm.lm_head(vout2.last_hidden_state[:, -1, :])
    print(f"\nvk step-2 cache len: {vout2.past_key_values.get_seq_length()}")
    print(f"vk step-2 top1: {tok.decode([int(vk_logits2.argmax())])!r}")

    c2 = float(torch.nn.functional.cosine_similarity(
        stock_logits2.float().flatten(), vk_logits2.float().flatten(), dim=0))
    print(f"step-2 logits cosine: {c2:.6f}")
    print(f"step-2 logits maxdiff: {float((stock_logits2.float()-vk_logits2.float()).abs().max()):.3e}")

    # ---- compare the caches directly -------------------------------------
    sk = list(stock_cache)
    vk = list(vk_cache)
    print(f"\ncache layers: stock={len(sk)} vk={len(vk)}")
    if len(sk) == len(vk):
        for i in range(len(sk)):
            skey, sval = sk[i][0], sk[i][1]
            vkey, vval = vk[i][0], vk[i][1]
            kd = float((skey.float() - vkey.float()).abs().max())
            vd = float((sval.float() - vval.float()).abs().max())
            print(f"  layer {i:2d}: k err {kd:.3e}  v err {vd:.3e}"
                  f"{'   <-- DIVERGES' if kd > 0.05 or vd > 0.05 else ''}")


if __name__ == "__main__":
    main()
