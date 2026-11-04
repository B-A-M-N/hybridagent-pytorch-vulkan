"""
PERMANENT CONTROL: zero-Vulkan-layer execution through the PyTorch wrapper.

This is the control that located the causal-mask bug. When the Vulkan engine
produced garbage, running `HybridBackbone` with **zero** GPU layers through the
same wrapper *also* diverged -- which proved the defect was in the wrapper's
orchestration rather than in any Vulkan kernel.

It is kept as a permanent regression test for exactly that reason: it isolates
"the wrapper" from "the kernels".

Three configurations are compared on identical inputs and seeds:
  A. stock        -- the unmodified Qwen3ForCausalLM
  B. wrapped-cpu  -- HybridBackbone with 0 Vulkan layers (pure CPU, same wrapper)
  C. wrapped-vk   -- HybridBackbone with N Vulkan layers

A and B must agree exactly. C must agree to bf16 tolerance.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
N_STEPS = int(os.environ.get("TVK_CONTROL_STEPS", "8"))
N_GPU = int(os.environ.get("TVK_GPU_LAYERS", "4"))


def greedy(model_or_bb, lm_head, ids, tok, n):
    """Greedy decode; returns the list of token ids.

    `model_or_bb` must accept (input_ids=, past_key_values=, use_cache=) and
    return something with `.logits` (stock) or `.last_hidden_state` (wrapper).
    """
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        out = model_or_bb(input_ids=ids, past_key_values=cache, use_cache=True)
    if hasattr(out, "logits"):
        cur = out.logits[:, -1, :]
    else:
        cur = lm_head(out.last_hidden_state[:, -1, :])
    toks = []
    for _ in range(n):
        nxt = cur.argmax(dim=-1, keepdim=True)
        toks.append(int(nxt[0, 0]))
        with torch.no_grad():
            out = model_or_bb(input_ids=nxt, past_key_values=cache,
                              use_cache=True)
        if hasattr(out, "logits"):
            cur = out.logits[:, -1, :]
        else:
            cur = lm_head(out.last_hidden_state[:, -1, :])
    return toks


def main():
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]

    # A. stock
    stock = greedy(lm, None, ids, tok, N_STEPS)

    # B. wrapped, zero Vulkan layers
    bb0 = HybridBackbone(qm, Placement.first_n(0, len(qm.layers)))
    wrapped_cpu = greedy(bb0, lm.lm_head, ids, tok, N_STEPS)

    # C. wrapped, N Vulkan layers
    bb_vk = HybridBackbone(qm, Placement.first_n(N_GPU, len(qm.layers)))
    wrapped_vk = greedy(bb_vk, lm.lm_head, ids, tok, N_STEPS)

    print(f"steps={N_STEPS}  gpu_layers={N_GPU}")
    print(f"stock        : {tok.decode(stock)!r}")
    print(f"wrapped-cpu  : {tok.decode(wrapped_cpu)!r}")
    print(f"wrapped-vk   : {tok.decode(wrapped_vk)!r}")
    print()
    print(f"A == B (stock vs wrapped-cpu)  : {stock == wrapped_cpu}")
    print(f"A == C (stock vs wrapped-vk)   : {stock == wrapped_vk}")

    ok = (stock == wrapped_cpu) and (stock == wrapped_vk)
    print("\nCONTROL PASS" if ok else "\nCONTROL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
