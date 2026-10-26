"""Single-layer KV-cache comparison: stock vs Vulkan, identical inputs.

The engine produces garbage because incremental decoding diverges while prefill
is correct. This isolates the cache write to ONE layer so the divergence cannot
compound across layers.

Compares, for the same hidden_states and position_embeddings:
  * the layer output
  * the K written into the cache
  * the V written into the cache
and reports the first quantity that differs.
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

    print(f"S={S}")

    # ---- stock layer ------------------------------------------------------
    stock = qm.layers[0]
    sc = DynamicCache()
    with torch.no_grad():
        o_stock = stock(emb, position_embeddings=pe, attention_mask=None,
                        past_key_values=sc, use_cache=True, cache_position=cp)
    h_stock = o_stock[0] if isinstance(o_stock, tuple) else o_stock
    sk = list(sc)
    print(f"stock cache: layers={len(sk)} K{tuple(sk[0][0].shape)} "
          f"V{tuple(sk[0][1].shape)}")

    # ---- vulkan layer -----------------------------------------------------
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vc = DynamicCache()
    with torch.no_grad():
        o_vk = vk(emb, position_embeddings=pe, attention_mask=None,
                  past_key_values=vc, use_cache=True, cache_position=cp)
    h_vk = o_vk[0] if isinstance(o_vk, tuple) else o_vk
    vk_ = list(vc)
    print(f"vk    cache: layers={len(vk_)} K{tuple(vk_[0][0].shape)} "
          f"V{tuple(vk_[0][1].shape)}")

    # ---- compare ----------------------------------------------------------
    print(f"\noutput  err: {float((h_stock.float()-h_vk.float()).abs().max()):.4e}")
    if sk[0][0].shape == vk_[0][0].shape:
        kd = float((sk[0][0].float() - vk_[0][0].float()).abs().max())
        vd = float((sk[0][1].float() - vk_[0][1].float()).abs().max())
        print(f"cache K err: {kd:.4e}")
        print(f"cache V err: {vd:.4e}")
        print(f"cache K rms: stock={float(sk[0][0].float().pow(2).mean().sqrt()):.4f} "
              f"vk={float(vk_[0][0].float().pow(2).mean().sqrt()):.4f}")
        print(f"cache V rms: stock={float(sk[0][1].float().pow(2).mean().sqrt()):.4f} "
              f"vk={float(vk_[0][1].float().pow(2).mean().sqrt()):.4f}")
        # per-token K error
        kt = (sk[0][0].float() - vk_[0][0].float()).abs().amax(dim=(0, 1, 3))
        print(f"cache K err per token: {[round(float(x),4) for x in kt[:8]]}")
    else:
        print("CACHE SHAPE MISMATCH -- this is the defect")


if __name__ == "__main__":
    main()
