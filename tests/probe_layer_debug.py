"""Decisive experiment: does the vulkan layer produce a sane hidden state at
all, or is the wrapper misreading the input?

Compares (a) CPU layer output, (b) vulkan layer output, (c) CPU layer output
with cache_position forced -- and prints the norm of each so a wrong-but-finite
result is obvious.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def main():
    torch.manual_seed(0)
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import VulkanDecoderLayer

    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))

    # (a) CPU reference
    with torch.no_grad():
        h_cpu = qm.layers[0](emb, attention_mask=None, position_ids=None,
                             past_key_values=None, use_cache=False,
                             cache_position=torch.arange(S),
                             position_embeddings=pe)

    # (b) vulkan wrapper
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    with torch.no_grad():
        h_vk = vk(emb, position_embeddings=pe, attention_mask=None,
                  past_key_values=None, cache_position=torch.arange(S))

    print(f"tokens S = {S}")
    print(f"h_cpu  shape={tuple(h_cpu.shape)} norm={float(h_cpu.float().norm()):.4f}")
    print(f"h_vk   shape={tuple(h_vk.shape)} norm={float(h_vk.float().norm()):.4f}")
    print(f"abs diff max  : {float((h_vk.float()-h_cpu.float()).abs().max()):.4e}")
    print(f"cosine sim    : {float(torch.nn.functional.cosine_similarity(h_cpu.float().flatten(), h_vk.float().flatten(), dim=0)):.6f}")
    print()
    # (c) is the CPU layer itself stable across two identical calls?
    with torch.no_grad():
        h_cpu2 = qm.layers[0](emb, attention_mask=None, position_ids=None,
                              past_key_values=None, use_cache=False,
                              cache_position=torch.arange(S),
                              position_embeddings=pe)
    print(f"cpu determinism: {float((h_cpu.float()-h_cpu2.float()).abs().max()):.4e}")
    print()
    # (d) does the wrapper differ from a *fresh* CPU call with cache_position=None?
    with torch.no_grad():
        h_cpu3 = qm.layers[0](emb, attention_mask=None, position_ids=None,
                              past_key_values=None, use_cache=False,
                              cache_position=None, position_embeddings=pe)
    print(f"cpu(cp=None) vs cpu(cp=arange): {float((h_cpu3.float()-h_cpu.float()).abs().max()):.4e}")
    print(f"cpu(cp=None) vs vk            : {float((h_cpu3.float()-h_vk.float()).abs().max()):.4e}")


if __name__ == "__main__":
    main()
