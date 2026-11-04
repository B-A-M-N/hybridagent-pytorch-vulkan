"""Quick check: one Vulkan layer vs one CPU layer, same weights, same input."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    print(f"tokens: {ids.shape[1]}")

    # CPU reference: one real Qwen3DecoderLayer
    cpu_layer = qm.layers[0]
    with torch.no_grad():
        h_cpu = cpu_layer(
            qm.embed_tokens(ids),
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=torch.arange(ids.shape[1]),
            position_embeddings=qm.rotary_emb(qm.embed_tokens(ids), torch.arange(ids.shape[1]).unsqueeze(0)),
        )

    # Vulkan: same layer through the wrapper
    vk_layer = VulkanDecoderLayer(cfg, 0)
    vk_layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    with torch.no_grad():
        h_vk = vk_layer(
            qm.embed_tokens(ids),
            position_embeddings=qm.rotary_emb(qm.embed_tokens(ids), torch.arange(ids.shape[1]).unsqueeze(0)),
            attention_mask=None,
            past_key_values=None,
            cache_position=torch.arange(ids.shape[1]),
        )

    err = float((h_vk.float() - h_cpu.float()).abs().max())
    print(f"layer 0 hidden err: {err:.3e}")
    print(f"  cpu  {tuple(h_cpu.shape)}")
    print(f"  vk   {tuple(h_vk.shape)}")
    print("PASS" if err < 0.5 else "FAIL")
    return 0 if err < 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
