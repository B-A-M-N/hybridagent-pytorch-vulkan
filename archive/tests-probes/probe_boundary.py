"""Boundary sweep S=14..26 using PyTorch forward hooks on the STOCK layer.

No hand-rolled intermediates: hooks capture the real module outputs from the
stock Qwen3DecoderLayer, so the reference is the library's own computation.

Hypothesis under test: the Vulkan output norm appears to cap near S=16, which
suggests the layer's flat buffers are being truncated or read with a fixed
length.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 64))

    print(f"{'S':>4} {'norm_ref':>10} {'norm_vk':>10} {'ratio':>8} {'tok0':>9} {'toklast':>9}")
    for S in (14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26):
        ids_s = ids[:, :S]
        emb = qm.embed_tokens(ids_s)
        pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))

        with torch.no_grad():
            o = qm.layers[0](emb, position_embeddings=pe, attention_mask=None,
                             past_key_values=None, use_cache=False,
                             cache_position=torch.arange(S))
        h_ref = o[0] if isinstance(o, tuple) else o

        vk = VulkanDecoderLayer(qm.config, 0)
        vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
        with torch.no_grad():
            h_vk = vk(emb, position_embeddings=pe, attention_mask=None,
                      past_key_values=None, cache_position=torch.arange(S))

        nr = float(h_ref.float().norm())
        nv = float(h_vk.float().norm())
        d = (h_ref.float() - h_vk.float()).abs().amax(dim=-1).reshape(-1)
        print(f"{S:>4} {nr:>10.2f} {nv:>10.2f} {nv / nr:>8.4f} "
              f"{float(d[0]):>9.4f} {float(d[-1]):>9.4f}")


if __name__ == "__main__":
    main()
