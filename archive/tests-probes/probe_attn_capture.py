"""Instrument the layer's attention output at S=20 (the break point).

S<=16 is correct, S>=20 is broken. Every kernel verifies exact in isolation, so
capture what the layer ACTUALLY feeds to attention2 and compare that against a
reference built from the same tensors -- no hand-rolled intermediates.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer, _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    # Capture the layer's attention inputs by monkey-patching attention2.
    captured = {}
    orig = tvk.attention2

    def spy(q, k_cache, v_cache, mask, probs, out, *a, **kw):
        captured['q'] = q.download((NH, a[1], HD)) if False else None
        captured['args'] = a
        captured['qsize'] = q.size
        captured['ksize'] = k_cache.size
        captured['vsize'] = v_cache.size
        captured['msize'] = mask.size
        captured['psize'] = probs.size
        captured['osize'] = out.size
        return orig(q, k_cache, v_cache, mask, probs, out, *a, **kw)

    tvk.attention2 = spy

    ids = torch.randint(0, 1000, (1, 600))
    for S in (16, 20):
        emb = qm.embed_tokens(ids[:, :S])
        pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
        vk = VulkanDecoderLayer(qm.config, 0)
        vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
        captured.clear()
        with torch.no_grad():
            vk(emb, position_embeddings=pe, attention_mask=None,
               past_key_values=None, cache_position=torch.arange(S))
        a = captured['args']
        print(f"S={S}: seq_start={a[0]} seq_len={a[1]} n_heads={a[2]} "
              f"heads_per_kv={a[3]} head_dim={a[4]} cap={a[5]}")
        print(f"   sizes: q={captured['qsize']} k={captured['ksize']} "
              f"v={captured['vsize']} mask={captured['msize']} "
              f"probs={captured['psize']} out={captured['osize']}")
        print(f"   expected: q={NH*S*HD*4} k={NKV*S*HD*4} v={NKV*S*HD*4} "
              f"mask={NH*S*S*4} probs={NH*S*S*4} out={NH*S*HD*4}")
        print()

    tvk.attention2 = orig


if __name__ == "__main__":
    main()
