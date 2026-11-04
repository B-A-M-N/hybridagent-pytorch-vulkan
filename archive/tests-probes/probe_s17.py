"""Compare the layer's attention output at S=16 (correct) vs S=17 (broken).

The layer is deterministic and breaks exactly at S=17. Capture what the layer
feeds to attention2 and what it gets back, and compare both against a reference
built from the same captured tensors. This isolates whether the break is in the
attention call or in an earlier stage whose error only shows up at S=17.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 64))

    cap = {}
    orig = tvk.attention2

    def spy(q, k, v, mask, probs, out, *a, **kw):
        cap['q'] = q.download((NH, a[1], HD))
        cap['k'] = k.download((NKV, a[5], HD))
        cap['v'] = v.download((NKV, a[5], HD))
        cap['mask'] = mask.download((NH, a[1], a[5]))
        cap['args'] = a
        r = orig(q, k, v, mask, probs, out, *a, **kw)
        cap['out'] = out.download((NH, a[1], HD))
        return r

    tvk.attention2 = spy

    for S in (16, 17):
        emb = qm.embed_tokens(ids[:, :S])
        pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
        vk = VulkanDecoderLayer(qm.config, 0)
        vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
        cap.clear()
        with torch.no_grad():
            vk(emb, position_embeddings=pe, attention_mask=None,
               past_key_values=None, cache_position=torch.arange(S))

        q, k, v, m = cap['q'], cap['k'], cap['v'], cap['mask']
        kh = k.repeat_interleave(NH // NKV, dim=0)
        vh = v.repeat_interleave(NH // NKV, dim=0)
        ref_w = torch.softmax(
            torch.matmul(q, kh.transpose(-1, -2)) * SCALING + m, dim=-1)
        ref_o = torch.matmul(ref_w, vh)
        we = float((cap['out'] - ref_o).abs().max())
        # is the captured input itself sane?
        print(f"S={S}: q_norm={float(q.norm()):.3f} k_norm={float(k.norm()):.3f} "
              f"v_norm={float(v.norm()):.3f}")
        print(f"   mask[0,0,:3]={[round(float(x),3) for x in m[0,0,:3]]}")
        print(f"   mask[0,-1,:3]={[round(float(x),3) for x in m[0,-1,:3]]}")
        print(f"   attn out err vs same-source ref: {we:.3e}")
        print()

    tvk.attention2 = orig


if __name__ == "__main__":
    main()
