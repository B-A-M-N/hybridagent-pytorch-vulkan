"""Check whether the layer's V projection is correct at S=16 vs S=17.

The captured V has RMS ~0.03 vs Q's ~1.5, a 50x discrepancy. Compare the layer's
actual V against a direct matmul from the same hidden state.
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


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 64))
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    for S in (16, 17):
        emb = qm.embed_tokens(ids[:, :S]).reshape(S, -1)
        h = rms(emb, w["input_layernorm.weight"])

        # reference V
        ref_v = h @ w["self_attn.v_proj.weight"].t()
        ref_k = h @ w["self_attn.k_proj.weight"].t()
        ref_q = h @ w["self_attn.q_proj.weight"].t()

        # layer's V via the same helper
        vk = VulkanDecoderLayer(cfg, 0)
        vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
        vk.upload_weights()

        hv = vk._rms_b(_Buf(h.reshape(S, -1)), vk._w["input_ln"])
        got_v = vk._lin_b(hv, vk._w["v"], vk._zero_kvd, NKV * HD).cpu()
        got_k = vk._lin_b(hv, vk._w["k"], vk._zero_kvd, NKV * HD).cpu()
        got_q = vk._lin_b(hv, vk._w["q"], vk._zero_hd, NH * HD).cpu()

        print(f"S={S}:")
        print(f"   ref q rms={float(ref_q.pow(2).mean().sqrt()):.4f}  "
              f"vk q rms={float(got_q.pow(2).mean().sqrt()):.4f}")
        print(f"   ref k rms={float(ref_k.pow(2).mean().sqrt()):.4f}  "
              f"vk k rms={float(got_k.pow(2).mean().sqrt()):.4f}")
        print(f"   ref v rms={float(ref_v.pow(2).mean().sqrt()):.4f}  "
              f"vk v rms={float(got_v.pow(2).mean().sqrt()):.4f}")
        print(f"   v err={float((got_v.reshape(-1) - ref_v.reshape(-1)).abs().max()):.3e}")
        print()


if __name__ == "__main__":
    main()
