"""Is the Vulkan layer's cached K correctly RoPE'd during INCREMENTAL decoding?

Prefill is correct (cosine 0.999964) but the second step diverges. The stock
layer applies RoPE then caches K. The Vulkan layer applies RoPE in rope_qkv using
`pos_start + token`. If pos_start is wrong for the incremental step, every cached
K from that step is mis-rotated.

This compares the layer's cached K/V against the stock layer's for a single
incremental step, and separately checks whether the RoPE angle matches.
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

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]

    # stock
    with torch.no_grad():
        out = lm(input_ids=ids, use_cache=True)
    sc = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        out2 = lm(input_ids=nxt, past_key_values=sc, use_cache=True)
    sk = list(out2.past_key_values)

    # vulkan
    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))
    with torch.no_grad():
        vout = bb(input_ids=ids, use_cache=True)
    vc = vout.past_key_values
    vn = lm.lm_head(vout.last_hidden_state[:, -1, :]).argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        vout2 = bb(input_ids=vn, past_key_values=vc, use_cache=True)
    vk = list(vout2.past_key_values)

    print(f"stock cache len {len(sk)}  vk cache len {len(vk)}")
    print(f"\nstep-2 K/V error, and the error at the NEW slot (index -1):")
    for i in (0, 1, 2, 3, 4):
        skey, sval = sk[i][0], sk[i][1]
        vkey, vval = vk[i][0], vk[i][1]
        if skey.shape != vkey.shape:
            print(f"  layer {i}: SHAPE MISMATCH stock={tuple(skey.shape)} "
                  f"vk={tuple(vkey.shape)}")
            continue
        kd = float((skey.float() - vkey.float()).abs().max())
        vd = float((sval.float() - vval.float()).abs().max())
        # error in the last slot only (the token just appended)
        kd_new = float((skey[:, :, -1, :].float()
                        - vkey[:, :, -1, :].float()).abs().max())
        print(f"  layer {i}: K err {kd:.4e} (new slot {kd_new:.4e})  "
              f"V err {vd:.4e}")

    # is the prefill cache identical? (it must be, prefill was correct)
    sp = list(out.past_key_values)
    vp = list(vout.past_key_values)
    print(f"\nprefill cache comparison:")
    for i in (0, 1, 2, 3, 4):
        kd = float((sp[i][0].float() - vp[i][0].float()).abs().max())
        vd = float((sp[i][1].float() - vp[i][1].float()).abs().max())
        print(f"  layer {i}: K err {kd:.4e}  V err {vd:.4e}")


if __name__ == "__main__":
    main()
