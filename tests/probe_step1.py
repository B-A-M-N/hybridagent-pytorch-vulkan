"""Capture the Vulkan layer's attention inputs on step 1 vs the stock layer's.

Greedy decode agrees at step 0 and diverges at step 1, then the Vulkan output
repeats -- the signature of a corrupted KV cache being attended over.

Capture, for step 1 (S=1):
  * the q/k/v handed to attention2
  * the cache length seen (cap)
  * the mask
and compare against the stock layer's step-1 inputs.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer, HybridBackbone, Placement  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
NH, NKV, HD = 32, 8, 128


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]

    # step 0 on both, then capture step 1
    sc = DynamicCache()
    with torch.no_grad():
        o = lm(input_ids=ids, past_key_values=sc, use_cache=True)
    nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))
    vc = DynamicCache()
    with torch.no_grad():
        v = bb(input_ids=ids, past_key_values=vc, use_cache=True)

    # ---- capture the vulkan layer's attention2 args on step 1 -------------
    cap = {}
    orig = tvk.attention2

    def spy(q, k, vv, mask, probs, out, *a, **kw):
        cap['a'] = a
        n_q = q.size // (4 * NH * HD)
        cap['n_q'] = n_q
        cap['q'] = q.download((NH, n_q, HD))
        cap['k'] = k.download((NKV, a[5], HD))
        cap['v'] = vv.download((NKV, a[5], HD))
        cap['mask'] = mask.download((NH, n_q, a[5]))
        cap['probs'] = probs.download((NH, n_q, a[5]))
        return orig(q, k, vv, mask, probs, out, *a, **kw)

    tvk.attention2 = spy
    try:
        with torch.no_grad():
            bb(input_ids=nxt, past_key_values=vc, use_cache=True)
    finally:
        tvk.attention2 = orig

    a = cap['a']
    print(f"step-1 attention2 args: seq_start={a[0]} seq_len={a[1]} "
          f"n_heads={a[2]} hpk={a[3]} hd={a[4]} cap={a[5]}")
    print(f"  captured q {tuple(cap['q'].shape)}")
    print(f"  captured k {tuple(cap['k'].shape)}  (expect [8, 13, 128])")
    print(f"  captured v {tuple(cap['v'].shape)}")
    print(f"  mask shape {tuple(cap['mask'].shape)}")
    m = cap['mask']
    print(f"  mask[0,0,:5] = {[round(float(x),3) for x in m[0,0,:5]]}")
    print(f"  mask[0,0,-3:] = {[round(float(x),3) for x in m[0,0,-3:]]}")
    print(f"  probs rowsum = {float(cap['probs'].sum(-1).mean()):.6f}")

    # ---- compare against the stock cache at the same point ----------------
    sk = list(sc)
    print(f"\nstock cache after step 0: len {sk[0][0].shape[2]}")
    vk_ = list(vc)
    print(f"vk    cache after step 0: len {vk_[0][0].shape[2]}")

    # the captured k should equal the cache contents
    ck = vk_[0][0][0]        # [n_kv, L, hd]
    if ck.shape == cap['k'].shape:
        print(f"\ncaptured k vs vk cache: "
              f"{float((ck.float()-cap['k'].float()).abs().max()):.4e}")
    else:
        print(f"\nSHAPE MISMATCH: cache {tuple(ck.shape)} "
              f"captured {tuple(cap['k'].shape)}")


if __name__ == "__main__":
    main()
