"""Instrument the refcounts of every buffer attention2 touches, per call.

The step-1 attention returns all-zero probabilities -- the signature of reading
a freed buffer. kb/vb are held as named locals, so the suspects are:
  * probs / ob  (raw VkBuffer allocated fresh each forward)
  * the _Buf wrappers themselves, if __del__ fires early

This logs _refs for each buffer immediately before and after the attention2
call on step 0 and step 1, and reports whether any hits zero.
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
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]

    bb = HybridBackbone(qm, Placement.first_n(4, len(qm.layers)))

    calls = {"n": 0}
    orig = tvk.attention2

    def spy(q, k, v, mask, probs, out, *a, **kw):
        n = calls["n"]
        calls["n"] += 1
        print(f"\n--- attention2 call #{n} ---")
        print(f"  args: seq_start={a[0]} seq_len={a[1]} n_heads={a[2]} "
              f"cap={a[5]}")
        for name, b in (("q", q), ("k", k), ("v", v), ("mask", mask),
                        ("probs", probs), ("out", out)):
            print(f"  {name:<6} handle={b.vk_handle} refs={b._refs} "
                  f"size={b.size}")
        r = orig(q, k, v, mask, probs, out, *a, **kw)
        print(f"  after dispatch:")
        for name, b in (("q", q), ("k", k), ("v", v), ("mask", mask),
                        ("probs", probs), ("out", out)):
            print(f"  {name:<6} refs={b._refs}")
        # what did the kernel actually produce?
        try:
            pr = probs.download((a[2], q.size // (4 * a[2] * a[4]), a[5]))
            print(f"  probs rowsum = {float(pr.sum(-1).mean()):.6f}  "
                  f"max = {float(pr.max()):.6f}")
        except Exception as e:
            print(f"  probs readback failed: {e}")
        return r

    tvk.attention2 = spy
    try:
        vc = DynamicCache()
        with torch.no_grad():
            v = bb(input_ids=ids, past_key_values=vc, use_cache=True)
        nxt = lm.lm_head(v.last_hidden_state[:, -1, :]).argmax(dim=-1,
                                                                keepdim=True)
        with torch.no_grad():
            bb(input_ids=nxt, past_key_values=vc, use_cache=True)
    finally:
        tvk.attention2 = orig


if __name__ == "__main__":
    main()
