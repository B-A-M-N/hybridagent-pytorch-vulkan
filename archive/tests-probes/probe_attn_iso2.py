"""Reproduce probe_attn_long's inputs and compare CPU vs GPU step by step.

probe_attn_long reported probs err 0.998 at S=529 while probe_attn_scale
reports 1.2e-07 with random inputs. The difference must be in how the inputs
were constructed. This isolates it.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import SYSTEM_PROMPT, load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "How many trailing zeros does 100! have?"}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    q_c = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, 32, 128),
              w["self_attn.q_norm.weight"])
    k_c = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, 8, 128),
              w["self_attn.k_norm.weight"])
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, 8, 128)

    print(f"S={S}")
    print(f"q_c shape {tuple(q_c.shape)} dtype {q_c.dtype}")
    print(f"k_c shape {tuple(k_c.shape)}")
    print(f"q_c norm {float(q_c.norm()):.3f}  k_c norm {float(k_c.norm()):.3f}")
    print(f"q_c has NaN: {bool(q_c.isnan().any())}  k_c NaN: {bool(k_c.isnan().any())}")

    # CPU reference
    qh = q_c.transpose(0, 1)
    kh = k_c.transpose(0, 1).repeat_interleave(4, dim=0)
    vh = v_c.transpose(0, 1).repeat_interleave(4, dim=0)
    scale = 1.0 / (128 ** 0.5)
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, 32, S, S).masked_fill(~allow, float("-inf"))
    attn = torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask
    probs = torch.softmax(attn, dim=-1)
    out_cpu = torch.matmul(probs, vh)

    # GPU
    q_in = q_c.transpose(0, 1).contiguous()
    kb = _Buf(k_c.transpose(0, 1).contiguous())
    vb = _Buf(v_c.transpose(0, 1).contiguous())
    mb = _Buf(mask.contiguous())
    pb = _Buf.out((32, S, S))
    ob = _Buf.out((32, S, 128))
    tvk.attention2(_Buf(q_in).buf, kb.buf, vb.buf, mb.buf, pb.buf, ob.buf,
                   0, S, 32, 4, 128, S, scale)
    got_p = pb.cpu()
    got_o = ob.cpu()

    print(f"\nprobs err: {float((got_p - probs).abs().max()):.3e}")
    print(f"out err  : {float((got_o - out_cpu).abs().max()):.3e}")

    # compare the score matrices directly
    gpu_scores = torch.log(got_p.clamp_min(1e-30))
    cpu_scores = torch.log(probs.clamp_min(1e-30))
    print(f"log-probs err: {float((gpu_scores - cpu_scores).abs().max()):.3e}")

    # per-head error
    per_head = (got_p - probs).abs().amax(dim=(1, 2))
    print(f"per-head probs err: min={float(per_head.min()):.3e} "
          f"max={float(per_head.max()):.3e}")
    bad = int((per_head > 1e-3).sum())
    print(f"heads with err>1e-3: {bad}/{NH}")


if __name__ == "__main__":
    main()
