"""Isolate attention at long sequence length using the real layer's tensors.

Layer-0 cosine is 0.933 at S=12 but 0.720 at S=529. Determine whether the
attention stage or an earlier stage is responsible.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer, _Buf  # noqa: E402


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import SYSTEM_PROMPT, load_hybrid_agent

    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "How many trailing zeros does 100! have?"}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    print(f"S = {S}")

    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    q_c = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, 32, 128),
              w["self_attn.q_norm.weight"])
    k_c = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, 8, 128),
              w["self_attn.k_norm.weight"])
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, 8, 128)

    # CPU attention over the full S
    qh = q_c.transpose(0, 1)
    kh = k_c.transpose(0, 1).repeat_interleave(4, dim=0)
    vh = v_c.transpose(0, 1).repeat_interleave(4, dim=0)
    scale = 1.0 / (128 ** 0.5)
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, 32, S, S).masked_fill(~allow, float("-inf"))
    attn = torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask
    probs = torch.softmax(attn, dim=-1)
    out_cpu = torch.matmul(probs, vh)

    print(f"CPU probs rowsum range: {probs.sum(-1).min():.6f}..{probs.sum(-1).max():.6f}")

    # Vulkan attention
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

    print(f"VK  probs rowsum range: {got_p.sum(-1).min():.6f}..{got_p.sum(-1).max():.6f}")
    print(f"probs err  : {float((got_p - probs).abs().max()):.3e}")
    print(f"out err    : {float((got_o - out_cpu).abs().max()):.3e}")

    # per-query-row error, to see WHERE it diverges
    per_row = (got_o - out_cpu).abs().amax(dim=(0, 2))
    print(f"per-row err: first 5 = {[round(float(x),4) for x in per_row[:5]]}")
    print(f"             last 5  = {[round(float(x),4) for x in per_row[-5:]]}")
    print(f"             max at row {int(per_row.argmax())}")


if __name__ == "__main__":
    main()
