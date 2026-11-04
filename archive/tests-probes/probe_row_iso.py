"""Find the first divergent query row in the Vulkan layer at S=529.

Token 0 is exact and error grows with position. Instrument the layer's
attention output per query row and compare against a CPU reference built from
the SAME q/k/v the layer actually produced, so the comparison is like-for-like.
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
    print(f"S={S}")

    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    # Build q/k/v exactly as the layer does, then RoPE them.
    h0 = emb.float()
    h = rms(h0, w["input_layernorm.weight"])
    q = rms((h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD),
            w["self_attn.q_norm.weight"])
    k = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD),
            w["self_attn.k_norm.weight"])
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)

    # ---- CPU attention, per-row, using the SAME q/k/v
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    vh = v.transpose(0, 1).repeat_interleave(NH // NKV, dim=0)
    scale = 1.0 / (HD ** 0.5)
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, NH, S, S).masked_fill(~allow, float("-inf"))
    ref_w = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask,
                          dim=-1)
    ref_o = torch.matmul(ref_w, vh)

    # ---- Vulkan attention, same inputs
    q_in = qh.contiguous()
    kb = _Buf(kh.contiguous())
    vb = _Buf(vh.contiguous())
    mb = _Buf(mask.contiguous())
    pb = _Buf.out((NH, S, S))
    ob = _Buf.out((NH, S, HD))
    tvk.attention2(_Buf(q_in).buf, kb.buf, vb.buf, mb.buf, pb.buf, ob.buf,
                   0, S, NH, NH // NKV, HD, S, scale)
    got_w = pb.cpu()
    got_o = ob.cpu()

    print(f"probs err: {float((got_w - ref_w).abs().max()):.3e}")
    print(f"out err  : {float((got_o - ref_o).abs().max()):.3e}")

    # per-row error
    per_row_w = (got_w - ref_w).abs().amax(dim=(0, 2))
    per_row_o = (got_o - ref_o).abs().amax(dim=(0, 2))
    print(f"\nper-row probs err: first3={[round(float(x),6) for x in per_row_w.reshape(-1)[:3]]}")
    print(f"                  last3 ={[round(float(x),6) for x in per_row_w.reshape(-1)[-3:]]}")
    print(f"per-row out err  : first3={[round(float(x),6) for x in per_row_o.reshape(-1)[:3]]}")
    print(f"                  last3 ={[round(float(x),6) for x in per_row_o.reshape(-1)[-3:]]}")
    print(f"  argmax probs row: {int(per_row_w.reshape(-1).argmax())}")
    print(f"  argmax out   row: {int(per_row_o.reshape(-1).argmax())}")


if __name__ == "__main__":
    main()
