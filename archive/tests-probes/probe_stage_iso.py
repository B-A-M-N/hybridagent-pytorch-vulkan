"""Find the first divergent stage inside the Vulkan layer at S=529.

Runs the layer with the real inputs and compares each intermediate against a
CPU reference computed in the same precision, reporting the first stage where
the error exceeds a tight tolerance.
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
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}
    print(f"S={S}")

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    # ---- CPU reference stages (fp32)
    h0 = emb.float()
    h = rms(h0, w["input_layernorm.weight"])
    # RAW projections (no norm) -- compared against the raw Vulkan projection.
    q_raw = (h @ w["self_attn.q_proj.weight"].t()).view(S, 32, 128)
    k_raw = (h @ w["self_attn.k_proj.weight"].t()).view(S, 8, 128)
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, 8, 128)
    q_c = rms(q_raw, w["self_attn.q_norm.weight"])
    k_c = rms(k_raw, w["self_attn.k_norm.weight"])

    qh = q_c.transpose(0, 1)
    kh = k_c.transpose(0, 1).repeat_interleave(4, dim=0)
    vh = v_c.transpose(0, 1).repeat_interleave(4, dim=0)
    scale = 1.0 / (128 ** 0.5)
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, 32, S, S).masked_fill(~allow, float("-inf"))
    attn_w = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask,
                           dim=-1)
    attn_o = torch.matmul(attn_w, vh).transpose(0, 1).reshape(S, 4096)
    o_out = attn_o @ w["self_attn.o_proj.weight"].t()
    res1 = h0 + o_out
    h2 = rms(res1, w["post_attention_layernorm.weight"])
    gu = torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].t()) * \
        (h2 @ w["mlp.up_proj.weight"].t())
    down = gu @ w["mlp.down_proj.weight"].t()
    res2 = res1 + down

    # ---- Vulkan stages
    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vk.upload_weights()

    def v_rms(t, wb):
        rows = t.numel() // t.shape[-1]
        xb = _Buf(t.reshape(rows, t.shape[-1]))
        ob = _Buf.out((rows, t.shape[-1]))
        tvk.rmsnorm(xb.buf, wb.buf, ob.buf, rows, t.shape[-1], 1e-6)
        return ob.cpu().reshape(t.shape)

    def v_lin(t, wb, bb, out_f):
        M = t.numel() // t.shape[-1]
        K = t.shape[-1]
        xb = _Buf(t.reshape(M, K))
        ob = _Buf.out((M, out_f))
        tvk.matmul_t_bias(xb.buf, wb.buf, bb.buf, ob.buf, M, K, out_f)
        return ob.cpu().reshape(t.shape[:-1] + (out_f,))

    hv = v_rms(h0, vk._w["input_ln"])
    print(f"stage input_ln      err {float((hv - h).abs().max()):.3e}")
    qv = v_lin(hv, vk._w["q"], vk._zero_hd, 4096).view(S, 32, 128)
    print(f"stage q_proj        err {float((qv - q_raw).abs().max()):.3e}")
    qv2 = v_rms(qv.cpu().reshape(S * 32, 128), vk._w["q_norm"])
    print(f"stage q_norm        err {float((qv2.reshape(S,32,128) - q_c).abs().max()):.3e}")
    kv = v_lin(hv, vk._w["k"], vk._zero_kvd, 1024).view(S, 8, 128)
    kv2 = v_rms(kv.cpu().reshape(S * 8, 128), vk._w["k_norm"])
    print(f"stage k_norm        err {float((kv2.reshape(S,8,128) - k_c).abs().max()):.3e}")
    vv = v_lin(hv, vk._w["v"], vk._zero_kvd, 1024).view(S, 8, 128)
    print(f"stage v_proj        err {float((vv - v_c).abs().max()):.3e}")

    # attention via the kernel
    q_in = qv2.reshape(S, 32, 128).transpose(0, 1).contiguous()
    kb = _Buf(k_c.transpose(0, 1).contiguous())
    vb = _Buf(v_c.transpose(0, 1).contiguous())
    mb = _Buf(mask.contiguous())
    pb = _Buf.out((32, S, S))
    ob = _Buf.out((32, S, 128))
    tvk.attention2(_Buf(q_in).buf, kb.buf, vb.buf, mb.buf, pb.buf, ob.buf,
                   0, S, 32, 4, 128, S, scale)
    got_w = pb.cpu()
    got_o = ob.cpu().permute(1, 0, 2).reshape(S, 4096)
    print(f"stage attention_w   err {float((got_w - attn_w).abs().max()):.3e}")
    print(f"stage attention_out err {float((got_o - attn_o).abs().max()):.3e}")

    ov = v_lin(got_o.contiguous(), vk._w["o"], vk._zero_h, 4096)
    print(f"stage o_proj        err {float((ov - o_out).abs().max()):.3e}")
    res1v = h0 + ov
    print(f"stage residual1     err {float((res1v - res1).abs().max()):.3e}")
    h2v = v_rms(res1v, vk._w["post_ln"])
    print(f"stage post_ln       err {float((h2v - h2).abs().max()):.3e}")


if __name__ == "__main__":
    main()
