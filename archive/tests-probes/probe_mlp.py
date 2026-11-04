"""Isolate the MLP stage of the offload layer vs CPU."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
H, I, NH, NKV, HD = 4096, 12288, 32, 8, 128
HD_ = NH * HD
KVD = NKV * HD


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    x = qm.embed_tokens(ids)
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

    h = rms(x.float()) * w["input_layernorm.weight"]
    q = (h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD)
    k = (h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD)
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    q = rms(q) * w["self_attn.q_norm.weight"]
    k = rms(k) * w["self_attn.k_norm.weight"]

    inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.arange(S).float(), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HD // 2

    def rope(t):
        return t * c.unsqueeze(1) + torch.cat(
            [-t[..., half:], t[..., :half]], dim=-1) * s.unsqueeze(1)

    q, k = rope(q), rope(k)
    krep = k.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    vrep = v.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)
    sc = HD ** -0.5
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    aw = torch.softmax(scores, dim=-1)
    attn = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)
    out = attn @ w["self_attn.o_proj.weight"].t()
    h_res = x + out
    h2 = rms(h_res.float()) * w["post_attention_layernorm.weight"]
    mlp_cpu = (torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].t())
               * (h2 @ w["mlp.up_proj.weight"].t())) @ w["mlp.down_proj.weight"].t()
    print(f"cpu mlp[0,:3] = {mlp_cpu[0,:3].tolist()}")

    # Vulkan MLP via the layer's helpers
    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    layer.upload_weights()

    def vk_lin(t, wb, bb, out_f):
        from backend.hybrid_offload import _Buf
        M = t.numel() // t.shape[-1]
        K = t.shape[-1]
        xb = _Buf(t.reshape(M, K))
        ob = _Buf.out((M, out_f))
        tvk.matmul_t_bias(xb.buf, wb.buf, bb.buf, ob.buf, M, K, out_f)
        return ob.cpu().reshape(t.shape[:-1] + (out_f,))

    def vk_rms(t, wb):
        from backend.hybrid_offload import _Buf
        rows = t.numel() // t.shape[-1]
        rs = t.shape[-1]
        xb = _Buf(t.reshape(rows, rs))
        ob = _Buf.out((rows, rs))
        tvk.rmsnorm(xb.buf, wb.buf, ob.buf, rows, rs, 1e-6)
        return ob.cpu().reshape(t.shape)

    h2_vk = vk_rms(h_res.float(), layer._w["post_ln"])
    print(f"post_ln err: {float((h2_vk - h2).abs().max()):.3e}")
    from backend.hybrid_offload import _Buf
    gu = _Buf.out((S, I))
    tvk.fused_gate_up(_Buf(h2_vk).buf, layer._w["gate_up"].buf, gu.buf, S, H, I)
    gu_t = gu.cpu()
    gate_ref = torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].t())
    up_ref = h2 @ w["mlp.up_proj.weight"].t()
    print(f"gate err: {float((gu_t[:, :I] - gate_ref).abs().max()):.3e}")
    print(f"up err  : {float((gu_t - gate_ref * up_ref).abs().max()):.3e}")
    mlp_vk = vk_lin(gu_t, layer._w["down"], layer._zero_h, H)
    print(f"mlp err: {float((mlp_vk - mlp_cpu).abs().max()):.3e}")


if __name__ == "__main__":
    main()
