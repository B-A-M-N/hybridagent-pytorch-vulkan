"""Print the offload layer's internal stages against the real CPU layer."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer, _Buf  # noqa: E402

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
    print(f"S={S}")

    x = qm.embed_tokens(ids)

    # --- real CPU layer, capturing intermediates via hooks
    cpu_layer = qm.layers[0]
    captured = {}

    def hook(name):
        def fn(mod, args, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return fn

    cpu_layer.self_attn.register_forward_hook(hook("attn_out"))
    cpu_layer.mlp.register_forward_hook(hook("mlp_out"))
    pe = qm.rotary_emb(x, torch.arange(S).unsqueeze(0))
    with torch.no_grad():
        h_cpu = cpu_layer(x, attention_mask=None, position_ids=None,
                          past_key_values=None, use_cache=False,
                          cache_position=None, position_embeddings=pe)
    print(f"cpu layer out[0,:3] = {h_cpu[0,0,:3].tolist()}")

    # --- offload layer
    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(cpu_layer.state_dict(), strict=True)
    with torch.no_grad():
        h_vk = layer(x, position_embeddings=pe, attention_mask=None,
                     past_key_values=None, cache_position=torch.arange(S))
    print(f"vk  layer out[0,:3] = {h_vk[0,0,:3].tolist()}")
    print(f"layer err: {float((h_vk.float() - h_cpu.float()).abs().max()):.3e}")

    # --- compare the attention output stage
    # recompute the CPU attention output for comparison
    w = {k: v.float() for k, v in cpu_layer.state_dict().items()}

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

    h = rms(x.float()) * w["input_layernorm.weight"]
    q = (h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD)
    k = (h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD)
    v = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    q = rms(q) * w["self_attn.q_norm.weight"]
    k = rms(k) * w["self_attn.k_norm.weight"]

    inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2).float() / HD))
    # per-token positions 0..S-1
    fr = torch.outer(torch.arange(S).float(), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HD // 2

    def rope(t):
        return t * c.unsqueeze(1) + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * s.unsqueeze(1)

    q, k = rope(q), rope(k)
    krep = k.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    vrep = v.permute(1, 0, 2).repeat_interleave(NH // NKV, dim=0)
    qh = q.permute(1, 0, 2)
    sc = HD ** -0.5
    scores = torch.einsum("hsd,hjd->hsj", qh, krep) * sc
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    aw = torch.softmax(scores, dim=-1)
    attn_cpu = torch.einsum("hsj,hjd->hsd", aw, vrep).permute(1, 0, 2).reshape(S, HD_)

    # Vulkan attention with per-token RoPE
    qkv = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    qkvb = _Buf(qkv)
    tvk.rope_qkv(qkvb.buf, 0, NH, NKV, HD, n_tok=S)
    qkv2 = qkvb.cpu()
    q2 = qkv2[:S * HD_].view(S, NH, HD)
    k2 = qkv2[S * HD_:S * HD_ + S * KVD].view(S, NKV, HD)
    v2 = qkv2[S * HD_ + S * KVD:].view(S, NKV, HD)

    kcache = k2.permute(1, 0, 2).contiguous()
    vcache = v2.permute(1, 0, 2).contiguous()
    q_in = q2.permute(1, 0, 2).contiguous()
    allow_vk = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(1, NH, S, S).masked_fill(~allow_vk.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = tvk.VkBuffer(max(1, NH * S * S) * 4, "device")
    ob = tvk.VkBuffer(max(1, NH * S * HD) * 4, "host")
    tvk.attention2(_Buf(q_in).buf, _Buf(kcache).buf, _Buf(vcache).buf,
                   _Buf(mask).buf, probs, ob, 0, S, NH, NH // NKV, HD, S, sc)
    attn_vk = ob.download((NH, S, HD)).permute(1, 0, 2).reshape(S, HD_)
    print(f"attn err: {float((attn_vk - attn_cpu).abs().max()):.3e}")


if __name__ == "__main__":
    main()
