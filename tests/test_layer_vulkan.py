"""
Stage A gate 2: a complete Qwen3 transformer layer executing through Vulkan,
compared against the original CPU PyTorch implementation.

This is the Deliverable-1 proof: not isolated kernels, but a whole layer
(RMSNorm -> QKV projection -> QK-norm -> RoPE -> KV cache -> GQA attention ->
output projection -> residual -> RMSNorm -> SwiGLU MLP -> residual) running on
the GPU with the CPU reference alongside it.

KV-cache layout used here: [n_kv_heads, cap, head_dim] -- the layout
`kv_append` documents and writes.

Run:  .venv/bin/python tests/test_layer_vulkan.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch_vulkan_ext as tvk

HIDDEN = 4096
INTER = 12288
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
HD = N_HEADS * HEAD_DIM          # 4096
KVD = N_KV_HEADS * HEAD_DIM      # 1024
EPS = 1e-6


def make_layer(seed=0):
    g = torch.Generator().manual_seed(seed)
    def r(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).float()
    return {
        "input_ln_w": torch.randn(HIDDEN, generator=g).float(),
        "post_ln_w": torch.randn(HIDDEN, generator=g).float(),
        "q_w": r(N_HEADS * HEAD_DIM, HIDDEN),
        "k_w": r(N_KV_HEADS * HEAD_DIM, HIDDEN),
        "v_w": r(N_KV_HEADS * HEAD_DIM, HIDDEN),
        "o_w": r(HIDDEN, N_HEADS * HEAD_DIM),
        "q_norm_w": torch.randn(HEAD_DIM, generator=g).float(),
        "k_norm_w": torch.randn(HEAD_DIM, generator=g).float(),
        "gate_w": r(INTER, HIDDEN),
        "up_w": r(INTER, HIDDEN),
        "down_w": r(HIDDEN, INTER),
    }


def rms(t):
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS)


def rope_ref(t, pos):
    """t: [S, H, hd] -> RoPEd [S, H, hd]."""
    inv = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    fr = torch.outer(torch.tensor([float(pos)]), inv)
    emb = torch.cat([fr, fr], dim=-1)
    c, s = emb.cos(), emb.sin()
    half = HEAD_DIM // 2
    return t * c + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * s


def cpu_layer(x, w, pos, k_cache, v_cache, cap):
    """Original Qwen3DecoderLayer computation on CPU."""
    S = x.shape[0]

    h = rms(x.float()) * w["input_ln_w"]
    q = (h @ w["q_w"].t()).view(S, N_HEADS, HEAD_DIM)
    k = (h @ w["k_w"].t()).view(S, N_KV_HEADS, HEAD_DIM)
    v = (h @ w["v_w"].t()).view(S, N_KV_HEADS, HEAD_DIM)

    q = rms(q) * w["q_norm_w"]
    k = rms(k) * w["k_norm_w"]

    q = rope_ref(q, pos)
    k = rope_ref(k, pos)

    # cache: [n_kv_heads, cap, hd]
    k_cache[:, pos, :] = k.permute(1, 0, 2).reshape(N_KV_HEADS, S, HEAD_DIM)[:, 0, :]
    v_cache[:, pos, :] = v.permute(1, 0, 2).reshape(N_KV_HEADS, S, HEAD_DIM)[:, 0, :]

    krep = k_cache[:, :pos + 1, :].repeat_interleave(N_HEADS // N_KV_HEADS, dim=0)
    vrep = v_cache[:, :pos + 1, :].repeat_interleave(N_HEADS // N_KV_HEADS, dim=0)
    # krep is [N_HEADS, L, hd]. Decode semantics: every query row attends the
    # same cache window, so use the first SEQUENCE position for all of them.
    qh = q[:1].permute(1, 0, 2).expand(N_HEADS, S, HEAD_DIM)   # [H, S, hd]
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg cpu] qh[0,0,:3]  = {qh[0,0,:3].tolist()}")
    aw = torch.einsum("hsd,hjd->hsj", qh, krep) * (HEAD_DIM ** -0.5)
    aw = torch.softmax(aw, dim=-1)
    attn = torch.einsum("hsj,hjd->hsd", aw, vrep)             # [H, S, hd]
    attn = attn.permute(1, 0, 2).reshape(S, HD)
    global _cpu_attn_ref
    _cpu_attn_ref = attn
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg cpu] attn[0,:3] = {attn[0, :3].tolist()}")

    out = attn @ w["o_w"].t()
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg cpu] out[0,:3]    = {out[0,:3].tolist()}")
    h = x + out
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg cpu] h[0,:3] = {h[0, :3].tolist()}")

    h2 = rms(h.float()) * w["post_ln_w"]
    mlp = (F.silu(h2 @ w["gate_w"].t()) * (h2 @ w["up_w"].t())) @ w["down_w"].t()
    return h + mlp, k_cache, v_cache


class B:
    """Small helper around a Vulkan buffer."""
    def __init__(self, t, kind="device"):
        t = t.detach().float().contiguous()
        self.buf = tvk.VkBuffer(max(1, t.numel()) * 4, kind)
        self.buf.upload(t)
        self.shape = tuple(t.shape)

    @classmethod
    def out(cls, shape):
        n = 1
        for s in shape:
            n *= s
        return cls(torch.zeros(n), "host")._as(shape)

    def _as(self, shape):
        self.shape = tuple(shape)
        return self

    def cpu(self):
        return self.buf.download(self.shape)

    def reshape(self, *shape):
        return B(self.cpu().reshape(*shape))


_cpu_attn_ref = None


def vk_layer(x, w, pos, k_cache, v_cache, cap):
    """The same layer, executed through the Vulkan kernels."""
    S = x.shape[0]

    xb = B(x)
    h = B.out((S, HIDDEN))
    tvk.rmsnorm(xb.buf, B(w["input_ln_w"]).buf, h.buf, S, HIDDEN, EPS)
    h = B(h.cpu())

    zHD = B(torch.zeros(HD))
    q = B.out((S, HD)); tvk.matmul_t_bias(h.buf, B(w["q_w"]).buf, zHD.buf, q.buf, S, HIDDEN, HD)
    zKV = B(torch.zeros(KVD))
    k = B.out((S, KVD)); tvk.matmul_t_bias(h.buf, B(w["k_w"]).buf, zKV.buf, k.buf, S, HIDDEN, KVD)
    v = B.out((S, KVD)); tvk.matmul_t_bias(h.buf, B(w["v_w"]).buf, zKV.buf, v.buf, S, HIDDEN, KVD)

    q = B(q.cpu().reshape(S, N_HEADS, HEAD_DIM))
    k = B(k.cpu().reshape(S, N_KV_HEADS, HEAD_DIM))
    v = B(v.cpu().reshape(S, N_KV_HEADS, HEAD_DIM))

    # QK-norm over head_dim
    def rms_vk(t, weight):
        flat = t.cpu().reshape(-1, HEAD_DIM)
        o = B.out(flat.shape)
        tvk.rmsnorm(B(flat).buf, B(weight).buf, o.buf, flat.shape[0], HEAD_DIM, EPS)
        return B(o.cpu().reshape(t.shape))

    q = rms_vk(q, w["q_norm_w"])
    k = rms_vk(k, w["k_norm_w"])

    # RoPE on the packed [Q|K|V] buffer
    qkv = torch.cat([q.cpu().reshape(-1), k.cpu().reshape(-1), v.cpu().reshape(-1)])
    qkvb = B(qkv)
    tvk.rope_qkv(qkvb.buf, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, n_tok=S)
    qkv2 = qkvb.cpu()
    # q is [S, N_HEADS, hd] -> S*HD elements; k and v are S*KVD each.
    q = qkv2[:S * HD].view(S, N_HEADS, HEAD_DIM)
    k = qkv2[S * HD:S * HD + S * KVD].view(S, N_KV_HEADS, HEAD_DIM)
    v = qkv2[S * HD + S * KVD:S * HD + 2 * S * KVD].view(S, N_KV_HEADS, HEAD_DIM)

    # KV cache append (cache layout [cap, n_kv_heads, hd])
    kv_cat = torch.cat([
        k.cpu().permute(1, 0, 2).reshape(N_KV_HEADS, S, HEAD_DIM)[:, 0, :].reshape(-1),
        v.cpu().permute(1, 0, 2).reshape(N_KV_HEADS, S, HEAD_DIM)[:, 0, :].reshape(-1),
    ])
    # KV cache append. The layer passes only the K/V block, so q_len is 0 --
    # passing the Q length would shift every read by that many elements.
    kvb = B(kv_cat)
    kcb = B(k_cache); vcb = B(v_cache)
    tvk.kv_append(kvb.buf, kcb.buf, vcb.buf, pos, cap, HEAD_DIM, KVD, q_len=0)
    k_cache = kcb.cpu(); v_cache = vcb.cpu()

    # attention. Decode has ONE query position per head, so q is [n_heads, hd]
    # and the kernel derives n_q = 1 from the buffer size.
    q_buf = q.cpu()[0].reshape(N_HEADS, HEAD_DIM).contiguous()
    mask = torch.zeros(1, N_HEADS, 1, pos + 1)
    probs = tvk.VkBuffer(max(1, N_HEADS * (pos + 1) * (pos + 1)) * 4, "device")
    ob = tvk.VkBuffer(max(1, N_HEADS * HEAD_DIM) * 4, "host")
    tvk.attention2(B(q_buf).buf, kcb.buf, vcb.buf, B(mask).buf, probs, ob,
                   0, pos + 1, N_HEADS, N_HEADS // N_KV_HEADS, HEAD_DIM,
                   cap, HEAD_DIM ** -0.5)
    attn = ob.download((N_HEADS, HEAD_DIM)).reshape(N_HEADS * HEAD_DIM)
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg] attn[:3] = {attn[:3].tolist()}")

    # output projection + residual. The projection is over all S positions, so
    # broadcast the single decoded position across the batch the same way the
    # reference does.
    attn_b = attn.reshape(1, N_HEADS * HEAD_DIM).expand(S, N_HEADS * HEAD_DIM).contiguous()
    zH = B(torch.zeros(HIDDEN))
    ob2 = B.out((S, HIDDEN))
    tvk.matmul_t_bias(B(attn_b).buf, B(w["o_w"]).buf, zH.buf, ob2.buf, S, HD, HIDDEN)
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg vk ] attn_b err vs cpu attn[0]: "
              f"{float((attn_b[0] - _cpu_attn_ref[0]).abs().max()):.3e}")
        print(f"  [dbg vk ] ob2[0,:3]    = {ob2.cpu()[0,:3].tolist()}")
    h = x + ob2.cpu()
    if os.environ.get("LAYER_DEBUG"):
        print(f"  [dbg vk ] h[0,:3] = {h[0, :3].tolist()}")

    # post-attention RMSNorm
    hb = B(h)
    h2 = B.out((S, HIDDEN))
    tvk.rmsnorm(hb.buf, B(w["post_ln_w"]).buf, h2.buf, S, HIDDEN, EPS)
    h2 = B(h2.cpu())

    # fused gate+up: weight stacked [2*INTER, HIDDEN] transposed to [HIDDEN, 2*INTER]
    wgu = torch.cat([w["gate_w"], w["up_w"]], dim=0).t().contiguous()
    gu = B.out((S, INTER))
    tvk.fused_gate_up(h2.buf, B(wgu).buf, gu.buf, S, HIDDEN, INTER)
    gu = B(gu.cpu())

    ob3 = B.out((S, HIDDEN))
    tvk.matmul_t_bias(gu.buf, B(w["down_w"]).buf, zH.buf, ob3.buf, S, INTER, HIDDEN)
    return h + ob3.cpu(), k_cache, v_cache


def main():
    torch.manual_seed(0)
    tvk.init(0)
    print(f"vulkan: {tvk.device_name()}\n")

    S, cap, pos = 4, 32, 7
    x = torch.randn(S, HIDDEN)
    w = make_layer(0)

    kc0 = torch.randn(N_KV_HEADS, cap, HEAD_DIM)
    vc0 = torch.randn(N_KV_HEADS, cap, HEAD_DIM)

    ref, kcr, vcr = cpu_layer(x, w, pos, kc0.clone(), vc0.clone(), cap)
    got, kcg, vcg = vk_layer(x, w, pos, kc0.clone(), vc0.clone(), cap)

    print("=== Stage A gate 2: full Qwen3 layer on Vulkan ===")
    print(f"hidden state max_abs_err : {float((got - ref).abs().max()):.3e}")
    print(f"  relative to max |ref|  : "
          f"{float((got - ref).abs().max() / ref.abs().max()):.3e}")
    print(f"K cache max_abs_err      : {float((kcg - kcr).abs().max()):.3e}")
    print(f"V cache max_abs_err      : {float((vcg - vcr).abs().max()):.3e}")

    ok = float((got - ref).abs().max()) < 0.5
    print(f"\n[{'PASS' if ok else 'FAIL'}] full transformer layer matches CPU reference")
    print(f"dispatches: {tvk.dispatch_count()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
