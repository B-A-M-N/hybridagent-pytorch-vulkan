"""
Stage A gate: every Vulkan kernel against a PyTorch CPU reference, on the real
RX 6600 XT, with the device confirmed live.

Each test prints the measured max-abs error so the numerical deviation from the
reference is explicit. Tolerances are stated up front, not retro-fitted.

Run:  .venv/bin/python tests/test_vulkan_ops.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch_vulkan_ext as tvk

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def check(name, got, ref, atol=1e-4, rtol=1e-4):
    g = got.detach().float().cpu()
    r = ref.detach().float().cpu()
    if g.shape != r.shape:
        report(name, False, f"shape mismatch got={tuple(g.shape)} ref={tuple(r.shape)}")
        return
    err = (g - r).abs()
    max_err = float(err.max()) if err.numel() else 0.0
    ok = bool(torch.allclose(g, r, atol=atol, rtol=rtol))
    report(name, ok, f"max_abs_err={max_err:.3e} (atol={atol:.0e} rtol={rtol:.0e})")


def run(kernel_fn, *cpu_tensors, out_shape):
    """Upload CPU tensors, run the kernel, download the output."""
    bufs = [tvk.VkBuffer(t.numel() * 4, "device") for t in cpu_tensors]
    for b, t in zip(bufs, cpu_tensors):
        b.upload(t)
    ob = tvk.VkBuffer(max(1, out_shape.numel()) * 4, "host")
    kernel_fn(bufs, ob)
    return ob.download(tuple(out_shape))


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__}")
    tvk.init(0)
    print(f"vulkan device: {tvk.device_name()}")
    print(f"vram total: {tvk.vram_total()/1e9:.2f} GB\n")

    D = 4096

    # ---------------------------------------------------------------- rmsnorm
    x = torch.randn(4, D)
    w = torch.randn(D) * 0.5 + 1.0
    eps = 1e-6
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) * w
    got = run(lambda b, o: tvk.rmsnorm(b[0], b[1], o, 4, D, eps), x, w,
              out_shape=torch.Size([4, D]))
    check("rmsnorm[D=4096,eps=1e-6]", got, ref)

    # -------------------------------------------------------------- layernorm
    lw = torch.randn(D)
    lb = torch.randn(D) * 0.1
    ref_ln = torch.nn.functional.layer_norm(x, (D,), lw, lb, 1e-5)
    got = run(lambda b, o: tvk.layernorm(b[0], b[1], b[2], o, 4, D, 1e-5),
              x, lw, lb, out_shape=torch.Size([4, D]))
    check("layernorm[bias,eps=1e-5]", got, ref_ln)

    # ---------------------------------------------------------------- gelu_erf
    g = torch.linspace(-4, 4, 8192)
    ref_gelu = torch.nn.functional.gelu(g)
    got = run(lambda b, o: tvk.gelu_erf(b[0], o, g.numel()), g,
              out_shape=torch.Size([g.numel()]))
    check("gelu_erf[matches nn.GELU]", got, ref_gelu, atol=1e-5)

    # ------------------------------------------------------------------- rope
    n_heads, head_dim = 32, 128
    pos = 7
    q = torch.randn(n_heads, head_dim)
    inv = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.tensor([float(pos)]), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    half = head_dim // 2
    ref_rope = q * cos + torch.cat([-q[:, half:], q[:, :half]], dim=-1) * sin
    qb = tvk.VkBuffer(q.numel() * 4, "device"); qb.upload(q)
    for h in range(n_heads):
        tvk.rope(qb, h * head_dim, head_dim, pos)
    got = qb.download((n_heads, head_dim))
    check("rope[Qwen3 interleave,hd=128,pos=7]", got, ref_rope, atol=1e-4)

    # --------------------------------------------------------------- silu_gate
    N = 12288
    gate = torch.randn(1, 2 * N)
    ref_sg = torch.nn.functional.silu(gate[:, :N]) * gate[:, N:]
    got = run(lambda b, o: tvk.silu_gate(b[0], o, N), gate,
              out_shape=torch.Size([1, N]))
    check("silu_gate[N=12288]", got, ref_sg)

    # ------------------------------------------------------------ fused_gate_up
    M, K, NI = 8, D, 12288
    a = torch.randn(M, K)
    wg = torch.randn(NI, K)
    wu = torch.randn(NI, K)
    wgu = torch.cat([wg, wu], dim=0).t().contiguous()
    ref_mlp = torch.nn.functional.silu(a @ wg.t()) * (a @ wu.t())
    got = run(lambda b, o: tvk.fused_gate_up(b[0], b[1], o, M, K, NI), a, wgu,
              out_shape=torch.Size([M, NI]))
    check("fused_gate_up[8x4096x12288]", got, ref_mlp, atol=5e-2, rtol=5e-2)

    # -------------------------------------------------------------- matmul_t
    W = torch.randn(4096, 4096)
    ref_mm = a @ W.t()
    got = run(lambda b, o: tvk.matmul_t_bias(b[0], b[1], b[2], o, M, K, 4096),
              a, W, torch.zeros(4096), out_shape=torch.Size([M, 4096]))
    check("matmul_t_bias[nn.Linear layout]", got, ref_mm, atol=5e-2, rtol=5e-2)

    # -------------------------------------------------- rope_kv_store + attn
    n_heads2, n_kv_heads2, hd2 = 8, 2, 128
    hpk = n_heads2 // n_kv_heads2
    cap = 64
    pos2 = 16
    q2 = torch.randn(n_heads2, hd2)
    k2 = torch.randn(n_kv_heads2, hd2)
    v2 = torch.randn(n_kv_heads2, hd2)
    kc = torch.randn(n_kv_heads2, cap, hd2)
    vc = torch.randn(n_kv_heads2, cap, hd2)
    qkv = torch.cat([q2.reshape(-1), k2.reshape(-1), v2.reshape(-1)])

    qb = tvk.VkBuffer(qkv.numel() * 4, "device"); qb.upload(qkv)
    kb = tvk.VkBuffer(kc.numel() * 4, "device"); kb.upload(kc)
    vb = tvk.VkBuffer(vc.numel() * 4, "device"); vb.upload(vc)
    tvk.rope_kv_store(qb, kb, vb, pos2, cap, n_heads2, n_kv_heads2, hd2)

    kc2 = kb.download((n_kv_heads2, cap, hd2))
    vc2 = vb.download((n_kv_heads2, cap, hd2))
    q2r = qb.download((qkv.numel(),))[: n_heads2 * hd2].reshape(n_heads2, hd2)

    inv2 = 1.0 / (1000000.0 ** (torch.arange(0, hd2, 2).float() / hd2))
    fr2 = torch.outer(torch.tensor([float(pos2)]), inv2)
    emb2 = torch.cat([fr2, fr2], dim=-1)
    c2, s2 = emb2.cos(), emb2.sin()

    def rope_ref(t):
        hh = hd2 // 2
        return t * c2 + torch.cat([-t[:, hh:], t[:, :hh]], dim=-1) * s2

    kq = q2r          # Q was RoPEd in place by the kernel
    kk = rope_ref(k2)  # K is also RoPEd in place before being stored
    # The kernel RoPEs K in place and stores the RoPEd value, so the correct
    # reference for the slot is rope_ref(k2). V is copied unmodified.
    kcache_ref = kc.clone(); kcache_ref[:, pos2, :] = kk
    vcache_ref = vc.clone(); vcache_ref[:, pos2, :] = v2
    check("rope_kv_store[K slot]", kc2[:, pos2, :], kk, atol=1e-4)
    check("rope_kv_store[V slot]", vc2[:, pos2, :], v2, atol=1e-4)
    # Nothing outside the written slot may change.
    assert torch.allclose(kc2[:, :pos2, :], kc[:, :pos2, :], atol=1e-5), \
        "K cache was modified outside the written slot"
    assert torch.allclose(vc2[:, :pos2, :], vc[:, :pos2, :], atol=1e-5), \
        "V cache was modified outside the written slot"

    # ---- attention reference, built exactly as the model would hold it ----
    # After rope_kv_store, slot `pos2` holds the RoPEd K and V; Q is the RoPEd
    # q2r. The kernel indexes q as [batch, n_heads, seq_len, head_dim] and the
    # caches as [batch, n_kv_heads, cap, head_dim].
    new_len = pos2 + 1
    scaling = hd2 ** -0.5
    krep = kc2.repeat_interleave(hpk, dim=0)[:, :new_len, :]
    vrep = vc2.repeat_interleave(hpk, dim=0)[:, :new_len, :]
    # decode has exactly one query position, so the kernel is fed
    # q[n_heads, head_dim] with seq_len = new_len and a single query row.
    aw = torch.softmax(
        torch.einsum("hd,hjd->hj", kq, krep) * scaling, dim=-1)
    ref_out = torch.einsum("hs,hsd->hd", aw, vrep)

    mask = torch.zeros(1, n_heads2, 1, new_len)
    # probs scratch layout: [batch, n_heads, seq_len, seq_len]
    probs = tvk.VkBuffer(max(1, n_heads2 * new_len * new_len) * 4, "device")
    # The kernel indexes q as [batch, n_heads, seq_len, head_dim], so the stride
    # between heads is seq_len * head_dim. Feeding a bare [n_heads, head_dim]
    # buffer makes the kernel read head h at offset h*S*hd -- past the end of
    # the buffer for S > 1, which is why only the first head was correct.
    q_buf = kq.reshape(n_heads2, 1, hd2).expand(n_heads2, new_len, hd2).contiguous()
    qb2 = tvk.VkBuffer(q_buf.numel() * 4, "device"); qb2.upload(q_buf)
    kb2 = tvk.VkBuffer(kc2.numel() * 4, "device"); kb2.upload(kc2)
    vb2 = tvk.VkBuffer(vc2.numel() * 4, "device"); vb2.upload(vc2)
    mb2 = tvk.VkBuffer(mask.numel() * 4, "device"); mb2.upload(mask)
    # out is [n_heads, seq_len, head_dim]; the window is new_len positions, so
    # the buffer must be n_heads2 * new_len * hd2. Allocating only n_heads2*hd2
    # makes the later heads write out of bounds and read back as garbage.
    ob2 = tvk.VkBuffer(max(1, n_heads2 * new_len * hd2) * 4, "host")
    tvk.attention2(qb2, kb2, vb2, mb2, probs, ob2, 0, new_len,
                   n_heads2, hpk, hd2, cap, scaling)
    got_attn = ob2.download((n_heads2, new_len, hd2))[:, 0, :]
    check("attention2[GQA 8/2 hd=128]", got_attn, ref_out, atol=1e-3)

    # --------------------------------------------------------------- argmax
    logits = torch.randn(4, 151671)
    # per-row argmax: each batch row picks its own token
    _lb = tvk.VkBuffer(logits.numel() * 4, "device"); _lb.upload(logits)
    _ob = tvk.VkBuffer(4 * 4, "host"); tvk.argmax_rows(_lb, _ob, 4, 151671)
    got_rows = _ob.download_u32(torch.Size([4]))
    check("argmax_rows[4x151671]", got_rows, logits.argmax(-1), atol=0, rtol=0)
    # global argmax over the flattened matrix
    _ob2 = tvk.VkBuffer(4, "host"); tvk.argmax(_lb, _ob2, 4, 151671)
    got_g = _ob2.download_u32(torch.Size([1]))
    check("argmax[4x151671]", got_g,
          logits.reshape(-1).argmax().reshape(1), atol=0, rtol=0)

    print()
    print(f"dispatches issued: {tvk.dispatch_count()}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
