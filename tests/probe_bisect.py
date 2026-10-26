"""Bisect the layer regression: check each stage against a CPU reference."""
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
    x = qm.embed_tokens(ids)
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

    # CPU reference
    h = rms(x.float()) * w["input_layernorm.weight"]
    q_c = (h @ w["self_attn.q_proj.weight"].t()).view(S, NH, HD)
    k_c = (h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD)
    v_c = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    q_c = rms(q_c) * w["self_attn.q_norm.weight"]
    k_c = rms(k_c) * w["self_attn.k_norm.weight"]

    # Vulkan
    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    layer.upload_weights()

    def vk_rms(t, wb):
        rows = t.numel() // t.shape[-1]
        xb = _Buf(t.reshape(rows, t.shape[-1]))
        ob = _Buf.out((rows, t.shape[-1]))
        tvk.rmsnorm(xb.buf, wb.buf, ob.buf, rows, t.shape[-1], 1e-6)
        return ob.cpu().reshape(t.shape)

    def vk_lin(t, wb, bb, out_f):
        M = t.numel() // t.shape[-1]
        K = t.shape[-1]
        xb = _Buf(t.reshape(M, K))
        ob = _Buf.out((M, out_f))
        tvk.matmul_t_bias(xb.buf, wb.buf, bb.buf, ob.buf, M, K, out_f)
        return ob.cpu().reshape(t.shape[:-1] + (out_f,))

    h_vk = vk_rms(x.float(), layer._w["input_ln"])
    print(f"h (input rmsnorm) err : {float((h_vk - h).abs().max()):.3e}")

    q_vk = vk_lin(h_vk, layer._w["q"], layer._zero_hd, HD_).view(S, NH, HD)
    print(f"q_proj err            : {float((q_vk - q_c).abs().max()):.3e}")

    # QK-norm: the [M, n_heads*hd] buffer must be viewed as [M*n_heads, hd]
    q_vk2 = vk_rms(q_vk.cpu().reshape(S * NH, HD), layer._w["q_norm"])
    print(f"q after q_norm err    : {float((q_vk2 - q_c).abs().max()):.3e}")

    k_vk = vk_lin(h_vk, layer._w["k"], layer._zero_kvd, KVD).view(S, NKV, HD)
    print(f"k_proj err            : {float((k_vk - k_c).abs().max()):.3e}")
    k_vk2 = vk_rms(k_vk.cpu().reshape(S * NKV, HD), layer._w["k_norm"])
    print(f"k after k_norm err    : {float((k_vk2 - k_c).abs().max()):.3e}")


if __name__ == "__main__":
    main()
