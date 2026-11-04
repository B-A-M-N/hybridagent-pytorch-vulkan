"""Which cache is correct: the stock layer's or the Vulkan layer's?

probe_kiso showed the Vulkan cached K matches a manual reference to 0.493
(bf16 noise at RMS 13.5), while the stock layer's cached K was reported as
diverging by 1.0. That inverts the assumption.

This measures BOTH caches against the same independent reference:
  * a manual projection + RMSNorm + apply_rotary_pos_emb, computed in fp32
  * the stock layer's cache
  * the Vulkan layer's cache

Whichever is closer to the reference is the correct one.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
NKV, HD = 8, 128


def main():
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model

    ids = tok("How many trailing zeros does 100! have?",
              return_tensors="pt")["input_ids"]
    S = ids.shape[1]
    emb = qm.embed_tokens(ids)
    pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
    cp = torch.arange(S)

    # ---- independent fp32 reference --------------------------------------
    w = {k: v.float() for k, v in qm.layers[0].state_dict().items()}

    def rms(t, wt):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * wt

    h = rms(emb.float(), w["input_layernorm.weight"])
    k_raw = rms((h @ w["self_attn.k_proj.weight"].t()).view(S, NKV, HD),
                w["self_attn.k_norm.weight"])
    v_raw = (h @ w["self_attn.v_proj.weight"].t()).view(S, NKV, HD)
    k4 = k_raw.transpose(0, 1).unsqueeze(0)
    v4 = v_raw.transpose(0, 1).unsqueeze(0)
    _, k_ref = apply_rotary_pos_emb(torch.zeros_like(k4), k4, pe[0].float(),
                                    pe[1].float())
    k_ref = k_ref[0]          # [n_kv, S, hd]
    v_ref = v4[0]

    # ---- stock cache ------------------------------------------------------
    sc = DynamicCache()
    with torch.no_grad():
        qm.layers[0](emb, position_embeddings=pe, attention_mask=None,
                     past_key_values=sc, use_cache=True, cache_position=cp)
    sk, sv = list(sc)[0][0][0], list(sc)[0][1][0]

    # ---- vulkan cache -----------------------------------------------------
    vk = VulkanDecoderLayer(qm.config, 0)
    vk.load_state_dict(qm.layers[0].state_dict(), strict=True)
    vc = DynamicCache()
    with torch.no_grad():
        vk(emb, position_embeddings=pe, attention_mask=None,
           past_key_values=vc, use_cache=True, cache_position=cp)
    vk_, vv = list(vc)[0][0][0], list(vc)[0][1][0]

    print(f"S={S}")
    print(f"\n{'tensor':<12}{'err vs fp32 ref':>18}{'rms':>12}")
    print("-" * 44)
    print(f"{'reference K':<12}{0.0:>18.4e}{float(k_ref.pow(2).mean().sqrt()):>12.4f}")
    print(f"{'stock K':<12}{float((sk.float()-k_ref).abs().max()):>18.4e}"
          f"{float(sk.float().pow(2).mean().sqrt()):>12.4f}")
    print(f"{'vulkan K':<12}{float((vk_.float()-k_ref).abs().max()):>18.4e}"
          f"{float(vk_.float().pow(2).mean().sqrt()):>12.4f}")
    print()
    print(f"{'reference V':<12}{0.0:>18.4e}{float(v_ref.pow(2).mean().sqrt()):>12.4f}")
    print(f"{'stock V':<12}{float((sv.float()-v_ref).abs().max()):>18.4e}"
          f"{float(sv.float().pow(2).mean().sqrt()):>12.4f}")
    print(f"{'vulkan V':<12}{float((vv.float()-v_ref).abs().max()):>18.4e}"
          f"{float(vv.float().pow(2).mean().sqrt()):>12.4f}")

    print(f"\nstock K vs vulkan K: "
          f"{float((sk.float()-vk_.float()).abs().max()):.4e}")
    print(f"stock V vs vulkan V: "
          f"{float((sv.float()-vv.float()).abs().max()):.4e}")


if __name__ == "__main__":
    main()
