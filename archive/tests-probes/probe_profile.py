"""Time each stage of the layer separately to find the 64 ms."""
import os
import sys
import time

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

    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    layer.upload_weights()

    pe = qm.rotary_emb(x, torch.arange(S).unsqueeze(0))
    cp = torch.arange(S)

    def t(label, fn, n=5):
        fn()
        ts = []
        for _ in range(n):
            t0 = time.time()
            fn()
            ts.append(time.time() - t0)
        ts.sort()
        print(f"{label:28s} {ts[n // 2] * 1000:8.2f} ms")

    h = x.detach().to(torch.float32)

    t("input rmsnorm", lambda: layer._rms(h, layer._w["input_ln"]))
    t("q_proj", lambda: layer._lin(h, layer._w["q"], layer._zero_hd, HD_))
    t("k_proj", lambda: layer._lin(h, layer._w["k"], layer._zero_kvd, KVD))
    t("v_proj", lambda: layer._lin(h, layer._w["v"], layer._zero_kvd, KVD))
    t("full layer", lambda: layer(x, position_embeddings=pe, attention_mask=None,
                                  past_key_values=None, cache_position=cp))


if __name__ == "__main__":
    main()
