"""Split the layer's time between GPU dispatches and host round-trips."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import VulkanDecoderLayer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]
    x = qm.embed_tokens(ids)
    pe = qm.rotary_emb(x, torch.arange(ids.shape[1]).unsqueeze(0))

    layer = VulkanDecoderLayer(cfg, 0)
    layer.load_state_dict(qm.layers[0].state_dict(), strict=True)
    layer.upload_weights()

    # time the full layer
    with torch.no_grad():
        layer(x, position_embeddings=pe, attention_mask=None,
              past_key_values=None, cache_position=torch.arange(ids.shape[1]))
    ts = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            layer(x, position_embeddings=pe, attention_mask=None,
                  past_key_values=None, cache_position=torch.arange(ids.shape[1]))
        ts.append(time.time() - t0)
    ts.sort()
    layer_ms = ts[2] * 1000
    print(f"full layer: {layer_ms:.1f} ms")

    # time just the host round-trips by downloading a large buffer repeatedly
    big = tvk.VkBuffer(4096 * 4, "device")
    t0 = time.time()
    for _ in range(20):
        big.download((4096,))
    dl_ms = (time.time() - t0) / 20 * 1000
    print(f"one host download of 4096 floats: {dl_ms:.3f} ms")

    # count round trips in a layer
    n_dl = 8   # rmsnorm, q, q_norm, k, k_norm, v, attention out, mlp chain
    print(f"~{n_dl} host round trips per layer -> ~{n_dl * dl_ms:.1f} ms of the "
          f"{layer_ms:.1f} ms")


if __name__ == "__main__":
    main()
