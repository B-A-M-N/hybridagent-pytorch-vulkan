"""
Measure per-layer CPU and GPU cost directly, then validate the placement model.

The earlier version of this test derived the GPU layer cost algebraically from
one whole-model measurement, which produced absurd numbers (177 ms/layer) and
wrong predictions. This measures each layer's forward time directly at the
workload's shape, which is what the controller actually needs.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))

from backend.placement import (activation_bytes, decide_placement,  # noqa: E402
                               max_layers_for_vram, LayerCost)
from backend.hybrid_offload import VulkanDecoderLayer, bytes_per_layer  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
VRAM = 8.57e9
TOTAL_LAYERS = 36


def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t = torch.cuda.Event if False else None
        import time
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    cfg = qm.config
    cpu_layer = qm.layers[0]

    per_layer = bytes_per_layer(cfg.hidden_size, cfg.intermediate_size,
                                cfg.num_attention_heads, cfg.head_dim,
                                cfg.num_key_value_heads)
    print(f"per-layer weights: {per_layer / 1e6:.1f} MB")

    vk = VulkanDecoderLayer(cfg, 0)
    vk.load_state_dict(cpu_layer.state_dict(), strict=True)

    print(f"\n{'S':>6} {'cpu/layer':>11} {'gpu/layer':>11} {'ratio':>7} "
          f"{'vram_max':>9} {'chosen':>7}")
    print("-" * 60)

    for S in (12, 512, 2048):
        ids = torch.randint(0, 1000, (1, S))
        emb = qm.embed_tokens(ids)
        pe = qm.rotary_emb(emb, torch.arange(S).unsqueeze(0))
        cp = torch.arange(S)

        with torch.no_grad():
            t_cpu = timeit(lambda: cpu_layer(
                emb, position_embeddings=pe, attention_mask=None,
                past_key_values=None, use_cache=False, cache_position=cp))
            t_gpu = timeit(lambda: vk(
                emb, position_embeddings=pe, attention_mask=None,
                past_key_values=None, cache_position=cp))

        act = activation_bytes(cfg.num_attention_heads, cfg.head_dim, S,
                               cfg.num_key_value_heads)
        hard_max = max_layers_for_vram(VRAM, per_layer, act)
        cost = LayerCost(cpu=t_cpu, gpu=t_gpu, transfer=0.0)
        plan = decide_placement(cfg, VRAM, S, cost, TOTAL_LAYERS)

        print(f"{S:>6} {t_cpu*1000:>10.2f}m {t_gpu*1000:>10.2f}m "
              f"{t_gpu/t_cpu:>6.2f}x {hard_max:>9} "
              f"{len(plan.layers_on_gpu):>7}")


if __name__ == "__main__":
    main()
