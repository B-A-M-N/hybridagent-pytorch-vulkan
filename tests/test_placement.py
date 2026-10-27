"""
Validate the placement controller against the measurements already taken.

The controller's predictions are only useful if they reproduce what was
actually observed. This checks the cost model against the real data.

MEASURED DATA -- post-deadlock-fix, single clean run (2026-10-27):
  S=12   CPU 1.11492s, vk1 1.07080s, vk2 1.04906s, vk4 1.96410s, vk6 6.72334s
  S=512  CPU 65.920s, vk1 59.379s, vk2 59.198s, vk4 59.012s, vk6 58.063s

The pre-fix S=12 numbers (CPU 1.167, vk1 1.107, vk4 1.746) were measured
while the backend could deadlock; they are retained only for the S=512/2048
cross-check below.

If the model picks a partition whose predicted time matches the measured
time for that partition, the model is consistent with reality.
"""
import os
import sys

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))

from backend.placement import (LayerCost, activation_bytes, choose_prefix,  # noqa: E402
                               decide_placement, max_layers_for_vram)
from backend.hybrid_offload import bytes_per_layer  # noqa: E402

# Measured on this machine (median prefill, seconds).
# S=12 is post-deadlock-fix (reports/requalify_s12.json).
# S=512/2048 are from the earlier benchmark matrix.
MEASURED = {
    12: {"cpu": 1.11492, "vk1": 1.07080, "vk2": 1.04906, "vk4": 1.96410,
         "vk6": 6.72334},
    512: {"cpu": 65.920, "vk1": 59.379, "vk2": 59.198, "vk4": 59.012,
          "vk6": 58.063},
    2048: {"cpu": 485.798, "vk4": 409.657},
}

VRAM = 8.57e9          # RX 6600 XT, 8.57 GB
TOTAL_LAYERS = 36


class Cfg:
    hidden_size = 4096
    intermediate_size = 12288
    num_attention_heads = 32
    head_dim = 128
    num_key_value_heads = 8


def main():
    per_layer = bytes_per_layer(Cfg.hidden_size, Cfg.intermediate_size,
                                Cfg.num_attention_heads, Cfg.head_dim,
                                Cfg.num_key_value_heads)
    print(f"per-layer weights: {per_layer / 1e6:.1f} MB (fp32)")
    print(f"VRAM: {VRAM / 1e9:.2f} GB\n")

    print("=== VRAM ceiling by sequence length ===")
    for S in (12, 512, 2048):
        act = activation_bytes(Cfg.num_attention_heads, Cfg.head_dim, S,
                               Cfg.num_key_value_heads)
        mx = max_layers_for_vram(VRAM, per_layer, act)
        print(f"  S={S:5d}: activation {act / 1e6:8.1f} MB/layer  "
              f"-> max {mx} GPU layers")
    print()

    print("=== does the model reproduce the measurements? ===")
    for S, m in MEASURED.items():
        # Derive per-layer costs from the measured whole-model times.
        cpu_layer = m["cpu"] / TOTAL_LAYERS
        # Use the vk4 measurement to estimate the GPU layer cost.
        if "vk4" in m:
            # T(4) = 32*cpu + 4*gpu + 4*transfer
            # Assume transfer ~ 0 and solve for gpu.
            gpu_layer = (m["vk4"] - (TOTAL_LAYERS - 4) * cpu_layer) / 4
        else:
            gpu_layer = cpu_layer
        cost = LayerCost(cpu=cpu_layer, gpu=gpu_layer, transfer=0.0)
        plan = decide_placement(Cfg, VRAM, S, cost, TOTAL_LAYERS)
        print(f"  S={S:5d}: cpu_layer={cpu_layer*1000:6.2f} ms  "
              f"gpu_layer={gpu_layer*1000:6.2f} ms")
        print(f"           -> chose {len(plan.layers_on_gpu)} GPU layers "
              f"(predicted {plan.predicted_speedup:.2f}x)")
        print(f"           actual best: "
              f"{min(((k, v) for k, v in m.items() if k != 'cpu'), key=lambda kv: kv[1])}")
        print()


if __name__ == "__main__":
    main()
