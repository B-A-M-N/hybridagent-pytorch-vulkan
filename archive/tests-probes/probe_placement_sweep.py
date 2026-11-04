"""
Does a per-layer GPU win survive to the whole model?

The per-layer measurement says the GPU layer is 14% faster at S=12. The
whole-model benchmark says 8 GPU layers was 37% SLOWER than all-CPU at S=12.
Both are real measurements. This test finds out why they disagree, by measuring
whole-model time as a function of GPU-layer count at a short sequence length.

If the cost model is right, T(g) should be linear in g:
    T(g) = (L-g)*t_cpu + g*t_gpu + g*t_transfer
so a regression of measured T(g) against g recovers t_transfer -- the per-layer
boundary cost that the isolated layer measurement cannot see.
"""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    for S in (12, 512):
        ids = torch.randint(0, 1000, (1, S))
        print(f"\n=== S={S} ===")
        print(f"{'gpu_layers':>10} {'total_s':>10} {'per_layer_ms':>13}")

        # all-CPU baseline
        with torch.no_grad():
            t0 = timeit(lambda: lm(input_ids=ids, use_cache=True))
        print(f"{0:>10} {t0:>10.3f} {t0/36*1000:>13.2f}")

        points = []
        for ng in (1, 2, 4, 6):
            bb = HybridBackbone(qm, Placement.first_n(ng, len(qm.layers)))
            with torch.no_grad():
                t = timeit(lambda: bb(input_ids=ids, use_cache=True))
            print(f"{ng:>10} {t:>10.3f} {t/36*1000:>13.2f}")
            points.append((ng, t))

        # Regress T(g) = a + b*g to recover the marginal cost of one GPU layer.
        if len(points) >= 2:
            n = len(points)
            sx = sum(p[0] for p in points)
            sy = sum(p[1] for p in points)
            sxx = sum(p[0] ** 2 for p in points)
            sxy = sum(p[0] * p[1] for p in points)
            b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            a = (sy - b * sx) / n
            print(f"  marginal cost per GPU layer: {b*1000:.2f} ms")
            print(f"  (negative = GPU layers are a net win)")


if __name__ == "__main__":
    main()
