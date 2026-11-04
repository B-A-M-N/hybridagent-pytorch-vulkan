"""
Separate SETUP cost from INFERENCE cost in the Vulkan offload path.

The "4000x slower" claim was a table:
    Single layer S=12 | CPU 28 ms | Vulkan (Q4) >120 s (timeout) | >4000x

Two things could make a Vulkan layer look 4000x slower than the CPU layer
while the GPU is actually fine:

  (a) The measurement included one-time setup -- uploading 771.8 MB of fp32
      weights to the device, and (if Q4) quantizing 5 weight matrices on the
      HOST in pure Python/numpy. We measured that host quantization alone is
      ~11.5 s PER LAYER.

  (b) The measurement included per-step work that is not GPU compute:
      host round trips for the KV cache, the attention mask, etc.

This script measures each phase separately with a timer around every stage,
so the 120 s can be attributed to a specific cause instead of guessed at.

Stages timed, per GPU layer:
    setup_fp32    - upload fp32 weights (one time)
    setup_q4      - quantize + upload Q4 weights (one time)
    forward_1st   - first forward after setup (includes pipeline warmup)
    forward_2nd   - steady-state forward
    steady_state  - forward minus setup, i.e. the number that should be
                    compared against the CPU layer
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def med(xs):
    return statistics.median(xs)


def main():
    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    S = 12
    ids = torch.randint(0, 1000, (1, S))
    n_gpu = 1

    print(f"device: {tvk.device_name()}")
    print(f"GPU layers under test: {n_gpu}, S={S}\n")

    # ---------------- CPU baseline ----------------
    def cpu_model():
        with torch.no_grad():
            lm(input_ids=ids, use_cache=True)
    cpu_model()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        cpu_model()
        ts.append(time.perf_counter() - t0)
    cpu_full = med(ts)
    cpu_per_layer = cpu_full / len(qm.layers)
    print(f"CPU  full 36-layer forward : {cpu_full*1000:9.1f} ms "
          f"(min {min(ts)*1000:.1f}, max {max(ts)*1000:.1f})  "
          f"({cpu_per_layer*1000:6.2f} ms/layer)")
    print()

    # ---------------- Vulkan: setup vs forward ----------------
    t0 = time.perf_counter()
    hb = HybridBackbone(qm, Placement.first_n(n_gpu, len(qm.layers)))
    print(f"build HybridBackbone (state_dict copy): {time.perf_counter()-t0:.3f} s")

    # Locate the offloaded layer object. HybridBackbone builds NEW
    # VulkanDecoderLayer wrappers, so they live in hb.layers, not qm.layers.
    gl = None
    for lyr in hb.layers:
        if type(lyr).__name__ == "VulkanDecoderLayer":
            gl = lyr
            break
    if gl is None:
        print("could not locate the offloaded layer wrapper")
        return

    print("=== per-phase timing for ONE GPU layer ===")

    # setup: fp32 weight upload
    t0 = time.perf_counter()
    gl.upload_weights()
    t_fp32 = time.perf_counter() - t0
    print(f"  setup fp32 upload        : {t_fp32:9.3f} s")

    # setup: Q4 quantize + upload (host-side)
    t0 = time.perf_counter()
    gl.upload_weights_q4()
    t_q4 = time.perf_counter() - t0
    print(f"  setup Q4 quantize+upload : {t_q4:9.3f} s   <-- HOST, pure Python/numpy")
    print(f"  setup TOTAL              : {t_fp32 + t_q4:9.3f} s")
    print()

    # forward timing, using the same layer wrapper
    with torch.no_grad():
        h = torch.randn(S, qm.config.hidden_size, dtype=torch.bfloat16)
        pos = torch.arange(S).unsqueeze(0)
        rot = qm.rotary_emb(h.float(), pos)
        cpos = torch.arange(S)

        t0 = time.perf_counter()
        gl(h, rot, None, past_key_values=None, cache_position=cpos)
        t_first = time.perf_counter() - t0
        print(f"  forward #1 (pipeline warmup): {t_first*1000:9.3f} ms")

        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            gl(h, rot, None, past_key_values=None, cache_position=cpos)
            ts.append(time.perf_counter() - t0)
        t_steady = med(ts)
        print(f"  forward #2..#6 (steady)    : {t_steady*1000:9.3f} ms")
    print()

    # ---------------- whole-model forward, after setup ----------------
    t0 = time.perf_counter()
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)
    t_model_1 = time.perf_counter() - t0
    print(f"whole model (1 GPU layer), 1st after setup: {t_model_1:9.3f} s")

    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        with torch.no_grad():
            hb(input_ids=ids, use_cache=True)
        ts.append(time.perf_counter() - t0)
    t_model = med(ts)
    print(f"whole model (1 GPU layer), steady        : {t_model:9.3f} s")
    print(f"CPU whole model                         : {cpu_full:9.3f} s")
    print()

    print("=== VERDICT ===")
    print(f"  setup amortized over N calls, per call:")
    for n in (1, 10, 100):
        print(f"    N={n:4d}: {(t_fp32 + t_q4)/n*1000 + t_model*1000:9.1f} ms "
              f"(setup {(t_fp32+t_q4)/n*1000:8.1f} ms + forward {t_model*1000:8.1f} ms)")
    print()
    print(f"  GPU layer steady-state forward : {t_steady*1000:8.2f} ms")
    print(f"  CPU layer forward              : {cpu_per_layer*1000:8.2f} ms")
    print(f"  ratio                          : {t_steady/cpu_per_layer:8.3f}x")


if __name__ == "__main__":
    main()
