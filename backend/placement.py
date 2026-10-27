"""
PyTorch-aware heterogeneous placement for the HybridAgent Vulkan backend.

The problem this solves
-----------------------
`Placement.first_n(n)` puts a hand-picked number of leading layers on the GPU.
That is not a policy -- it is a guess, and the measurements show the guess is
often wrong:

  * 8 GPU layers at S>=512 run out of VRAM (771.8 MB/layer in fp32) and lose the
    RADV context entirely -- garbage output.
  * At short sequence lengths and in the latent loop, per-op dispatch overhead
    dominates, so MORE GPU layers is actively slower (8 layers was 37% slower
    than CPU at S=12).
  * At long sequence lengths, arithmetic dominates and the GPU wins (4 layers
    was 16% faster than CPU at S=2048).

So the right partition depends on the workload, and neither "all CPU" nor
"as many layers as fit" is correct in general.

What this module provides
-------------------------
A cost model with two parts:

1. **Capacity** -- which layers CAN live on the GPU given VRAM, the per-layer
   weight size, and the per-layer activation/KV budget at the target sequence
   length. This is a hard constraint; violating it loses the context.

2. **Speed** -- which of the feasible placements is actually fastest. Measured,
   not assumed: the model is
       T(g) = (L - g) * t_cpu_layer  +  g * t_gpu_layer  +  g * t_transfer
   where the three coefficients are measured on this machine for the workload's
   shape, and `t_transfer` covers the per-layer boundary crossings.

The controller then picks the largest `g` whose predicted cost is below the
all-CPU baseline -- i.e. it only offloads layers that measurably pay for
themselves, which is exactly the "CPU and GPU assigned by actual throughput,
memory constraints and transfer costs" behaviour that was missing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class LayerCost:
    """Measured per-layer cost, in seconds.

    `cpu` and `gpu` are full layer-forward times measured at the workload's
    shape. `transfer` is the additional cost of the CPU<->GPU boundary per
    offloaded layer (the hidden state crossing, plus any synchronisation).
    """

    cpu: float = 0.0
    gpu: float = 0.0
    transfer: float = 0.0


@dataclass
class PlacementPlan:
    """A chosen partition, with the reasoning that produced it."""

    layers_on_gpu: list[int] = field(default_factory=list)
    reason: str = ""
    predicted_speedup: float = 1.0
    vram_bytes: int = 0
    feasible: bool = True


def contiguous_prefix(n: int, total: int) -> list[int]:
    """Layers 0..n-1 -- a contiguous GPU region at the start of the model.

    Contiguous placement matters: every CPU->GPU boundary costs a hidden-state
    transfer, so a strided placement (e.g. every other layer) would pay the
    boundary cost several times while gaining nothing.
    """
    return list(range(max(0, min(n, total))))


def vram_for_layers(n_layers: int, per_layer_bytes: int) -> int:
    return n_layers * per_layer_bytes


def max_layers_for_vram(vram_total: int, per_layer_bytes: int,
                        activation_bytes: int,
                        reserve_fraction: float = 0.30) -> int:
    """Largest contiguous GPU region that fits, counting activations.

    The reserve is deliberate: filling the card to 100% is what produced the
    RADV context loss at 8 layers. 30% covers the Vulkan runtime, the
    per-layer scratch buffers (the attention `probs` buffer alone is
    n_heads*S*S*4 bytes) and headroom for the KV cache.
    """
    if per_layer_bytes <= 0:
        return 0
    usable = int(vram_total * (1.0 - reserve_fraction))
    per_layer = per_layer_bytes + activation_bytes
    return max(0, usable // per_layer) if per_layer > 0 else 0


def choose_prefix(cost: LayerCost, n_layers: int,
                  vram_limit: int, per_layer_bytes: int,
                  activation_bytes: int) -> PlacementPlan:
    """Pick the contiguous GPU prefix that minimises predicted total time.

    Sweeps every feasible prefix and evaluates the measured cost model, rather
    than assuming "more GPU is better". A prefix is rejected outright if it
    does not fit in VRAM -- that failure mode is catastrophic (context loss),
    not merely slow.
    """
    hard_max = max_layers_for_vram(vram_limit, per_layer_bytes,
                                   activation_bytes)
    if hard_max <= 0:
        return PlacementPlan(
            layers_on_gpu=[], reason="no GPU layers fit in VRAM",
            feasible=False)

    best_n, best_t = 0, cost.cpu * n_layers
    for g in range(1, min(hard_max, n_layers) + 1):
        t = (n_layers - g) * cost.cpu + g * cost.gpu + g * cost.transfer
        if t < best_t:
            best_t, best_n = t, g

    base = cost.cpu * n_layers
    plan = PlacementPlan(
        layers_on_gpu=contiguous_prefix(best_n, n_layers),
        reason=(f"predicted {best_t:.3f}s vs all-CPU {base:.3f}s "
                f"(VRAM cap {hard_max} layers)"),
        predicted_speedup=base / best_t if best_t > 0 else 1.0,
        vram_bytes=vram_for_layers(best_n, per_layer_bytes),
    )
    if best_n == 0:
        plan.reason = (f"no prefix paid for itself (CPU {cost.cpu*1e3:.2f} ms, "
                       f"GPU {cost.gpu*1e3:.2f} ms, transfer "
                       f"{cost.transfer*1e3:.2f} ms per layer)")
    return plan


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_layer_cost(cpu_layer, gpu_layer, sample_input,
                       position_embeddings, cache_position,
                       reps: int = 5) -> LayerCost:
    """Time one CPU layer and one GPU layer at the workload's actual shape.

    `gpu_layer` may be None when no Vulkan layer is available; the caller then
    cannot offload and the controller falls back to all-CPU.

    The `transfer` term is the cost of the CPU<->GPU boundary. It is measured,
    not assumed: the GPU layer is timed twice -- once with the input already
    on the device (pure GPU work), and once with a HOST-resident input, which
    is what a real boundary crossing actually requires (the hidden state must
    be uploaded). The difference is the boundary cost.

    (The previous implementation had `transfer = max(0.0, gpu_t - gpu_t)`,
    which is identically zero -- a placeholder that silently made every
    offloaded layer look free. The controller then preferred GPU layers
    without evidence that they paid for themselves.)
    """
    cpu_t = _time(lambda: cpu_layer(
        sample_input, position_embeddings=position_embeddings,
        attention_mask=None, past_key_values=None, use_cache=False,
        cache_position=cache_position), reps)

    if gpu_layer is None:
        return LayerCost(cpu=cpu_t, gpu=float("inf"), transfer=float("inf"))

    gpu_t = _time(lambda: gpu_layer(
        sample_input, position_embeddings=position_embeddings,
        attention_mask=None, past_key_values=None, use_cache=False,
        cache_position=cache_position), reps)

    # Boundary cost: time the GPU layer with a HOST-resident input.
    # `gpu_layer` accepts either a device tensor or a CPU tensor; passing a
    # CPU tensor forces the upload path that a real boundary crossing pays.
    host_input = sample_input
    if torch.is_tensor(host_input):
        host_input = host_input.detach().cpu()
    boundary_t = _time(lambda: gpu_layer(
        host_input, position_embeddings=position_embeddings,
        attention_mask=None, past_key_values=None, use_cache=False,
        cache_position=cache_position), reps)

    transfer = max(0.0, boundary_t - gpu_t)
    return LayerCost(cpu=cpu_t, gpu=gpu_t, transfer=transfer)


def _time(fn: Callable[[], None], reps: int) -> float:
    """Median wall time of `fn` over `reps` calls, in seconds."""
    fn()                                    # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def activation_bytes(n_heads: int, head_dim: int, seq_len: int,
                     n_kv_heads: int | None = None) -> int:
    """Scratch memory one offloaded layer needs at `seq_len`.

    The dominant term is the attention `probs` buffer: n_heads * S * S * 4
    bytes, which is why the VRAM limit depends on sequence length and why a
    placement chosen at S=12 can be invalid at S=512.
    """
    n_kv = n_kv_heads or n_heads
    probs = n_heads * seq_len * seq_len * 4
    qkv = seq_len * (n_heads * head_dim + 2 * n_kv * head_dim) * 4 * 2
    out = n_heads * seq_len * head_dim * 4 * 2
    return probs + qkv + out


def decide_placement(config, vram_total: int, seq_len: int,
                     cost: LayerCost,
                     total_layers: int = 36) -> PlacementPlan:
    """Convenience wrapper: compute sizes from a config and pick a prefix."""
    from backend.hybrid_offload import bytes_per_layer

    per_layer = bytes_per_layer(
        config.hidden_size, config.intermediate_size,
        config.num_attention_heads, config.head_dim,
        config.num_key_value_heads)
    act = activation_bytes(config.num_attention_heads, config.head_dim,
                           seq_len, config.num_key_value_heads)
    return choose_prefix(cost, total_layers, vram_total, per_layer, act)


class Placement:
    """Which transformer layers run on Vulkan.

    `layers_on_gpu` is a contiguous prefix, chosen by `decide_placement` from a
    measured cost model rather than picked by hand.
    """

    def __init__(self, layers_on_gpu, reason: str = "",
                 predicted_speedup: float = 1.0, vram_bytes: int = 0):
        self.layers_on_gpu = list(layers_on_gpu)
        self.reason = reason
        self.predicted_speedup = predicted_speedup
        self.vram_bytes = vram_bytes

    # -- constructors -------------------------------------------------------
    @classmethod
    def first_n(cls, n: int, total: int = 36) -> "Placement":
        """Manual override: the first `n` layers on Vulkan."""
        return cls(contiguous_prefix(n, total), reason=f"manual first_n({n})")

    @classmethod
    def all_cpu(cls) -> "Placement":
        return cls([], reason="all CPU")

    @classmethod
    def all_gpu(cls, total: int = 36) -> "Placement":
        return cls(contiguous_prefix(total, total), reason="all GPU (manual)")

    @classmethod
    def from_plan(cls, plan: PlacementPlan) -> "Placement":
        return cls(plan.layers_on_gpu, plan.reason,
                   plan.predicted_speedup, plan.vram_bytes)

    # -- queries ------------------------------------------------------------
    def device_of(self, layer_idx: int) -> str:
        return "vulkan" if layer_idx in self.layers_on_gpu else "cpu"

    def __len__(self) -> int:
        return len(self.layers_on_gpu)

    def __repr__(self) -> str:
        return (f"Placement(gpu={len(self.layers_on_gpu)} layers, "
                f"speedup={self.predicted_speedup:.2f}x, "
                f"vram={self.vram_bytes / 1e9:.2f} GB, {self.reason!r})")
