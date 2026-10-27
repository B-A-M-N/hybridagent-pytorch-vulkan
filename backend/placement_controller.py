"""
Placement controller: automatic CPU/GPU layer assignment based on sequence length.

Status: RECONCILED with `backend/placement.py` (2026-10-27).

This module previously carried its own hardcoded policy -- a crossover
threshold of 32 tokens and a hand-written VRAM table -- which duplicated the
cost model in `backend/placement.py` and had gone stale: the crossover was
measured while the Vulkan backend could deadlock, so the numbers described a
pathology rather than the hardware.

The controller now DELEGATES to the measured cost model in
`backend/placement.py`:

  * the VRAM ceiling is computed from the real per-layer weight size and the
    activation budget at the target sequence length, not a hardcoded table;
  * the number of GPU layers is chosen by the measured cost model, which
    only offloads layers that measurably pay for themselves.

`backend/placement.py` remains the single source of truth for the policy;
this module is the thin, sequence-length-keyed entry point that
`backend/profile_full_model.py` uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from backend.hybrid_offload import HybridBackbone, Placement


def _model_config():
    """The HybridAgent-8B model geometry, as a plain object.

    Kept as a function so importing this module does not load the model.
    """
    class _C:
        hidden_size = 4096
        intermediate_size = 12288
        num_attention_heads = 32
        head_dim = 128
        num_key_value_heads = 8
    return _C


@dataclass
class PlacementConfig:
    """Configuration for the placement controller.

    The previous version of this class hardcoded a crossover of 32 tokens and
    a per-sequence-length VRAM table. Both were measured while the Vulkan
    backend could deadlock, so they described a pathology, not the hardware,
    and they duplicated the cost model in `backend/placement.py`.

    The policy now lives in `backend/placement.py`. What remains here is:
      * the VRAM ceiling, computed from the real per-layer weight size and the
        activation budget at the requested sequence length, and
      * the measured per-layer cost coefficients, which the caller supplies.
    """
    # VRAM of the target GPU, in bytes.
    vram_total: int = 8.57e9
    # Fraction of VRAM held back for the Vulkan runtime, per-layer scratch
    # buffers (the attention `probs` buffer alone is n_heads*S*S*4 bytes) and
    # KV-cache headroom. Filling the card to 100% is what produced the RADV
    # context loss at 8 GPU layers.
    vram_reserve_fraction: float = 0.30
    # Fallback number of GPU layers when no measured cost is available.
    # Used only if the caller cannot supply one.
    default_gpu_layers: int = 2

    def max_layers_for_seq(self, seq_len: int, total_layers: int = 36,
                           config=None) -> int:
        """Max GPU layers that fit in VRAM at this sequence length."""
        from backend.hybrid_offload import bytes_per_layer
        from backend.placement import activation_bytes, max_layers_for_vram

        if config is None:
            config = _model_config()
        per_layer = bytes_per_layer(
            config.hidden_size, config.intermediate_size,
            config.num_attention_heads, config.head_dim,
            config.num_key_value_heads)
        act = activation_bytes(config.num_attention_heads, config.head_dim,
                               seq_len, config.num_key_value_heads)
        return max_layers_for_vram(self.vram_total, per_layer, act,
                                   self.vram_reserve_fraction)


class PlacementController:
    """
    Decides layer placement for a given input.

    Policy:
    - The VRAM ceiling is a HARD constraint: exceeding it loses the Vulkan
      context entirely, so it is checked first.
    - Within that ceiling, the number of GPU layers comes from the measured
      cost model in `backend/placement.py`, which only offloads layers that
      measurably pay for themselves.
    - If no measured cost is available, the controller falls back to
      `default_gpu_layers` (capped by VRAM) rather than guessing from a
      hardcoded sequence-length threshold.
    """

    def __init__(self, config: Optional[PlacementConfig] = None,
                 cost=None, model_config=None):
        self.config = config or PlacementConfig()
        # Measured per-layer cost (cpu/gpu/transfer seconds). Optional.
        self.cost = cost
        self.model_config = model_config or _model_config()

    def decide(self, seq_len: int, total_layers: int = 36) -> Placement:
        """Return a Placement for the given sequence length."""
        from backend.placement import LayerCost, choose_prefix,             decide_placement

        max_allowed = self.config.max_layers_for_seq(
            seq_len, total_layers, self.model_config)

        if self.cost is not None:
            plan = decide_placement(self.model_config, self.config.vram_total,
                                    seq_len, self.cost, total_layers)
            return Placement.from_plan(plan)

        # No measured cost: use the fallback, still capped by VRAM.
        gpu_layers = min(self.config.default_gpu_layers, max_allowed)
        return Placement.first_n(gpu_layers, total_layers)

    def make_backbone(self, qwen_model, seq_len: int) -> HybridBackbone:
        """Create a HybridBackbone with the decided placement."""
        placement = self.decide(seq_len, len(qwen_model.layers))
        return HybridBackbone(qwen_model, placement)


# Singleton for convenience
_default_controller: Optional[PlacementController] = None


def get_controller() -> PlacementController:
    global _default_controller
    if _default_controller is None:
        _default_controller = PlacementController()
    return _default_controller


def decide_placement(seq_len: int, total_layers: int = 36) -> Placement:
    """Convenience function."""
    return get_controller().decide(seq_len, total_layers)


def make_backbone(qwen_model, seq_len: int) -> HybridBackbone:
    """Convenience function."""
    return get_controller().make_backbone(qwen_model, seq_len)


if __name__ == "__main__":
    # Quick self-test: VRAM ceiling per sequence length, and the fallback
    # partition when no measured cost is supplied.
    controller = PlacementController()
    print("VRAM ceiling and fallback partition by sequence length:")
    for S in [12, 32, 64, 128, 256, 512, 1024, 2048]:
        mx = controller.config.max_layers_for_seq(S)
        p = controller.decide(S)
        print(f"  S={S:5d} -> vram_max={mx}  fallback={p.num_gpu_layers} GPU layers")