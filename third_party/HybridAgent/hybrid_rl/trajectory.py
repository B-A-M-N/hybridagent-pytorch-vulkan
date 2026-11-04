"""Recorded hybrid trajectory tau = (z, y, o): latent inputs z at latent
slots, sampled text tokens y with their rollout log-probs and sampling support, and tool
observations o. Only text actions carry policy likelihood (Hybrid Policy Factorization).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class TextAction:
    pos: int
    token_id: int
    logp: float | torch.Tensor            # device scalar during rollout, float after finish
    blocked_ids: tuple[int, ...] = field(default_factory=tuple)   # hard support mask at sampling


@dataclass(eq=False)
class HybridTrajectory:
    prompt_len: int = 0
    slot_tokens: list[int | None] = field(default_factory=list)      # None = latent slot
    latent_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    observation_slots: set[int] = field(default_factory=set)
    text_actions: list[TextAction] = field(default_factory=list)
    output: str = ""
    prompt_ends_in_think: bool = False
    n_text_tokens: int = 0
    n_latent_steps: int = 0
    n_latent_spans: int = 0
    n_tool_calls: int = 0
    turns: int = 0
    response_len: int = 0
    cropped_slots: int = 0
    max_sequence_tokens: int = 0
    valid: bool = True
    invalid_reason: str = ""
    reward: float = 0.0
    advantage: float = 0.0

    def __len__(self) -> int:
        return len(self.slot_tokens)
