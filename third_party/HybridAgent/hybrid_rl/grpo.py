"""Group-normalized advantages and the clipped policy objective over sampled text positions,
with the importance ratio rho = pi_theta / pi_theta_old from the replayed and recorded log-probs.
"""
from __future__ import annotations

import torch

from hybrid_rl.config import ADV_EPS, CLIP_EPS, DUAL_CLIP_C, LOG_RATIO_CLIP


def group_advantages(rewards: list[float], eps: float = ADV_EPS) -> tuple[list[float], bool]:
    """(r - mean) / (std + eps) within one prompt group; flags zero-variance groups."""
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() == 0:
        return [], True
    if values.numel() < 2:
        return [0.0] * int(values.numel()), True
    mean, std = float(values.mean()), float(values.std())
    advantages = (values - mean) / (std + eps)
    return advantages.tolist(), bool(std < eps)


def importance_ratio(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    if new_logp.shape != old_logp.shape:
        raise ValueError(f"log-prob shape mismatch: {new_logp.shape} vs {old_logp.shape}")
    log_ratio = (new_logp - old_logp).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    return log_ratio.exp()


def ppo_clip_surrogate(
    new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """min(rho A, clip(rho) A) per token, with the negative-advantage branch bounded at c A."""
    ratio = importance_ratio(new_logp, old_logp)
    adv = torch.as_tensor(float(advantage), device=ratio.device, dtype=ratio.dtype)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
    surrogate = torch.minimum(unclipped, clipped)
    if advantage < 0:
        surrogate = torch.maximum(surrogate, torch.full_like(surrogate, DUAL_CLIP_C * float(advantage)))
    return surrogate, ratio


def policy_loss(
    text_logp: torch.Tensor, old_text_logp: torch.Tensor, advantage: float, n_tokens_global: int,
) -> tuple[torch.Tensor, dict]:
    """One trajectory's share of the objective, normalized by the global token count of
    its minibatch so that summed gradients equal the gradient of the global token mean."""
    if text_logp.numel() == 0:
        return text_logp.sum() * 0.0, {"n_text": 0, "clip_count": 0, "ratio_sum": 0.0}
    surrogate, ratio = ppo_clip_surrogate(text_logp, old_text_logp, advantage)
    loss = -surrogate.sum() / max(int(n_tokens_global), 1)
    with torch.no_grad():
        stats = {
            "n_text": int(text_logp.numel()),
            "clip_count": int(((ratio - 1.0).abs() > CLIP_EPS).sum()),
            "ratio_sum": float(ratio.sum()),
        }
    return loss, stats
