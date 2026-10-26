"""
Integration of the heterogeneous CPU/Vulkan backbone into the original
`HybridGenerationEngine`.

This is a *subclass*, not a rewrite. Every invariant of the original engine is
preserved by inheritance:

  * `<latent>` entry and `</latent>` exit anchors
  * `min_latent_steps` / `max_latent_steps` / forced exits
  * the exit-anchor readout decision (`argmax(logits) == </latent>`)
  * sampling masks (`blocked_token_sets`), repetition penalty, top-k/top-p
  * think-state transitions, EOS handling, tool-call recognition
  * generation budgets, `kv_ledger`, `feed_queue`, working memory
  * batch-1 lockstep stepping and the tool barrier

What changes is only where the backbone forward runs and which device produces
the continuous latent input. The original `_lockstep_forward` and `_decode_step`
are reused verbatim; `forward_step` is intercepted so the hybrid backbone is
called instead of `Qwen3ForCausalLM`.

KV cache: the engine's `expand_cache_batch` and `forward_step` both assume the
cache tensors are ordinary tensors reachable through `vars(layer)` and
`outputs.past_key_values`. Keeping the original `DynamicCache` on CPU satisfies
that contract exactly.
"""
from __future__ import annotations

import torch

from backend.hybrid_offload import (
    DeviceStats, HybridBackbone, Placement, bytes_per_layer, vram_budget_layers,
)
from backend.hybridlink_offload import LatentInjector


def make_hybrid_model(hf_model, placement: Placement) -> HybridBackbone:
    """Wrap a `Qwen3ForCausalLM`'s inner `Qwen3Model` in a hybrid backbone.

    The returned module keeps `lm_head`, `embed_tokens`, `norm`, `rotary_emb`,
    `config` and `layers` so the engine's `self.backbone.get_input_embeddings()`
    and every original call site still work.
    """
    class _W(torch.nn.Module):
        pass

    bb = HybridBackbone(hf_model.model, placement)
    hf_model.model = bb
    return hf_model


def install_forward_step():
    """Replace `sampling.forward_step` with a version that runs the hybrid backbone.

    The engine calls `forward_step(self.backbone, ...)` from `hybrid_agent.sampling`.
    We patch that function so both the original engine and any subclass pick up
    the heterogeneous backbone without changing their own code.
    """
    from hybrid_agent import sampling

    original = sampling.forward_step

    def patched(lm, *, input_embeds, past_key_values, attention_mask,
                position_ids, cache_position):
        t0 = torch.cuda.Event if False else None   # placeholder, timing below
        import time
        t = time.time()
        outputs = lm(
            inputs_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            stats=getattr(lm, "_tvk_stats", None),
        )
        if getattr(lm, "_tvk_stats", None) is not None:
            lm._tvk_stats.cpu_time_s += time.time() - t
        return (outputs.past_key_values, outputs.logits[:, -1, :],
                outputs.hidden_states[-1][:, -1, :])

    sampling.forward_step = patched
    return original


def uninstall_forward_step(original):
    from hybrid_agent import sampling
    sampling.forward_step = original


class HybridGenerationEngineVk:
    """Marker mixin: the original engine, plus hybrid-device plumbing.

    Not a subclass of `HybridGenerationEngine`; instead it is applied by
    `make_engine_class()`, which subclasses the original so the entire
    generation control logic is inherited unchanged.
    """


def make_engine_class(base_engine_cls, placement: Placement,
                      latent_path: str = "cpu"):
    """Build an engine class that runs the original controller on the hybrid backbone.

    The returned class subclasses `base_engine_cls`; only `__init__` is extended
    to install the Vulkan HybridLink path and to attach device stats. All of the
    original generation semantics are inherited.
    """

    class _Engine(base_engine_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tvk_stats = DeviceStats()
            self.tvk_placement = placement
            # Latent input path: HybridLink on CPU (A) or Vulkan (B)
            self.latent_injector = LatentInjector(
                self.model.hybridlink, self.model.backbone.config.hidden_size,
                path=latent_path,
            )

        # -- hook: latent inputs come from the chosen HybridLink path --------
        def _latent_inputs(self, hidden_rows: torch.Tensor) -> torch.Tensor:
            embed_w = self.embed_layer.weight
            out = self.latent_injector(hidden_rows)
            return out.to(dtype=embed_w.dtype)

    return _Engine
