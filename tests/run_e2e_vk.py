"""
END-TO-END: the ORIGINAL HybridAgent PyTorch engine, running the original
Qwen3 backbone through the Vulkan backend, with genuine continuous latent
reasoning.

This is the project's actual deliverable. Nothing here reimplements generation:
the stock `HybridGenerationEngine` is subclassed so every control behaviour
(latent anchors, min/max latent steps, sampling masks, repetition penalty,
think guards, tool recognition, budgets) is inherited unchanged. Only two
things are redirected:

  1. the backbone forward runs through `HybridBackbone` (Vulkan layers + CPU
     layers), and
  2. `run_hybridlink` routes through the Vulkan HybridLink path.

Everything else -- the engine, the sampler, the tool executor, the readout -- is
the released code.

Run:
    PYTHONPATH=backend/torch_vulkan_ext:third_party/HybridAgent:. \
        .venv/bin/python tests/run_e2e_vk.py
"""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")

QUESTIONS = [
    "How many trailing zeros does 100! have?",
    "How many minutes does light take to travel from the Sun to the Earth? "
    "Round to the nearest minute.",
]


class Recorder:
    """Wraps the tool executor and records every action."""

    def __init__(self, python_timeout=60, **kw):
        from hybrid_agent.tools import build_tool_executor
        self._i = build_tool_executor(python_timeout=python_timeout,
                                       require_search=False, **kw)
        self.calls = []

    def execute(self, tag, content, **k):
        self.calls.append((tag, content))
        return self._i.execute(tag, content, **k)

    def get_tool(self, t):
        return self._i.get_tool(t)

    def identify_tool(self, t):
        return self._i.identify_tool(t)

    @staticmethod
    def extract_content(t, tag):
        from hybrid_agent.tools.executor import ToolExecutor
        return ToolExecutor.extract_content(t, tag)


def main():
    import torch_vulkan_ext as tvk
    tvk.init(0)

    from hybrid_agent import SYSTEM_PROMPT, load_hybrid_agent
    from hybrid_agent.engine import HybridGenerationEngine
    from backend.hybrid_offload import HybridBackbone, Placement
    from backend.hybridlink_offload import LatentInjector

    torch.manual_seed(0)
    print("loading HybridAgent-8B-Agentic (CPU, bf16) ...", flush=True)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")

    # ---- wrap the backbone: 4 layers on Vulkan, the rest on CPU ----------
    N_GPU = int(os.environ.get("TVK_GPU_LAYERS", "4"))
    placement = Placement.first_n(N_GPU, len(model.backbone.model.layers))
    bb = HybridBackbone(model.backbone.model, placement)
    model.backbone.model = bb
    print(f"placement: {N_GPU} Vulkan layers, "
          f"{len(model.backbone.model.layers) - N_GPU} CPU layers", flush=True)

    # ---- route run_hybridlink through the Vulkan HybridLink --------------
    injector = LatentInjector(model.hybridlink,
                              model.backbone.config.hidden_size,
                              path="vulkan")
    orig_hybridlink = model.run_hybridlink

    def run_hybridlink_vk(hidden_states, output_dtype=None):
        out = injector(hidden_states)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    model.run_hybridlink = run_hybridlink_vk

    # ---- the ORIGINAL engine, unchanged ----------------------------------
    tools = Recorder(python_timeout=60)
    for q in QUESTIONS:
        eng = HybridGenerationEngine(
            model, tok, torch.device("cpu"), tools, SYSTEM_PROMPT, q,
            batch_size=1,
            max_response_tokens=400,
            max_tool_calls=0, max_search_calls=0, max_python_calls=0,
            terminate_on_tool_cap=True, max_turns=8,
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
            min_latent_steps=4, max_latent_steps=60,
            record_latent_readout=True,
        )
        t0 = time.time()
        eng.run(torch.Generator(device="cpu").manual_seed(0))
        wall = time.time() - t0

        row = eng.rows[0]
        res = eng.row_result(row)
        print(f"\nQ: {q[:70]}")
        print(f"   wall {wall:.0f}s | latent_steps {res['latent_steps']} | "
              f"latent_spans {res['latent_spans']} | "
              f"anchor_exits {res['anchor_exits']} | "
              f"forced_exits {res['forced_exits']}")
        print(f"   text tokens {res['generation_steps'] - res['latent_steps']} | "
              f"end_reason {res['end_reason']}")
        if res.get("latent_readouts"):
            for s in res["latent_readouts"][:4]:
                print(f"   span: steps={s['steps']} exit={s['exit']} "
                      f"pseudo={s['pseudo_text'][:80]!r}")
        print(f"   output[:180]={res['output'][:180]!r}")

    model.run_hybridlink = orig_hybridlink


if __name__ == "__main__":
    main()
