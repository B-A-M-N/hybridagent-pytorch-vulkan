"""
Benchmark genuine latent generation: latent steps/sec and full-trajectory cost.

This is the HybridAgent-specific metric. Ordinary token throughput is not the
point of the architecture -- the continuous latent loop is. This measures:

  * latent steps per second (the HybridLink + latent-injection loop)
  * text tokens per second in the same run
  * the ratio between them
  * wall time for a complete trajectory
  * correctness of the latent spans (entry, exit, step counts)

Runs the ORIGINAL HybridGenerationEngine with the backbone redirected through
the Vulkan backend, and separately with the stock backbone, on the same
questions and seeds.

Usage:
    PYTHONPATH=backend/torch_vulkan_ext:third_party/HybridAgent:. \
        .venv/bin/python benchmarks/benchmark_latent.py
"""
import argparse
import json
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
    """Minimal tool-executor wrapper; tools are disabled for this benchmark."""

    def __init__(self, **kw):
        from hybrid_agent.tools import build_tool_executor
        self._i = build_tool_executor(python_timeout=60, require_search=False,
                                      **kw)
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


def run_once(model, tok, question, n_gpu, max_latent=60):
    """One engine run; returns timing and latent statistics."""
    from hybrid_agent import SYSTEM_PROMPT
    from hybrid_agent.engine import HybridGenerationEngine
    from backend.hybrid_offload import HybridBackbone, Placement

    if n_gpu > 0:
        bb = HybridBackbone(model.backbone.model,
                            Placement.first_n(n_gpu,
                                              len(model.backbone.model.layers)))
        model.backbone.model = bb

    tools = Recorder()
    eng = HybridGenerationEngine(
        model, tok, torch.device("cpu"), tools, SYSTEM_PROMPT, question,
        batch_size=1,
        max_response_tokens=400,
        max_tool_calls=0, max_search_calls=0, max_python_calls=0,
        terminate_on_tool_cap=True, max_turns=8,
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        min_latent_steps=4, max_latent_steps=max_latent,
        record_latent_readout=True,
    )
    t0 = time.perf_counter()
    eng.run(torch.Generator(device="cpu").manual_seed(0))
    wall = time.perf_counter() - t0

    res = eng.row_result(eng.rows[0])
    latent_steps = res["latent_steps"]
    text_steps = res["generation_steps"] - latent_steps
    return {
        "wall_s": wall,
        "latent_steps": latent_steps,
        "latent_spans": res["latent_spans"],
        "text_steps": text_steps,
        "latent_steps_per_s": (latent_steps / wall) if wall > 0 else 0.0,
        "text_tokens_per_s": (text_steps / wall) if wall > 0 else 0.0,
        "end_reason": res["end_reason"],
        "output": res["output"][:200],
        "pseudo": [s["pseudo_text"][:80]
                   for s in (res.get("latent_readouts") or [])][:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-layers", default="0,4")
    ap.add_argument("--max-latent", type=int, default=60)
    ap.add_argument("--out", default="benchmarks/latent_results.json")
    args = ap.parse_args()

    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")

    results = []
    for n_gpu in [int(x) for x in args.gpu_layers.split(",")]:
        for q in QUESTIONS:
            r = run_once(model, tok, q, n_gpu, args.max_latent)
            r["gpu_layers"] = n_gpu
            r["question"] = q[:50]
            results.append(r)
            print(f"gpu={n_gpu} | {q[:40]:40s} | "
                  f"latent {r['latent_steps']:3d} steps "
                  f"({r['latent_steps_per_s']:.2f}/s) | "
                  f"text {r['text_steps']:3d} tok "
                  f"({r['text_tokens_per_s']:.2f}/s) | "
                  f"wall {r['wall_s']:.0f}s | {r['end_reason']}", flush=True)
            if r["pseudo"]:
                for p in r["pseudo"]:
                    print(f"      pseudo: {p!r}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
