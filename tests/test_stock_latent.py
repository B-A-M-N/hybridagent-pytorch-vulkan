"""
DECISIVE D3 TEST: does the STOCK, UNMODIFIED HybridAgent engine enter latent
mode on the authors' own reference prompts?

This is comparison #1 from the review methodology. If the stock engine enters
latent mode here, then the earlier "model doesn't do latent reasoning" finding
was an artifact of the prompts I chose, not a property of the checkpoint.

Runs the released `HybridGenerationEngine` with its own SYSTEM_PROMPT and the
released checkpoint, and reports latent spans / steps / end reason.
"""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))

from hybrid_agent import (  # noqa: E402
    SYSTEM_PROMPT, HybridGenerationEngine, load_hybrid_agent,
)
from hybrid_agent.tools import build_tool_executor  # noqa: E402
from hybrid_agent.tools.executor import ToolExecutor  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")

QUESTIONS = [
    "How many trailing zeros does 100! have?",
    "At the speed of sound in air at 20 degrees Celsius, how many seconds does "
    "sound take to travel 10 kilometers? Round to the nearest second.",
    "How many minutes does light take to travel from the Sun to the Earth? "
    "Round to the nearest minute.",
]


class Recorder:
    """Wraps the tool executor and records every action."""

    def __init__(self, python_timeout=60, **kw):
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
        return ToolExecutor.extract_content(t, tag)


def main():
    torch.manual_seed(0)
    print("loading stock HybridAgent-8B-Agentic (CPU, bf16) ...", flush=True)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    tools = Recorder(python_timeout=60)
    print(f"system prompt: {len(SYSTEM_PROMPT)} chars, "
          f"mentions <latent>: {'<latent>' in SYSTEM_PROMPT}\n", flush=True)

    for q in QUESTIONS:
        eng = HybridGenerationEngine(
            model, tok, torch.device("cpu"), tools, SYSTEM_PROMPT, q,
            batch_size=1,
            max_response_tokens=800,
            max_tool_calls=0, max_search_calls=0, max_python_calls=0,
            terminate_on_tool_cap=True, max_turns=8,
            temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
            min_latent_steps=4, max_latent_steps=200,
            record_latent_readout=True,
        )
        t0 = time.time()
        eng.run(torch.Generator(device="cpu").manual_seed(0))
        wall = time.time() - t0

        row = eng.rows[0]
        res = eng.row_result(row)
        print(f"Q: {q[:70]}")
        print(f"   wall {wall:.0f}s | latent_steps {res['latent_steps']} | "
              f"latent_spans {res['latent_spans']} | "
              f"anchor_exits {res['anchor_exits']} | "
              f"forced_exits {res['forced_exits']}")
        print(f"   text tokens {res['generation_steps'] - res['latent_steps']} | "
              f"end_reason {res['end_reason']}")
        if res.get("latent_readouts"):
            for s in res["latent_readouts"]:
                print(f"   span: steps={s['steps']} exit={s['exit']} "
                      f"pseudo={s['pseudo_text'][:90]!r}")
        print(f"   output[:200]={res['output'][:200]!r}")
        print(flush=True)


if __name__ == "__main__":
    main()
