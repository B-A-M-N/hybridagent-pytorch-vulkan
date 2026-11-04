"""
Stage 1, take 3.

What take 2 established:
  * Fabrication is FIXED. With an explicit "never write <result> yourself"
    instruction, the model stopped inventing observations.
  * The model runs the REAL test suite and reads the REAL traceback.
  * The model produced the CORRECT fix:
        def is_prime(n):
            if n <= 1:
                return False
            ...
  * But it emitted that code as a bare expression instead of writing the file,
    so the repository was never actually modified. Run ended `end_reason=budget`.

Take 3 targets the remaining gap: making the file write explicit and mandatory,
with a self-verification step that must pass before the final answer.

The prompt is the only thing changed. `reject_fabricated_result` stays ON, and
upstream `third_party/HybridAgent` is untouched.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))

from hybrid_agent import HybridGenerationEngine, ToolExecutor, load_hybrid_agent  # noqa: E402
from hybrid_agent.tools import build_tool_executor  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")

CODING_PROMPT = """You are a software engineer working in a Python repository.

TOOL: <python> ... </python> runs Python CODE. It is not a shell.

Rules:
- Never write <result> yourself. The system appends the real output.
- To run a command, use subprocess inside Python:
      import subprocess
      r = subprocess.run(["python", "-m", "pytest", "-q"], capture_output=True, text=True)
      print(r.stdout); print(r.stderr)
- To READ a file:
      print(open("src/calculator.py").read())
- To WRITE a file you MUST call open() with mode "w":
      open("src/calculator.py", "w").write(new_content)
  Merely printing the corrected code does NOT change the file.

Procedure (do all steps, in order):
1. Read src/calculator.py and tests/test_calculator.py.
2. Run the test suite with subprocess and read the real failure.
3. Write the fixed file with open(path, "w").write(...).
4. Re-run the test suite with subprocess and confirm it passes.
5. Reply with <answer>\\boxed{...}</answer> only after step 4 succeeded.

Do not skip step 3 or step 4. Do not describe what you would do - do it."""


class RecordingToolExecutor:
    def __init__(self, python_timeout: int = 90, **kw):
        self._inner = build_tool_executor(
            python_timeout=python_timeout, require_search=False, **kw)
        self.calls: list[tuple[str, str]] = []

    def execute(self, tag: str, content: str, **kw) -> str:
        self.calls.append((tag, content))
        print(f"\n  [TOOL {tag}] {content[:300]}", flush=True)
        out = self._inner.execute(tag, content, **kw)
        print(f"  [RESULT] {out[:300]}", flush=True)
        return out

    def get_tool(self, tag):
        return self._inner.get_tool(tag)

    def identify_tool(self, text):
        return self._inner.identify_tool(text)

    @staticmethod
    def extract_content(text, tag):
        return ToolExecutor.extract_content(text, tag)


def main() -> int:
    torch.manual_seed(0)
    print("loading HybridAgent-8B (CPU, bf16) ...", flush=True)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    print("loaded\n", flush=True)

    tools = RecordingToolExecutor(python_timeout=90)

    engine = HybridGenerationEngine(
        model, tok, torch.device("cpu"), tools,
        system_prompt=CODING_PROMPT,
        question=(
            "A test in tests/test_calculator.py is failing. Run the suite, read "
            "the failure, fix src/calculator.py by writing the file, then run the "
            "suite again and confirm it passes."
        ),
        batch_size=1,
        max_response_tokens=2500,
        max_tool_calls=10,
        max_search_calls=0,
        max_python_calls=10,
        terminate_on_tool_cap=True,
        max_turns=20,
        temperature=0.1,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        min_latent_steps=2,
        max_latent_steps=80,
        record_latent_readout=True,
    )

    t0 = time.time()
    engine.run(torch.Generator(device="cpu").manual_seed(0))
    wall = time.time() - t0

    row = engine.rows[0]
    res = engine.row_result(row)

    print("\n" + "=" * 70)
    print("STAGE 1 (take 3) CPU CODING TEST")
    print("=" * 70)
    print(f"wall            : {wall:.1f}s")
    print(f"text tokens     : {res['generation_steps'] - res['latent_steps']}")
    print(f"latent steps    : {res['latent_steps']}")
    print(f"python calls    : {res['python_calls']}")
    print(f"turns           : {res['turns']}")
    print(f"end reason      : {res['end_reason']}")

    # verify the file on disk
    import subprocess
    proc = subprocess.run(["/usr/bin/python3", "-m", "pytest", "-q"],
                          cwd="/tmp/codex-fixrepo",
                          capture_output=True, text=True)
    tests_pass = proc.returncode == 0

    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    chk("used the <python> tool", res["python_calls"] > 0, f"({res['python_calls']})")
    chk("ran the test suite", any("pytest" in c for _, c in tools.calls))
    wrote = any('open(' in c and '"w"' in c for _, c in tools.calls)
    chk("wrote the file with open(...,'w')", wrote)
    chk("TEST SUITE ACTUALLY PASSES", tests_pass,
        "(verified on disk, not from model claims)")
    chk("genuine latent reasoning", res["latent_steps"] > 0,
        f"({res['latent_steps']} latent steps)")
    chk("produced a final answer", "<answer>" in res["output"])
    chk("did not fabricate a result", "invalid:fabricated" not in res["end_reason"])

    print(f"\n--- pytest on disk ---\n{proc.stdout[-600:]}")
    print(f"pytest returncode: {proc.returncode}")

    print("\n--- model output (first 1500 chars) ---\n")
    print(res["output"][:1500])

    n_fail = sum(1 for _, ok in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} capability checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
