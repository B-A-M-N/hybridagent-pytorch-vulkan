"""Command line for Interpretable Latent Mode.

  # answer a question and open the latent thoughts
  python -m interpretable_latent_mode --model <checkpoint dir or HF id> \
      --question "How many trailing zeros does 100! have?" --output report.html

  # the same for a JSONL file of questions, or render batch-inference outputs
  python -m interpretable_latent_mode --model <ckpt> --input examples/questions.jsonl --output report.html
  python -m interpretable_latent_mode --input outputs/results.jsonl --output report.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from .render import render_html, render_text
from .trace import GENERATION_DEFAULTS, inspect, trace_from_record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Interpretable Latent Mode: read HybridAgent's latent thoughts")
    p.add_argument("--model", help="HybridAgent checkpoint: local dir or HF model id (omit to render --input records)")
    p.add_argument("--question", help="a single question")
    p.add_argument("--input", help="JSONL of questions, or of hybrid_inference.run_batch records when --model is omitted")
    p.add_argument("--output", help="HTML report path (default: print the terminal view)")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--readout_top_k", type=int, default=5, help="nearest-token candidates per latent step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="questions to process from --input (0 = all)")
    p.add_argument("--show_steps", action="store_true", help="terminal view: list the nearest token of every latent step")
    p.add_argument("--no_color", action="store_true")
    p.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    for key, value in GENERATION_DEFAULTS.items():
        if isinstance(value, bool):
            p.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value, metavar="BOOL")
        else:
            p.add_argument(f"--{key}", type=type(value), default=value)
    args = p.parse_args(argv)
    if not args.question and not args.input:
        p.error("give --question or --input")
    if not args.model and args.question:
        p.error("--question needs --model")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    traces = []
    if args.model:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        from hybrid_agent import build_tool_executor, load_hybrid_agent

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_hybrid_agent(args.model, device=device)
        tools = build_tool_executor()
        questions = [args.question] if args.question else []
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                questions += [json.loads(line)["question"] for line in f if line.strip()]
        if args.limit:
            questions = questions[: args.limit]
        settings = {key: getattr(args, key) for key in GENERATION_DEFAULTS}
        for i, question in enumerate(questions):
            print(f"[{i + 1}/{len(questions)}] {question[:80]}", file=sys.stderr, flush=True)
            traces += inspect(
                model, tokenizer, question, device=device, tool_executor=tools, readout_top_k=args.readout_top_k,
                num_samples=args.num_samples, seed=args.seed + i, model_name=os.path.basename(str(args.model).rstrip("/")),
                **settings,
            )
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if args.limit:
            records = records[: args.limit]
        traces = [trace_from_record(r) for r in records]

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(render_html(traces))
        print(f"wrote {args.output} ({len(traces)} traces)", file=sys.stderr)
        if args.open:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(args.output)}")
    else:
        for trace in traces:
            print(render_text(trace, color=not args.no_color, show_steps=args.show_steps))
            print()


if __name__ == "__main__":
    main()
