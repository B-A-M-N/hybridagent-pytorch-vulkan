"""
Full-model performance qualification for the HybridAgent Vulkan backend.

Compares, on identical inputs and seeds:
  * stock CPU          -- unmodified Qwen3ForCausalLM
  * wrapped CPU        -- HybridBackbone, 0 Vulkan layers
  * wrapped Vulkan     -- HybridBackbone, 1 / 4 / 8 Vulkan layers

Measures, per configuration and per sequence length:
  * prefill wall time (median of N runs)
  * decode tokens/sec (greedy, with a persistent KV cache)
  * correctness (token agreement vs stock; max abs logit error)

Writes a JSON + CSV record so results are reproducible and diffable.

Usage:
    PYTHONPATH=backend/torch_vulkan_ext:third_party/HybridAgent:. \
        .venv/bin/python benchmarks/benchmark_matrix.py --out benchmarks/results
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")


def _sync():
    """Ensure the Vulkan queue is drained before timing resumes."""
    import torch_vulkan_ext as tvk
    try:
        tvk.submit()
    except Exception:
        pass


def bench_prefill(model_or_bb, lm_head, ids, use_cache=True, reps=3):
    """Median wall time for one forward pass."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        with torch.no_grad():
            model_or_bb(input_ids=ids, use_cache=use_cache)
        _sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), ts


def bench_decode(model_or_bb, lm_head, ids, n_tokens=16, reps=2):
    """Greedy decode with a persistent KV cache; returns tokens/sec and tokens.

    The prefill is done OUTSIDE the timed region so the number is decode-only.
    """
    from transformers.cache_utils import DynamicCache

    per_rep = []
    toks = []
    for _ in range(reps):
        cache = DynamicCache()
        with torch.no_grad():
            out = model_or_bb(input_ids=ids, past_key_values=cache, use_cache=True)
        cur = (out.logits[:, -1, :] if hasattr(out, "logits")
               else lm_head(out.last_hidden_state[:, -1, :]))
        generated = []
        nxt = cur.argmax(dim=-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            generated.append(int(nxt[0, 0]))
            with torch.no_grad():
                out = model_or_bb(input_ids=nxt, past_key_values=cache,
                                  use_cache=True)
            cur = (out.logits[:, -1, :] if hasattr(out, "logits")
                   else lm_head(out.last_hidden_state[:, -1, :]))
            nxt = cur.argmax(dim=-1, keepdim=True)
        _sync()
        dt = time.perf_counter() - t0
        per_rep.append(n_tokens / dt if dt > 0 else float("nan"))
        toks.append(generated)
    return statistics.median(per_rep), toks


def logit_error(model_or_bb, lm_head, ids):
    """Max abs difference of the final logits vs stock (run separately)."""
    with torch.no_grad():
        out = model_or_bb(input_ids=ids, use_cache=True)
    if hasattr(out, "logits"):
        return out.logits[:, -1, :].float()
    return lm_head(out.last_hidden_state[:, -1, :]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--lengths", default="12,512,1024")
    ap.add_argument("--layers", default="0,1,4,8")
    ap.add_argument("--decode-tokens", type=int, default=16)
    args = ap.parse_args()

    import torch_vulkan_ext as tvk
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement

    torch.manual_seed(0)
    model, tok = load_hybrid_agent(MODEL_DIR, dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    lm = model.backbone

    lengths = [int(x) for x in args.lengths.split(",")]
    layer_counts = [int(x) for x in args.layers.split(",")]

    # One input tensor per sequence length, generated ONCE. Every configuration
    # must be measured on the SAME tokens -- otherwise the correctness columns
    # compare different inputs and are meaningless.
    inputs = {}
    for S in lengths:
        g = torch.Generator().manual_seed(1234 + S)
        inputs[S] = torch.randint(0, 1000, (1, S), generator=g)

    results = {"config": {}, "runs": []}

    # ---- reference: stock -------------------------------------------------
    print("=== stock CPU ===", flush=True)
    for S in lengths:
        ids = inputs[S]
        with torch.no_grad():
            lm(input_ids=ids, use_cache=True)          # warmup
        med, raw = bench_prefill(lm, None, ids)
        ref_logits = logit_error(lm, None, ids)
        tps, toks = bench_decode(lm, None, ids, args.decode_tokens)
        print(f"  S={S:5d} prefill {med:8.3f}s   decode {tps:7.2f} tok/s",
              flush=True)
        results["runs"].append({
            "config": "stock", "gpu_layers": 0, "seq_len": S,
            "prefill_s": med, "prefill_raw": raw,
            "decode_tps": tps, "decode_tokens": toks[0],
        })
        results["config"].setdefault(f"stock_S{S}", ref_logits)

    # ---- wrapped configurations ------------------------------------------
    for ng in layer_counts:
        bb = HybridBackbone(qm, Placement.first_n(ng, len(qm.layers)))
        name = "wrapped_cpu" if ng == 0 else f"vk{ng}"
        print(f"=== {name} ===", flush=True)
        for S in lengths:
            ids = inputs[S]
            with torch.no_grad():
                bb(input_ids=ids, use_cache=True)
            med, raw = bench_prefill(bb, lm.lm_head, ids)
            tps, toks = bench_decode(bb, lm.lm_head, ids, args.decode_tokens)
            got = logit_error(bb, lm.lm_head, ids)
            ref = results["config"][f"stock_S{S}"]
            err = float((got - ref).abs().max())
            agree = toks[0] == results["runs"][
                [r["seq_len"] for r in results["runs"]
                 if r["config"] == "stock"].index(S)]["decode_tokens"]
            print(f"  S={S:5d} prefill {med:8.3f}s   decode {tps:7.2f} tok/s"
                  f"   logit_err {err:.3e}   token_agree {agree}", flush=True)
            results["runs"].append({
                "config": name, "gpu_layers": ng, "seq_len": S,
                "prefill_s": med, "prefill_raw": raw,
                "decode_tps": tps, "decode_tokens": toks[0],
                "logit_max_abs_err": err, "token_agree": agree,
            })

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # CSV
    with open(os.path.join(args.out, "results.csv"), "w") as f:
        f.write("config,gpu_layers,seq_len,prefill_s,decode_tps,"
                "logit_max_abs_err,token_agree\n")
        for r in results["runs"]:
            f.write(f"{r['config']},{r['gpu_layers']},{r['seq_len']},"
                    f"{r['prefill_s']:.4f},{r.get('decode_tps', float('nan')):.4f},"
                    f"{r.get('logit_max_abs_err', '')},"
                    f"{r.get('token_agree', '')}\n")
    print(f"\nwrote {args.out}/results.json and results.csv", flush=True)


if __name__ == "__main__":
    main()
