"""Probe: does the real HybridAgent emit the <latent> anchor, and what does the
latent-exit readout look like? Runs on CPU with the original model."""
import sys
import time

import torch

sys.path.insert(0, "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent")
torch.set_num_threads(24)

from hybrid_agent import load_hybrid_agent  # noqa: E402

MODEL = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"


def main():
    t0 = time.time()
    model, tok = load_hybrid_agent(MODEL, dtype=torch.bfloat16, device="cpu")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    for q in ["What is 84 * 3 / 2?", "How many trailing zeros does 100! have?"]:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            out = model.backbone(enc["input_ids"])
        lg = out.logits[0, -1]
        nxt = int(lg.argmax())
        print(f"\nQ: {q}", flush=True)
        print(f"  forward {time.time() - t1:.1f}s", flush=True)
        print(f"  next token: {tok.decode([nxt])!r} (id {nxt})", flush=True)
        print(f"  <latent> rank: {int((lg > lg[151669]).sum()) + 1}", flush=True)
        print(f"  top5: {[(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(*lg.topk(5))]}",
              flush=True)


if __name__ == "__main__":
    main()
