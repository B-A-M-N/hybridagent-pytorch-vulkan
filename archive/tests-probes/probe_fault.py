"""Find which layer op triggers the GPUVM fault."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import HybridBackbone, Placement  # noqa: E402


def main():
    torch.manual_seed(0)
    tvk.init(0)
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    qm = model.backbone.model
    ids = tok("What is 84 * 3 / 2?", return_tensors="pt")["input_ids"]

    for n in (1, 2, 4, 8):
        bb = HybridBackbone(qm, Placement.first_n(n, 36))
        try:
            with torch.no_grad():
                out = bb(input_ids=ids, use_cache=True, output_hidden_states=True)
            print(f"{n} layers: OK cache={out.past_key_values.get_seq_length()}",
                  flush=True)
        except Exception as e:
            print(f"{n} layers: FAILED {str(e)[:70]}", flush=True)
            break


if __name__ == "__main__":
    main()
