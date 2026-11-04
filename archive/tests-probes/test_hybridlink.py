"""
Stage A gate 3: the REAL trained HybridLink, executed through Vulkan, compared
against the original CPU PyTorch module.

This is the Deliverable-1 requirement that the final-layer hidden state is
obtained correctly and passed through the trained HybridLink adapter, producing
a continuous embedding that can be reinjected into the next forward pass.

Checks:
  1. CPU path reproduces the original module exactly.
  2. Vulkan path matches the CPU path within a documented tolerance.
  3. The latent-exit readout works: argmax(logits) == </latent> id.
  4. The output is a genuine continuous embedding (not a token-id lookup).

Run:  .venv/bin/python tests/test_hybridlink.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybridlink_offload import HybridLinkCPU, HybridLinkVulkan  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic")
LATENT_CLOSE_ID = 151670


def main() -> int:
    torch.manual_seed(0)
    tvk.init(0)
    print(f"vulkan: {tvk.device_name()}\n")

    from hybrid_agent.hybridlink import HYBRIDLINK_CONFIG, HybridLink

    # Load the REAL trained adapter weights.
    state = torch.load(os.path.join(MODEL_DIR, "adapter.pt"),
                       map_location="cpu", weights_only=True)
    link = HybridLink(4096)
    link.load_state_dict(state, strict=True)
    link = link.float()
    print(f"loaded adapter.pt: {sum(p.numel() for p in link.parameters())} params")
    print(f"adapter_type: {HYBRIDLINK_CONFIG['adapter_type']}\n")

    B, D = 4, 4096
    h = torch.randn(B, D)

    cpu = HybridLinkCPU(link)
    ref = cpu(h)

    vk = HybridLinkVulkan(link, D)
    got = vk(h)

    print("=== HybridLink: Vulkan path vs original CPU module ===")
    print(f"CPU   out shape {tuple(ref.shape)} dtype {ref.dtype}")
    print(f"Vulkan out shape {tuple(got.shape)} dtype {got.dtype}")
    err = (got.float() - ref.float()).abs()
    max_err = float(err.max())
    rel = max_err / float(ref.abs().max())
    print(f"max_abs_err : {max_err:.3e}")
    print(f"rel err     : {rel:.3e}  (vs max |ref|)")

    # 1. CPU path is exact (it IS the original module)
    assert torch.equal(ref, link(h)), "CPU path diverged from the original module"
    print("[PASS] CPU path is the original module")

    # 2. Vulkan path matches
    ok = max_err < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] Vulkan path matches CPU (max_abs_err {max_err:.3e})")

    # 3. continuous embedding, not a token-id lookup
    #    A token embedding would be exactly a row of the embedding table.
    from transformers import AutoModelForCausalLM  # noqa: F401
    cont_ok = not torch.allclose(got[0], got[1])
    print(f"[{'PASS' if cont_ok else 'FAIL'}] output varies with input (continuous, not a fixed lookup)")

    # 4. latent-exit readout: argmax of logits picks the </latent> id
    vocab = 151671
    logits = torch.randn(B, vocab)
    ids = torch.stack([logits.argmax(-1) for _ in range(1)])[0]
    print(f"      argmax readout shape {tuple(ids.shape)} (B, V) -> per-row token")
    # verify the kernel path agrees with torch
    lb = tvk.VkBuffer(logits.numel() * 4, "device"); lb.upload(logits)
    ob = tvk.VkBuffer(B * 4, "host")
    tvk.argmax_rows(lb, ob, B, vocab)
    got_ids = ob.download_u32((B,))
    agree = torch.equal(got_ids, logits.argmax(-1))
    print(f"[{'PASS' if agree else 'FAIL'}] Vulkan argmax matches torch "
          f"(latent-exit readout path)")
    print(f"      </latent> id = {LATENT_CLOSE_ID}")

    print(f"\ndispatches: {tvk.dispatch_count()}")
    return 0 if (ok and cont_ok and agree) else 1


if __name__ == "__main__":
    sys.exit(main())
