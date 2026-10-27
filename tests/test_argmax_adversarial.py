"""
Adversarial tests for the argmax / argmax_rows Vulkan reduction kernels.

Context
-------
A previous report claimed the argmax shader had been repaired using
`__shfl_down_sync`. That is an NVIDIA CUDA intrinsic and does not exist in
Vulkan GLSL. The actual sources were audited (see the audit in
tests/probe_4000x_cause.py notes):

  backend/torch_vulkan_ext/csrc/shaders/argmax.comp
      - shared-memory tree reduction, 256 threads
      - NO subgroup operations, NO CUDA intrinsics
  backend/torch_vulkan_ext/csrc/shaders/argmax_rows.comp
      - one thread per row, serial scan down the row
      - NO subgroup operations, NO CUDA intrinsics

`spirv-dis` confirms the compiled argmax.spv declares only `OpCapability Shader`
-- no `OpCapability GroupNonUniform` / `GroupNonUniformVote` / `SubgroupBallot`,
which is what a subgroup-based or warp-shuffle-based reduction would require.
The compiled binary was also verified byte-identical to a fresh
`glslangValidator --target-env vulkan1.1 -V` compile of the .comp source.

These tests then verify the kernels are numerically CORRECT, which the claim
about the repair ultimately has to be judged on.

Cases (deterministic, reference value computed by torch.argmax):
  - tiny (1x1, 1x7, 3x1)
  - non-power-of-two columns (7, 31, 151671-ish via 4097)
  - negative values
  - all-equal values (duplicate / tie handling)
  - boundary lengths around the 256-thread workgroup size
  - values spanning the full fp32 range
  - NaN-free deterministic random data at many shapes
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402

FAILURES = []


def check(name, got, want, tol=0):
    ok = (int(got) == int(want)) if tol == 0 else abs(int(got) - int(want)) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(f"{name}: got {int(got)} want {int(want)}")
    print(f"  [{status}] {name:52s} got={int(got):8d} want={int(want):8d}")
    return ok


def run_argmax(x: torch.Tensor) -> int:
    """Global argmax over the whole matrix; returns the flat index."""
    a = tvk.Allocation(x.numel() * 4, "device")
    av = a.view()
    av.upload(x.reshape(-1, 1).float() if x.dim() == 1 else x.float())
    o = tvk.Allocation(4, "device")
    ov = o.view()
    tvk.argmax(av, ov, x.shape[0], x.shape[1])
    out = ov.download((1,), dtype=torch.uint32)
    return int(out.reshape(-1)[0].item())


def run_argmax_rows(x: torch.Tensor):
    """Row-wise argmax; returns the list of column indices."""
    a = tvk.Allocation(x.numel() * 4, "device")
    av = a.view()
    av.upload(x.float())
    o = tvk.Allocation(x.shape[0] * 4, "device")
    ov = o.view()
    tvk.argmax_rows(av, ov, x.shape[0], x.shape[1])
    out = ov.download((x.shape[0],), dtype=torch.uint32)
    return [int(v) for v in out.reshape(-1).tolist()]


def main():
    torch.manual_seed(1234)
    tvk.init(0)
    print(f"device: {tvk.device_name()}\n")

    print("=== global argmax (argmax.comp) ===")
    cases = []

    # tiny / degenerate
    cases.append(("1x1 single element", torch.tensor([[3.0]])))
    cases.append(("1x7 one row", torch.randn(1, 7)))
    cases.append(("3x1 one column", torch.randn(3, 1)))

    # non-power-of-two widths
    for cols in (7, 31, 33, 63, 65, 255, 257, 1023, 4097):
        cases.append((f"{cols} cols (non-pow2)", torch.randn(4, cols)))

    # boundary around the 256-thread workgroup
    for total in (255, 256, 257, 511, 512, 513):
        cases.append((f"{total} elements (workgroup boundary)",
                      torch.randn(1, total)))

    # negative values
    t = torch.randn(4, 64) - 100.0
    cases.append(("all negative", t))

    # duplicates / ties
    t = torch.zeros(4, 64)
    t[2, 33] = 5.0
    t[0, 7] = 5.0
    cases.append(("ties: two equal maxima", t))

    t = torch.full((4, 64), 2.5)
    cases.append(("all values identical", t))

    # extreme magnitudes
    t = torch.tensor([[1.0, -3.0e38, 2.0, 3.0e38, -1.0]])
    cases.append(("full fp32 range", t))

    t = torch.tensor([[0.0, -0.0, 1.0]])
    cases.append(("+0.0 vs -0.0", t))

    for name, t in cases:
        want = int(torch.argmax(t.reshape(-1)).item())
        got = run_argmax(t)
        check(name, got, want)

    print("\n=== row-wise argmax (argmax_rows.comp) ===")
    row_cases = [
        ("1x1", torch.tensor([[7.0]])),
        ("1x7", torch.randn(1, 7)),
        ("3x1", torch.randn(3, 1)),
        ("4x31 non-pow2", torch.randn(4, 31)),
        ("4x257 non-pow2", torch.randn(4, 257)),
        ("4x4097 large", torch.randn(4, 4097)),
        ("all negative", torch.randn(4, 64) - 100.0),
        ("ties across rows", torch.zeros(4, 64)),
        ("row-wise different maxima", torch.randn(8, 100)),
    ]
    for name, t in row_cases:
        want = [int(v) for v in torch.argmax(t, dim=1).tolist()]
        got = run_argmax_rows(t)
        ok = got == want
        if not ok:
            FAILURES.append(f"rows {name}: got {got[:8]} want {want[:8]}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} "
              f"got={got[:4]} want={want[:4]}")

    print("\n=== randomized sweep ===")
    bad = 0
    for i in range(40):
        rows = int(torch.randint(1, 9, (1,)))
        cols = int(torch.randint(1, 600, (1,)))
        t = torch.randn(rows, cols)
        want = int(torch.argmax(t.reshape(-1)).item())
        got = run_argmax(t)
        if got != want:
            bad += 1
            FAILURES.append(f"random {rows}x{cols}: got {got} want {want}")
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] 40 random shapes, {bad} mismatches")

    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for f in FAILURES[:20]:
            print("   ", f)
        sys.exit(1)
    print("ALL ARGMAX TESTS PASS")


if __name__ == "__main__":
    main()
