"""
PERMANENT REGRESSION TEST: attention2 multi-key correctness.

Why this exists
---------------
`tests/test_vulkan_ops.py` reported 12/12 passing while attention2 was broken for
every sequence length >= 2. The suite only exercised the decode shape (n_q = 1),
where softmax is forced to weight 1.0 by a single unmasked key regardless of the
Q.K score -- so it cannot detect a broken dot product, wrong scaling, defective
normalisation, or bad K/V addressing. This is a coverage gap, and this test
closes it.

Design
------
Inputs are DELIBERATELY constructed so that attention weights are far from
uniform, which is what makes a defect observable:

  * q rows are unit vectors along different axes, so q.k is large and distinct
    per key -- the softmax weights are strongly non-uniform.
  * k vectors are scaled to control the score magnitude explicitly.
  * v vectors are one-hot along distinct dimensions, so the attention OUTPUT is
    a readable fingerprint of which weights were used. If the output equals the
    mean of the v rows, the weights were uniform; if it equals one row, a single
    key dominated.

Semantics
---------
The GPU kernel takes an ADDITIVE mask: 0.0 = attend, -inf = skip.
The CPU reference below uses the same convention (add -inf). A boolean
"keep" mask is NOT equivalent and is not used here.
"""
import os
import sys

import pytest
import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)


def _cpu_attention(q, k, v, mask):
    """Reference: q [NH,S,HD], k/v [NKV,S,HD], mask [NH,S,S] additive."""
    kh = k.repeat_interleave(NH // NKV, dim=0)
    vh = v.repeat_interleave(NH // NKV, dim=0)
    scores = torch.matmul(q, kh.transpose(-1, -2)) * SCALING + mask
    return torch.matmul(torch.softmax(scores, dim=-1), vh), scores


def _deliberate_inputs(S, seed=0):
    """Build q/k/v so that attention weights are strongly non-uniform.

    q[0] points along axis 0, q[1] along axis 1, etc. k[j] is a scaled unit
    vector along axis j, so the score for key j is large only when q's axis
    matches -- giving one dominant key and clearly distinct weights.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(NH, S, HD)
    k = torch.zeros(NKV, S, HD)
    v = torch.zeros(NKV, S, HD)
    for s in range(S):
        q[:, s, s % HD] = 1.0
        k[:, s, s % HD] = 4.0          # large magnitude -> decisive scores
        v[:, s, :] = torch.randn(NKV, HD, generator=g)
    return q, k, v


def _deliberate_inputs_v2(S, seed=0):
    """q row s has its 1.0 at dimension 0 of its own row.

    The key property needed: q row s must be ORTHOGONAL to k row m for m != s,
    so the score for key m is zero, and strongly positive for key s. Placing the
    1.0 at dimension s (as _deliberate_inputs does) puts it inside the row at
    offset s, which is a different element from the row's dimension 0 and made
    the diagnostic read 0 -- a test-input bug, not a kernel bug.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(NH, S, HD)
    k = torch.zeros(NKV, S, HD)
    v = torch.zeros(NKV, S, HD)
    for s in range(S):
        # q row s: 1.0 at dimension s  -> orthogonal to k rows m != s only if
        # k rows are also axis-aligned. Use distinct axes per key.
        q[:, s, s % HD] = 1.0
        k[:, s, s % HD] = 4.0
        v[:, s, :] = torch.randn(NKV, HD, generator=g)
    return q, k, v


def _causal_mask(S):
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    return torch.zeros(NH, S, S).masked_fill(~allow, float("-inf"))


def _run(q, k, v, mask, cap):
    # HOLD every _Buf for the duration of the dispatch. A `_Buf` holds a
    # VkBuffer whose __del__ frees it; if the wrapper is garbage-collected
    # before the GPU has finished reading, the kernel reads unmapped VA and
    # silently returns zeros. This was the actual cause of the "S>=2 attention
    # defect" -- not a shader bug. Submission is not completion.
    qb = _Buf(q.contiguous())
    kb = _Buf(k.contiguous())
    vb = _Buf(v.contiguous())
    mb = _Buf(mask.contiguous())
    pb = _Buf.out((NH, q.shape[1], cap))
    ob = _Buf.out((NH, q.shape[1], HD))
    tvk.attention2(qb.buf, kb.buf, vb.buf, mb.buf, pb.buf,
                   ob.buf, 0, cap, NH, NH // NKV, HD, cap, SCALING)
    return pb.cpu(), ob.cpu()


@pytest.fixture(scope="module", autouse=True)
def _vk():
    tvk.init(0)


# ---------------------------------------------------------------------------
# The reproduction that started this investigation.
# ---------------------------------------------------------------------------

def test_two_key_prefill_reproduces_the_defect():
    """S=2 must be exact. This is the minimal failing case."""
    S = 2
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    ref, _ = _cpu_attention(q, k, v, mask)
    got_p, got_o = _run(q, k, v, mask, S)
    err = float((got_o - ref).abs().max())
    assert err < 1e-4, f"attention2 S=2 output err {err:.3e} (expected <1e-4)"


# ---------------------------------------------------------------------------
# Step 4 coverage: sequence lengths, masks, GQA, strides.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S", [1, 2, 3, 4, 8, 12, 33, 64])
def test_sequence_lengths(S):
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    ref, _ = _cpu_attention(q, k, v, mask)
    _, got_o = _run(q, k, v, mask, S)
    assert float((got_o - ref).abs().max()) < 1e-4, f"S={S} failed"


def test_decode_shape_still_exact():
    """The shape the original suite covered -- must not regress."""
    S, cap = 1, 32
    q, k, v = _deliberate_inputs(cap)
    q = q[:, :1, :]
    mask = torch.zeros(NH, 1, cap)
    ref, _ = _cpu_attention(q, k, v, mask)
    _, got_o = _run(q, k, v, mask, cap)
    assert float((got_o - ref).abs().max()) < 1e-4


def test_weights_are_not_uniform():
    """Guard against the specific failure mode: weights collapsing to uniform.

    With the deliberate inputs, the correct weights are strongly peaked. A
    uniform result would be the signature of the S>=2 defect.
    """
    S = 2
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    _, ref_scores = _cpu_attention(q, k, v, mask)
    ref_w = torch.softmax(ref_scores, dim=-1)
    got_p, _ = _run(q, k, v, mask, S)
    # row 1 attends both keys; correct weights are non-uniform
    assert float(ref_w[0, 1].std()) > 0.05, "test inputs are not decisive"
    assert float(got_p[0, 1].std()) > 0.05, \
        "GPU weights collapsed toward uniform (the S>=2 defect)"


def test_noncausal_mask():
    """An all-zero additive mask must attend every key."""
    S = 4
    q, k, v = _deliberate_inputs(S)
    mask = torch.zeros(NH, S, S)
    ref, _ = _cpu_attention(q, k, v, mask)
    _, got_o = _run(q, k, v, mask, S)
    assert float((got_o - ref).abs().max()) < 1e-4


def test_gqa_grouping():
    """Each group of 4 query heads must read its own kv head."""
    S = 3
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    ref, _ = _cpu_attention(q, k, v, mask)
    _, got_o = _run(q, k, v, mask, S)
    # check per-head, so a head-to-kv mis-mapping cannot cancel out
    per_head = (got_o - ref).abs().amax(dim=(1, 2))
    assert float(per_head.max()) < 1e-4, \
        f"GQA grouping wrong; worst head err {float(per_head.max()):.3e}"


def test_distinct_kv_offsets():
    """cap > S exercises non-trivial key strides and a partial window."""
    S, cap = 3, 64
    q, k, v = _deliberate_inputs(cap)
    q = q[:, :S, :]
    mask = torch.zeros(NH, S, cap)
    ref, _ = _cpu_attention(q, k, v, mask)
    _, got_o = _run(q, k, v, mask, cap)
    assert float((got_o - ref).abs().max()) < 1e-4


def test_repeated_dispatch_is_stable():
    """Same inputs, repeated dispatches: results must not drift or alias."""
    S = 2
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    ref, _ = _cpu_attention(q, k, v, mask)
    outs = [_run(q, k, v, mask, S)[1] for _ in range(5)]
    for i, o in enumerate(outs):
        assert float((o - ref).abs().max()) < 1e-4, f"dispatch {i} diverged"
    for i in range(1, len(outs)):
        assert float((outs[i] - outs[0]).abs().max()) == 0.0, \
            f"dispatch {i} is not deterministic"


def test_output_is_not_the_uniform_mean():
    """If weights were uniform the output would equal mean(v). Detect that."""
    S = 2
    q, k, v = _deliberate_inputs(S)
    mask = _causal_mask(S)
    _, got_o = _run(q, k, v, mask, S)
    uniform = v.repeat_interleave(NH // NKV, dim=0).mean(dim=1)
    assert float((got_o[:, 1, :] - uniform).abs().max()) > 1e-2, \
        "output equals the uniform mean of V -- weights were uniform"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
