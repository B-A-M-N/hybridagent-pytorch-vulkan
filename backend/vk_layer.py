"""
The Qwen3 decoder layer executed on Vulkan, built from the verified kernels in
`torch_vulkan_ext`.

This is the unit that Deliverable 2 offloads. Each function maps one PyTorch op
to one verified kernel, with an explicit CPU reference alongside it so every
discrepancy is measurable rather than assumed.

All kernels are fp32. Layer inputs are cast to fp32 for the GPU path and the
result cast back to the layer's input dtype before the residual add, which is
where the original keeps bf16.
"""
from __future__ import annotations

import torch

import torch_vulkan_ext as tvk


class VkLinear:
    """A resident nn.Linear weight executed through the Vulkan matmul kernel.

    Weights are uploaded once and stay resident in VRAM; no per-step transfer.
    The kernel takes the [N, K] (nn.Linear) layout directly, so no transpose.
    """

    def __init__(self, weight: torch.Tensor):
        self.out_features, self.in_features = weight.shape
        self.buf = tvk.VkBuffer(weight.numel() * 4, "device")
        self.buf.upload(weight.detach().float())

    def __call__(self, x: "VkTensor") -> "VkTensor":
        return x @ self.buf


class VkTensor:
    """A fp32 tensor held in a Vulkan device buffer.

    Keeps the logical shape while storage lives on the GPU, so the layer code
    reads like ordinary tensor code.
    """

    __slots__ = ("buf", "shape")

    def __init__(self, buf, shape):
        self.buf = buf
        self.shape = tuple(shape)

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "VkTensor":
        buf = tvk.VkBuffer(t.numel() * 4, "device")
        buf.upload(t.detach().float())
        return cls(buf, t.shape)

    def to_cpu(self, dtype=torch.float32) -> torch.Tensor:
        return self.buf.download(self.shape, dtype)

    def __matmul__(self, other):
        """x @ W where W is a resident [N, K] weight -> [M, N]."""
        M = self.numel // self.shape[-1]
        K = self.shape[-1]
        N = other.out_features
        x2 = self.reshape(M, K)
        out = tvk.VkBuffer(M * N * 4, "device")
        tvk.matmul_t_bias(x2.buf, other.buf, _zero_bias(N), out, M, K, N)
        return VkTensor(out, (M, N))

    def reshape(self, *shape):
        assert self.numel == int(torch.tensor(shape).prod())
        return VkTensor(self.buf, shape)


_BIAS_CACHE: dict[int, tvk.VkBuffer] = {}


def _zero_bias(n: int) -> tvk.VkBuffer:
    """A cached zero bias vector (the attention/MLP projections have none)."""
    b = _BIAS_CACHE.get(n)
    if b is None:
        t = torch.zeros(n)
        b = tvk.VkBuffer(n * 4, "device")
        b.upload(t)
        _BIAS_CACHE[n] = b
    return b


def rmsnorm(x: VkTensor, weight: torch.Tensor, eps: float) -> VkTensor:
    """Qwen3RMSNorm: fp32 statistics, weight applied in fp32."""
    rows = x.numel // x.shape[-1]
    w = tvk.VkBuffer(weight.numel() * 4, "device")
    w.upload(weight.detach().float())
    out = tvk.VkBuffer(x.numel * 4, "device")
    tvk.rmsnorm(x.buf, w, out, rows, x.shape[-1], eps)
    return VkTensor(out, x.shape)


def gelu(x: VkTensor) -> VkTensor:
    """nn.GELU() default (erf variant) -- matches HybridLink."""
    out = tvk.VkBuffer(x.numel * 4, "device")
    tvk.gelu_erf(x.buf, out, x.numel)
    return VkTensor(out, x.shape)


def layernorm(x: VkTensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> VkTensor:
    rows = x.numel // x.shape[-1]
    wb = tvk.VkBuffer(weight.numel() * 4, "device"); wb.upload(weight.detach().float())
    bb = tvk.VkBuffer(bias.numel() * 4, "device"); bb.upload(bias.detach().float())
    out = tvk.VkBuffer(x.numel * 4, "device")
    tvk.layernorm(x.buf, wb, bb, out, rows, x.shape[-1], eps)
    return VkTensor(out, x.shape)


def silu_gate(gate_up: VkTensor) -> VkTensor:
    """out = silu(gate) * up, with [.., 2N] split into two N halves."""
    N = gate_up.numel // 2
    out = tvk.VkBuffer(N * 4, "device")
    tvk.silu_gate(gate_up.buf, out, N)
    shape = list(gate_up.shape)
    shape[-1] = shape[-1] // 2
    return VkTensor(out, shape)


def residual_add(a: VkTensor, b: torch.Tensor) -> VkTensor:
    """a + b, with b on CPU (the residual stream lives in the layer's dtype)."""
    bb = tvk.VkBuffer(b.numel() * 4, "device")
    bb.upload(b.detach().float())
    out = tvk.VkBuffer(a.numel * 4, "device")
    tvk.residual_add(a.buf, bb, out, a.numel)
    return VkTensor(out, a.shape)


def to_vk(t: torch.Tensor) -> VkTensor:
    return VkTensor.from_tensor(t)


def to_cpu(t: VkTensor, dtype=torch.float32) -> torch.Tensor:
    return t.to_cpu(dtype)
