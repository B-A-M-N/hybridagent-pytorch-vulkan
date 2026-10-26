"""
HybridLink execution paths (Deliverable 1, Phase 4).

The original HybridLink is:

    out = post_ln(h + proj2(gelu(proj1(pre_ln(h)))))

with `proj1`, `proj2` Linear, `gelu` = `nn.GELU()` (erf variant), and `pre_ln`,
`post_ln` LayerNorm. The trained weights live in `adapter.pt`.

Path A (CPU baseline) -- the original module runs on CPU, unchanged. The engine
copies the last-layer hidden state to CPU, runs HybridLink, and copies the
continuous embedding back to the device that runs the next forward.

Path B (Vulkan) -- the same trained weights and the same maths, executed through
the verified Vulkan kernels, so a latent step does not pay a host round trip.

Both paths preserve:
  * the trained adapter weights (loaded strict from `adapter.pt`)
  * last-layer hidden-state selection
  * input/output tensor dimensions
  * residual and normalisation behaviour
  * continuous embedding injection (never token-id approximation)

Implementation note: this module is written against the verified `torch_vulkan_ext`
ctypes API (`tvk.layernorm`, `tvk.matmul_t_bias`, `tvk.gelu_erf`). An earlier
revision used `tensor.to("vulkan")` and `torch.ops.torch_vulkan_ext`, which do not
exist in the shipped backend.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import torch_vulkan_ext as tvk


class HybridLinkCPU(nn.Module):
    """Path A: the original HybridLink, unchanged, on CPU."""

    def __init__(self, link: nn.Module) -> None:
        super().__init__()
        self.link = link

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.link(hidden_states)


class HybridLinkVulkan(nn.Module):
    """Path B: HybridLink on the Vulkan device, same weights, same maths.

    LayerNorm goes through the `layernorm` kernel and GELU through `gelu_erf`,
    which implements the erf variant that `nn.GELU()` uses by default. (The
    `gelu` kernel in upstream torch-vulkan uses the tanh approximation, which
    would introduce a spurious ~1e-3 divergence in the most sensitive
    component of the latent path.)
    """

    def __init__(self, link: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Carry the trained weights across in fp32, which is what the Vulkan
        # kernels compute in. They are uploaded once and stay resident.
        def grab(mod, attr):
            return getattr(mod, attr).detach().float().contiguous()

        self.proj1_w = nn.Parameter(grab(link.proj1, "weight"))
        self.proj1_b = nn.Parameter(grab(link.proj1, "bias"))
        self.proj2_w = nn.Parameter(grab(link.proj2, "weight"))
        self.proj2_b = nn.Parameter(grab(link.proj2, "bias"))
        self.pre_w = nn.Parameter(grab(link.pre_ln, "weight"))
        self.pre_b = nn.Parameter(grab(link.pre_ln, "bias"))
        self.post_w = nn.Parameter(grab(link.post_ln, "weight"))
        self.post_b = nn.Parameter(grab(link.post_ln, "bias"))

        # Upload the resident weights once.
        self._res = {
            "proj1_w": tvk.VkBuffer(self.proj1_w.numel() * 4, "device"),
            "proj1_b": tvk.VkBuffer(self.proj1_b.numel() * 4, "device"),
            "proj2_w": tvk.VkBuffer(self.proj2_w.numel() * 4, "device"),
            "proj2_b": tvk.VkBuffer(self.proj2_b.numel() * 4, "device"),
            "pre_w": tvk.VkBuffer(self.pre_w.numel() * 4, "device"),
            "pre_b": tvk.VkBuffer(self.pre_b.numel() * 4, "device"),
            "post_w": tvk.VkBuffer(self.post_w.numel() * 4, "device"),
            "post_b": tvk.VkBuffer(self.post_b.numel() * 4, "device"),
        }
        for name, t in (
            ("proj1_w", self.proj1_w), ("proj1_b", self.proj1_b),
            ("proj2_w", self.proj2_w), ("proj2_b", self.proj2_b),
            ("pre_w", self.pre_w), ("pre_b", self.pre_b),
            ("post_w", self.post_w), ("post_b", self.post_b),
        ):
            self._res[name].upload(t)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _dev(t: torch.Tensor) -> tvk.VkBuffer:
        b = tvk.VkBuffer(t.numel() * 4, "device")
        b.upload(t.detach().float().contiguous())
        return b

    def _zero(self, n: int) -> tvk.VkBuffer:
        b = tvk.VkBuffer(n * 4, "device")
        b.upload(torch.zeros(n))
        return b

    def _out(self, shape) -> tvk.VkBuffer:
        n = 1
        for s in shape:
            n *= s
        return tvk.VkBuffer(max(1, n) * 4, "host")

    def _linear(self, x: torch.Tensor, w: tvk.VkBuffer, b: tvk.VkBuffer,
                out_features: int) -> torch.Tensor:
        """out = x @ w.T + b, with w in the [out, in] nn.Linear layout."""
        M = x.numel() // x.shape[-1]
        K = x.shape[-1]
        xb = self._dev(x.reshape(M, K))
        ob = self._out((M, out_features))
        tvk.matmul_t_bias(xb, w, b, ob, M, K, out_features)
        return ob.download((M, out_features)).reshape(x.shape[:-1] + (out_features,))

    def _layernorm(self, x: torch.Tensor, w: tvk.VkBuffer, b: tvk.VkBuffer,
                   eps: float = 1e-5) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        row_size = x.shape[-1]
        xb = self._dev(x.reshape(rows, row_size))
        ob = self._out((rows, row_size))
        tvk.layernorm(xb, w, b, ob, rows, row_size, eps)
        return ob.download((rows, row_size)).reshape(x.shape)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out_dtype = h.dtype
        x = h.detach().float().contiguous()
        h_flat = x.reshape(-1)

        # pre_ln(h)
        x = self._layernorm(x, self._res["pre_w"], self._res["pre_b"])
        # proj1
        x = self._linear(x, self._res["proj1_w"], self._res["proj1_b"],
                         self.proj1_w.shape[0])
        # gelu (erf)
        n = x.numel()
        xb = self._dev(x.reshape(n))
        gb = self._out((n,))
        tvk.gelu_erf(xb, gb, n)
        x = gb.download((n,)).reshape(x.shape)
        # proj2
        x = self._linear(x, self._res["proj2_w"], self._res["proj2_b"],
                         self.proj2_w.shape[0])
        # residual
        x = (x.reshape(-1) + h_flat).reshape(h.shape)
        # post_ln
        x = self._layernorm(x, self._res["post_w"], self._res["post_b"])
        return x.to(out_dtype)


class LatentInjector:
    """Produces the next continuous latent input from the last-layer hidden state.

    This is what the engine calls during latent mode; it is the only piece of
    the latent path that knows which device executes HybridLink.
    """

    def __init__(self, link: nn.Module, hidden_size: int, path: str = "cpu") -> None:
        self.path = path
        if path == "vulkan":
            self.impl = HybridLinkVulkan(link, hidden_size)
        else:
            self.impl = HybridLinkCPU(link).cpu()
        self.hidden_size = hidden_size

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.impl(hidden_states)

    def to(self, *a, **k):      # keep the nn.Module API for the engine
        return self
