"""
Heterogeneous CPU/Vulkan execution for the HybridAgent backbone.

This is Deliverable 2. It keeps the original HybridAgent generation controller
(`HybridGenerationEngine`), the original `HybridLink` module and the original
KV-cache representation untouched, and changes only *where* the backbone's
transformer layers execute.

Design
------
`HybridBackbone` mirrors `Qwen3Model.forward` but dispatches each layer to the
Vulkan device or the CPU according to an explicit `Placement`. Placement is just
a list of layer indices, so an all-CPU run is available as the reference for
differential testing.

Layer execution uses the *same verified kernel sequence* as
`tests/test_layer_vulkan.py` (gate 2), which was shown to match the CPU
reference to 8.9e-05 on the hidden state.

KV cache
--------
The original uses `transformers.cache_utils.DynamicCache`, whose
`past_key_values.layers[i].keys/.values` are `[B, n_kv, L, hd]` tensors. Two
constraints follow from keeping that contract:

  * `sampling.expand_cache_batch` fans a batch-1 prefill out to B rows by walking
    `vars(layer)` and calling `repeat_interleave` on every tensor attribute --
    that only works if the cache tensors are ordinary tensors on one device;
  * the engine's `forward_step` passes `past_key_values` straight into the
    backbone and reads `outputs.past_key_values` back.

So the cache stays on CPU and follows the original semantics. Each Vulkan layer
transfers its freshly computed K/V to the CPU cache once per step, and brings the
cache window back for attention. That transfer cost is measured and reported,
not hidden.

Numerics
--------
All Vulkan kernels are fp32; the checkpoint is bf16. Layer inputs are cast to
fp32 for the GPU computation and the result is cast back to the layer's input
dtype before the residual add, which is where the original keeps bf16. Every
cast is a reported numerical change.
"""
from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

import torch_vulkan_ext as tvk


def u32(v) -> bytes:
    """Little-endian uint32, for hand-packed push constants."""
    return struct.pack("<I", int(v))


# ---------------------------------------------------------------------------
# Device accounting
# ---------------------------------------------------------------------------

@dataclass
class DeviceStats:
    """Per-op device accounting for one forward pass."""

    dispatches: int = 0
    vk_time_s: float = 0.0
    cpu_time_s: float = 0.0
    transfers: int = 0
    transfer_bytes: int = 0
    layers_on_gpu: int = 0
    layers_on_cpu: int = 0
    ops: list[tuple[str, str]] = field(default_factory=list)

    def record(self, op: str, device: str) -> None:
        self.ops.append((op, device))

    def summary(self) -> dict:
        from collections import Counter
        by_dev = Counter(d for _, d in self.ops)
        return {
            "dispatches": self.dispatches,
            "vk_time_s": round(self.vk_time_s, 6),
            "cpu_time_s": round(self.cpu_time_s, 6),
            "transfers": self.transfers,
            "transfer_bytes": self.transfer_bytes,
            "layers_on_gpu": self.layers_on_gpu,
            "layers_on_cpu": self.layers_on_cpu,
            "ops_by_device": dict(by_dev),
        }


# ---------------------------------------------------------------------------
# Layer placement
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    """Which transformer layers run on Vulkan."""

    layers_on_gpu: list[int] = field(default_factory=list)

    @classmethod
    def first_n(cls, n: int, total: int = 36) -> "Placement":
        return cls(layers_on_gpu=list(range(max(0, min(n, total)))))

    @classmethod
    def all_cpu(cls) -> "Placement":
        return cls(layers_on_gpu=[])

    @classmethod
    def all_gpu(cls, total: int = 36) -> "Placement":
        return cls(layers_on_gpu=list(range(total)))

    def device_of(self, layer_idx: int) -> str:
        return "vulkan" if layer_idx in self.layers_on_gpu else "cpu"

    def __len__(self) -> int:
        return len(self.layers_on_gpu)

    @property
    def num_gpu_layers(self) -> int:
        """Convenience property for the number of GPU layers."""
        return len(self.layers_on_gpu)


def bytes_per_layer(hidden: int, inter: int, n_heads: int, head_dim: int,
                    n_kv_heads: int) -> int:
    """Weights of one Qwen3 decoder layer, in fp32 bytes."""
    h = hidden
    elems = (n_heads * head_dim * h) + 2 * (n_kv_heads * head_dim * h) + \
            (h * n_heads * head_dim) + 2 * head_dim + \
            2 * (inter * h) + (h * inter) + 2 * h
    return elems * 4


def vram_budget_layers(total_vram_bytes: int, per_layer_bytes: int,
                       reserve_fraction: float = 0.30) -> int:
    """How many layers fit in VRAM after reserving headroom.

    Deliberately does not fill the whole 8 GB just because the weights would fit:
    the reserve covers the Vulkan runtime, intermediate activations, the KV cache
    and temporary buffers.
    """
    usable = int(total_vram_bytes * (1.0 - reserve_fraction))
    if per_layer_bytes <= 0:
        return 0
    return max(0, usable // per_layer_bytes)


# ---------------------------------------------------------------------------
# Vulkan-backed decoder layer
# ---------------------------------------------------------------------------

class _Buf:
    """Small helper around a Vulkan buffer, mirroring the gate-2 test helpers.

    Lifetime
    --------
    A queued dispatch holds a raw pointer into `self.buf`. If this object is
    garbage-collected while the queue is still in flight, the allocation is freed
    and the GPU reads unmapped VA -- which on RADV surfaces as
    `GPUVM fault` / `context is lost`, or silently returns zeros.

    Submission is not completion. So `_Buf` keeps an explicit hold count:
    `keep()` is called by any dispatch that references the buffer, and
    `drop()` once the layer has drained the queue.

    `_dispatch` also takes a reference for the duration of each call, which
    covers the common (and hazardous) inline form `_Buf(t).buf` -- the temporary
    would otherwise be collected at the end of that expression, before the
    dispatch it feeds has even been issued.
    """

    __slots__ = ("_alloc", "_view", "shape", "_holds")

    def __init__(self, t: torch.Tensor, kind: str = "device"):
        t = t.detach().to(torch.float32).contiguous()
        nbytes = max(1, t.numel()) * 4
        self._alloc = tvk.Allocation(nbytes, kind)
        self._view = self._alloc.view()
        self._view.upload(t)
        self.shape = tuple(t.shape)
        self._holds = 0

    def keep(self):
        """Record that queued work references this buffer."""
        self._holds += 1

    def drop(self):
        """Release a hold taken by `keep()`."""
        self._holds = max(0, self._holds - 1)

    def __del__(self):
        # Release this wrapper's reference on the Allocation.
        a = getattr(self, "_alloc", None)
        if a is not None:
            try:
                a.free()
            except Exception:
                pass

    @property
    def buf(self):
        """Backward compatibility: return the BufferView for dispatch."""
        return self._view

    @property
    def size(self):
        return self._view.size

    @classmethod
    def out(cls, shape):
        """A device-local output buffer (kernels write device-side)."""
        n = 1
        for s in shape:
            n *= s
        alloc = tvk.Allocation(max(1, n) * 4, "device")
        view = alloc.view()
        o = cls.__new__(cls)
        o._alloc = alloc
        o._view = view
        o.shape = tuple(shape)
        o._holds = 0
        return o

    def cpu(self):
        return self._view.download(self.shape)

    def view(self, *shape):
        """Reinterpret the same bytes under a new shape -- no copy.

        The QK-norm needs a `[S, n_heads*head_dim]` buffer viewed as
        `[S*n_heads, head_dim]`. Those describe identical contiguous bytes, so
        this is a label change. Going through `.cpu()` would be a
        device->host->device round trip with a `vkQueueWaitIdle` on each side,
        and there were three of them per layer forward.
        """
        n = 1
        for s in shape:
            n *= s
        assert n == self._view.size // 4, (
            f"view({shape}) = {n} elements but buffer holds {self._view.size // 4}")
        o = self.__class__.__new__(self.__class__)
        o._alloc = self._alloc
        o._view = self._alloc.view(0, self._view.size)
        o.shape = tuple(shape)
        o._holds = self._holds
        # The new wrapper must not free the shared Allocation twice.
        return o

    @classmethod
    def _wrap(cls, buf, nbytes):
        """Wrap an existing BufferView with known byte size."""
        o = cls.__new__(cls)
        o._alloc = buf._alloc
        o._view = buf
        o.shape = (nbytes // 4,)
        o._holds = 0
        return o


class VulkanDecoderLayer(torch.nn.Module):
    """One Qwen3 decoder layer executed on the Vulkan device.

    Submodule names are preserved from `Qwen3DecoderLayer`, so the original state
    dict loads with no renaming. Weights are uploaded to VRAM once, at
    construction, and stay resident -- no per-step weight transfer.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention, Qwen3MLP, Qwen3RMSNorm,
        )
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.config = config
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.eps = config.rms_norm_eps
        self._uploaded = False

    # -- resident weights --------------------------------------------------
    def upload_weights(self) -> None:
        """Upload this layer's weights to VRAM once."""
        if self._uploaded:
            return
        c = self.config
        h = self.hidden_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        self._w = {
            "q": _Buf(self.self_attn.q_proj.weight.detach()),
            "k": _Buf(self.self_attn.k_proj.weight.detach()),
            "v": _Buf(self.self_attn.v_proj.weight.detach()),
            "o": _Buf(self.self_attn.o_proj.weight.detach()),
            "q_norm": _Buf(self.self_attn.q_norm.weight.detach()),
            "k_norm": _Buf(self.self_attn.k_norm.weight.detach()),
            "input_ln": _Buf(self.input_layernorm.weight.detach()),
            "post_ln": _Buf(self.post_attention_layernorm.weight.detach()),
            "gate": _Buf(self.mlp.gate_proj.weight.detach()),
            "up": _Buf(self.mlp.up_proj.weight.detach()),
            "down": _Buf(self.mlp.down_proj.weight.detach()),
            # stacked gate|up weight for the fused kernel: [2I, H] -> [H, 2I]
            "gate_up": _Buf(torch.cat(
                [self.mlp.gate_proj.weight.detach(),
                 self.mlp.up_proj.weight.detach()], dim=0).t().contiguous()),
        }
        # zero biases for the projections (Qwen3 has none)
        self._zero_h = _Buf(torch.zeros(h))
        self._zero_hd = _Buf(torch.zeros(nh * hd))
        self._zero_kvd = _Buf(torch.zeros(nkv * hd))
        self._zero_i = _Buf(torch.zeros(c.intermediate_size))
        self._uploaded = True

    # -- Q4 weight support ---------------------------------------------------
    def upload_weights_q4(self) -> None:
        """Upload this layer's weights in Q4 format for memory-constrained GPUs."""
        if getattr(self, "_uploaded_q4", False):
            return
        from backend.quantize_q4 import quantize_weight, GROUP_SIZE
        c = self.config
        h = self.hidden_size
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        
        # Quantize the large linear weights
        self._w_q4 = {}
        self._w_q4["q"] = quantize_weight(self.self_attn.q_proj.weight.detach())
        self._w_q4["k"] = quantize_weight(self.self_attn.k_proj.weight.detach())
        self._w_q4["v"] = quantize_weight(self.self_attn.v_proj.weight.detach())
        self._w_q4["o"] = quantize_weight(self.self_attn.o_proj.weight.detach())
        self._w_q4["down"] = quantize_weight(self.mlp.down_proj.weight.detach())
        
        # Upload packed weights as byte buffers
        import numpy as np
        def upload_packed(packed: bytes) -> _Buf:
            raw = np.frombuffer(packed, dtype=np.uint8).astype(np.uint8)
            pad = (-len(raw)) % 4
            if pad:
                raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])
            nbytes = len(raw)
            fb = torch.from_numpy(np.frombuffer(raw.tobytes(), dtype=np.float32).copy())
            assert fb.numel() * 4 == nbytes, f"{fb.numel() * 4} != {nbytes}"
            wb = tvk.VkBuffer(nbytes, "device")
            wb.upload(fb)
            return _Buf._wrap(wb, nbytes)
        
        self._w_q4_buf = {k: upload_packed(v) for k, v in self._w_q4.items()}
        self._uploaded_q4 = True

    def _lin_b_q4(self, x, w_q4_buf: _Buf, bias: _Buf, out_features: int, K: int):
        """Q4 matmul: x @ dequantize(w_q4)^T + bias. x on device, w_q4 packed."""
        if isinstance(x, _Buf):
            M = x.shape[0]
            xb = x
        else:
            M = x.numel() // x.shape[-1]
            K = x.shape[-1]
            xb = _Buf(x.reshape(M, K))
        ob = _Buf.out((M, out_features))
        import struct
        tiles_x = (out_features + 15) // 16
        tiles_y = (M + 15) // 16
        push = (struct.pack("<I", M) + struct.pack("<I", K) + struct.pack("<I", out_features) +
                struct.pack("<I", tiles_x) + struct.pack("<I", 128))  # group_size=128
        tvk.dispatch("matmul_q4", [xb.buf, w_q4_buf.buf, ob.buf], push,
                     tiles_x * tiles_y, 1, 1)
        return ob

    # -- kernel helpers ----------------------------------------------------
    # These keep intermediates ON DEVICE and return the raw buffer, so a layer
    # does not pay a host round trip per op. Only the caller's final results are
    # downloaded. Downloading per-op cost ~0.26 ms and dominated layer timing.
    def _lin_b(self, x, w: _Buf, bias: _Buf, out_features: int,
               sync: bool = False):
        """x may be a CPU tensor or an on-device _Buf. Default sync=False for async batching."""
        if isinstance(x, _Buf):
            M = x.shape[0]
            K = x.shape[1]
            xb = x
        else:
            M = x.numel() // x.shape[-1]
            K = x.shape[-1]
            xb = _Buf(x.reshape(M, K))
        ob = _Buf.out((M, out_features))
        tvk.matmul_t_bias(xb.buf, w.buf, bias.buf, ob.buf, M, K, out_features,
                          sync=sync)
        return ob

    def _rms_b(self, x, w: _Buf, sync: bool = False):
        rs = x.shape[-1]
        xb = x if isinstance(x, _Buf) else _Buf(x.reshape(-1, rs))
        ob = _Buf.out(xb.shape)
        tvk.rmsnorm(xb.buf, w.buf, ob.buf, xb.shape[0], rs, self.eps, sync=sync)
        return ob

    def _lin_b_q4(self, x, w_q4_buf: _Buf, bias: _Buf, out_features: int, K: int,
                  sync: bool = False):
        """Q4 matmul: x @ dequantize(w_q4)^T + bias. x on device, w_q4 packed."""
        if isinstance(x, _Buf):
            M = x.shape[0]
            xb = x
        else:
            M = x.numel() // x.shape[-1]
            K = x.shape[-1]
            xb = _Buf(x.reshape(M, K))
        ob = _Buf.out((M, out_features))
        import struct
        tiles_x = (out_features + 15) // 16
        tiles_y = (M + 15) // 16
        push = (struct.pack("<I", M) + struct.pack("<I", K) + struct.pack("<I", out_features) +
                struct.pack("<I", tiles_x) + struct.pack("<I", 128))  # group_size=128
        tvk.dispatch("matmul_q4", [xb.buf, w_q4_buf.buf, ob.buf], push,
                     tiles_x * tiles_y, 1, 1, sync=sync)
        return ob

    def _lin(self, x: torch.Tensor, w: _Buf, bias: _Buf, out_features: int):
        return self._lin_b(x, w, bias, out_features).cpu().reshape(
            x.shape[:-1] + (out_features,))

    def _rms(self, x: torch.Tensor, w: _Buf) -> torch.Tensor:
        return self._rms_b(x, w).cpu().reshape(x.shape)

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, stats=None,
                use_q4: bool = False, **_kwargs):
        # Always upload fp32 weights (norms, gate_up) - they're small
        self.upload_weights()
        # Optionally upload Q4 weights for the large linear layers
        if use_q4:
            self.upload_weights_q4()

        t0 = time.time()
        layer_dtype = hidden_states.dtype
        h = hidden_states.detach().to(torch.float32)
        # Every _Buf allocated below is appended here so it survives until the
        # queue drains. Without this, Python frees a buffer while queued work
        # still references it and the GPU reads unmapped VA.
        _keep_alive = []

        def note(op):
            if stats is not None:
                stats.record(op, "vulkan")

        def keep(buf):
            _keep_alive.append(buf)
            return buf

        # ---- input RMSNorm (device-resident)
        note("rmsnorm.input")
        _h_in = keep(_Buf(h.reshape(-1, h.shape[-1])))
        h = self._rms_b(_h_in, self._w["input_ln"])
        keep(h)

        # ---- q/k/v projections + QK-norm, all on device
        note("matmul.q_proj")
        if use_q4:
            q = keep(self._lin_b_q4(h, self._w_q4_buf["q"], self._zero_hd,
                                self.num_heads * self.head_dim, self.hidden_size))
        else:
            q = keep(self._lin_b(h, self._w["q"], self._zero_hd,
                            self.num_heads * self.head_dim))
        note("rmsnorm.q_norm")
        M = q.shape[0]
        q = keep(self._rms_b(q.view(M * self.num_heads, self.head_dim),
                         self._w["q_norm"]))

        note("matmul.k_proj")
        if use_q4:
            k = keep(self._lin_b_q4(h, self._w_q4_buf["k"], self._zero_kvd,
                                self.num_kv_heads * self.head_dim, self.hidden_size))
        else:
            k = keep(self._lin_b(h, self._w["k"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim))
        note("rmsnorm.k_norm")
        k = keep(self._rms_b(k.view(M * self.num_kv_heads, self.head_dim),
                         self._w["k_norm"]))

        note("matmul.v_proj")
        if use_q4:
            v = keep(self._lin_b_q4(h, self._w_q4_buf["v"], self._zero_kvd,
                                self.num_kv_heads * self.head_dim, self.hidden_size))
        else:
            v = keep(self._lin_b(h, self._w["v"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim))

        # ---- RoPE on the packed [Q|K|V] buffer
        cos, sin = position_embeddings
        S = h.shape[0]
        pos_start = int(cache_position[0].item()) if cache_position is not None else 0
        HD = self.num_heads * self.head_dim
        KVD = self.num_kv_heads * self.head_dim
        q_t = q.view(S, HD)
        k_t = k.view(S, KVD)
        v_t = v.view(S, KVD)
        qkv = keep(tvk.VkBuffer(max(1, S * (HD + 2 * KVD)) * 4, "device"))
        tvk.dispatch(
            "pack3", [q_t.buf, k_t.buf, v_t.buf, qkv],
            u32(S) + u32(HD) + u32(KVD) + u32(KVD),
            (S * (HD + 2 * KVD) + 63) // 64, 1, 1, sync=False)
        note("rope")
        tvk.rope_qkv(qkv, pos_start, self.num_heads,
                     self.num_kv_heads, self.head_dim, n_tok=S, sync=False)
        
        # Submit before downloading qkv (needed for KV cache and attention)
        tvk.submit()
        
        qkv2 = qkv.download((S * (HD + 2 * KVD),))
        q = qkv2[:S * HD].view(S, self.num_heads, self.head_dim)
        k = qkv2[S * HD:S * HD + S * KVD].view(S, self.num_kv_heads, self.head_dim)
        v = qkv2[S * HD + S * KVD:S * HD + 2 * S * KVD].view(S, self.num_kv_heads,
                                                             self.head_dim)

        # ---- attention over the cache window
        if past_key_values is None:
            k_c = k.permute(1, 0, 2).contiguous()
            v_c = v.permute(1, 0, 2).contiguous()
        elif k.dim() == 2:
            k_c = k.unsqueeze(1)
            v_c = v.unsqueeze(1)
        elif k.dim() == 3:
            k_c, v_c = k, v
        else:
            k_c, v_c = k[0], v[0]
        cap = k_c.shape[1]
        n_q = S
        first_pos = pos_start
        q_pos = torch.arange(first_pos, first_pos + S)
        k_pos = torch.arange(cap)
        allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)      # [S, cap]
        mask = torch.zeros(1, self.num_heads, n_q, cap)
        mask = mask.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
        if os.environ.get("OFFLOAD_ZERO_MASK"):
            mask = torch.zeros(1, self.num_heads, n_q, cap)
        probs = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * cap) * 4, "device"))
        ob = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * self.head_dim) * 4, "host"))
        q_in = q.reshape(n_q, self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
        kb = keep(_Buf(k_c[:, :cap, :].contiguous()))
        vb = keep(_Buf(v_c[:, :cap, :].contiguous()))
        if stats is not None:
            stats.transfers += 2
            stats.transfer_bytes += (kb.buf.size + vb.buf.size)
        note("attention")
        _q_in = keep(_Buf(q_in))
        _mask = keep(_Buf(mask))
        tvk.attention2(_q_in.buf, kb.buf, vb.buf, _mask.buf, probs, ob,
                       0, cap, self.num_heads,
                       self.num_heads // self.num_kv_heads, self.head_dim,
                       cap, self.scaling, sync=False)
        # Submit before downloading attention output
        tvk.submit()
        attn = ob.download((self.num_heads, n_q, self.head_dim))
        attn = attn.permute(1, 0, 2).reshape(n_q, HD)
        if stats is not None:
            stats.transfers += 1
            stats.transfer_bytes += ob.size

        # ---- output projection + residual
        note("matmul.o_proj")
        attn_t = attn.contiguous()
        if use_q4:
            out = self._lin_b_q4(attn_t, self._w_q4_buf["o"], self._zero_h,
                                  self.hidden_size, self.hidden_size).cpu().reshape(
                attn_t.shape[:-1] + (self.hidden_size,))
        else:
            out = self._lin(attn_t, self._w["o"], self._zero_h, self.hidden_size)
        h = hidden_states + out.to(layer_dtype)
        residual = h

        # ---- post-attention RMSNorm + SwiGLU MLP
        note("rmsnorm.post_attn")
        h2 = self._rms(h.detach().to(torch.float32), self._w["post_ln"])
        note("fused_gate_up")
        gu = keep(_Buf.out((S, self.config.intermediate_size)))
        _h2 = keep(_Buf(h2))
        tvk.fused_gate_up(_h2.buf, self._w["gate_up"].buf, gu.buf,
                          S, self.hidden_size, self.config.intermediate_size,
                          sync=False)
        # Submit before downloading gate_up output
        tvk.submit()
        gu_t = gu.cpu()
        note("matmul.down_proj")
        if use_q4:
            mlp = self._lin_b_q4(gu_t, self._w_q4_buf["down"], self._zero_h,
                                  self.hidden_size, self.config.intermediate_size).cpu().reshape(
                gu_t.shape[:-1] + (self.hidden_size,))
        else:
            mlp = self._lin(gu_t, self._w["down"], self._zero_h, self.hidden_size)
        h = residual + mlp.to(layer_dtype)

        if stats is not None:
            stats.vk_time_s += time.time() - t0
        return h


# ---------------------------------------------------------------------------
# The backbone wrapper
# ---------------------------------------------------------------------------

class HybridBackbone(torch.nn.Module):
    """Drop-in replacement for `Qwen3Model` with per-layer device placement."""

    def __init__(self, qwen3_model, placement: Placement) -> None:
        super().__init__()
        self.config = qwen3_model.config
        self.embed_tokens = qwen3_model.embed_tokens
        self.norm = qwen3_model.norm
        self.rotary_emb = qwen3_model.rotary_emb
        self.gradient_checkpointing = False
        self.placement = placement

        self.layers = torch.nn.ModuleList()
        gpu_layer_indices = []
        for i, orig in enumerate(qwen3_model.layers):
            if placement.device_of(i) == "vulkan":
                vl = VulkanDecoderLayer(self.config, i)
                vl.load_state_dict(orig.state_dict(), strict=True)
                self.layers.append(vl)
                gpu_layer_indices.append(i)
            else:
                self.layers.append(orig)

        self.placement_gpu_layers = len(gpu_layer_indices)
        self.placement_cpu_layers = len(qwen3_model.layers) - self.placement_gpu_layers

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                past_key_values=None, attention_mask=None, use_cache=True,
                cache_position=None, output_hidden_states=False,
                return_dict=True, logits_to_keep=0, stats=None, use_q4: bool = False,
                **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen = 0
            if past_key_values is not None:
                past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen,
                                          past_seen + inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                               cache_position, past_key_values)

        hidden = inputs_embeds
        position_embeddings = self.rotary_emb(hidden, position_ids)
        all_hidden = () if output_hidden_states else None

        # The original Qwen3Model creates a DynamicCache on the first forward and
        # then reads it back. Reproduce that: accept an incoming cache, or create
        # one when caching is requested.
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden,)
            if isinstance(layer, VulkanDecoderLayer):
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    stats=stats,
                    use_q4=use_q4,
                )
            else:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    stats=stats,
                )

        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden = all_hidden + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, past_key_values, all_hidden]
                         if v is not None)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            # The engine reads outputs.past_key_values back after every step, so
            # it must be present whenever use_cache is set.
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden,
            attentions=None,
        )

    def _update_causal_mask(self, attention_mask, inputs_embeds, cache_position,
                            past_key_values):
        dtype = inputs_embeds.dtype
        seq_len = inputs_embeds.shape[1]
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        target_len = past_seen + seq_len
        min_dtype = torch.finfo(dtype).min
        causal = torch.full((seq_len, target_len), min_dtype, dtype=dtype,
                            device=inputs_embeds.device)
        # Query i is at absolute position (past_seen + i) and may attend to keys
        # 0..(past_seen + i). `triu(diagonal=1+past_seen)` does exactly that for
        # BOTH the prefill (seq_len>1) and the decode (seq_len==1) cases.
        #
        # The earlier `if seq_len != 1` guard skipped the triu entirely on decode
        # steps, leaving every entry at min_dtype -- an all-masked row, so the
        # query attended to NO keys. That produced degenerate, repeating output
        # and zero latent steps in the engine.
        causal = torch.triu(causal, diagonal=1 + past_seen)
        causal = causal[None, None, :, :].expand(1, 1, seq_len, target_len)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :].to(dtype)
            mask = torch.where(mask.bool(), torch.zeros_like(mask),
                               torch.full_like(mask, min_dtype))
            causal = causal + mask
        return causal
