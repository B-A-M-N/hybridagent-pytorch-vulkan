"""
torch_vulkan_ext — raw-Vulkan transformer kernels for the HybridAgent backbone.

torch-vulkan provides PrivateUse1 registration, the Vulkan allocator and the
fp32 mm/add/mul/softmax/embedding/layernorm kernels. It does NOT wire up the
kernels a transformer layer needs: RoPE, RMSNorm, the SiLU gate, the persistent
KV cache, single-token attention, or argmax. This package supplies exactly
those, as SPIR-V compute shaders driven by a small raw-Vulkan dispatch layer.

Architecture note: the Vulkan backend is a plain C ABI library loaded with
ctypes (`_C.so`, built by CMake into `backend/torch_vulkan_ext/lib/`). It links
against libvulkan only — no Kompute, no Python headers — so it builds anywhere
glslangValidator and the Vulkan loader are present.

Kompute was deliberately not used here: its `TensorT<T>` with TensorTypes::eDevice
allocates a device-local buffer *plus a host-visible staging buffer of the same
size*, which would double VRAM for resident weights.
"""
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

import torch

_LIB_NAME = "tvkvk"

# ---------------------------------------------------------------------------
# ctypes bindings
# ---------------------------------------------------------------------------

def _find_lib() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("TVKVK_LIB"),
        os.path.join(here, "lib", f"{_LIB_NAME}.so"),
        os.path.join(here, "_C.so"),
        os.path.join(here, f"{_LIB_NAME}.so"),
        os.path.join(here, "..", "lib", f"{_LIB_NAME}.so"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not find the tvkvk shared library. Build it with "
        "`bash scripts/install.sh` (or cmake in backend/torch_vulkan_ext)."
    )


_lib = None
_lock = threading.Lock()

# Reusable host staging buffers. Every device->host and host->device transfer
# used to allocate and destroy a fresh VkBuffer; at ~1.2 ms per round trip for a
# 12x4096 tensor, and ~8 round trips per transformer layer, that dominated the
# layer time rather than the GPU work itself. Buffers are created once per size
# and reused for the process lifetime.
_STAGING: dict[int, object] = {}

# Buffers held alive by queued async dispatches; released in submit().
_pending_holds: list = []


def _staging(nbytes: int):
    """A persistent host-visible buffer of exactly `nbytes` (created once)."""
    b = _STAGING.get(nbytes)
    if b is None:
        b = VkBuffer(nbytes, "host")
        _STAGING[nbytes] = b
    return b


def _load():
    global _lib
    if _lib is not None:
        return _lib
    _lib = ctypes.CDLL(_find_lib())
    l = _lib
    l.tvk_init.argtypes = [ctypes.c_int]
    l.tvk_init.restype = ctypes.c_int
    l.tvk_device_name.restype = ctypes.c_char_p
    l.tvk_set_shader_dir.argtypes = [ctypes.c_char_p]
    l.tvk_vram_total.restype = ctypes.c_longlong
    l.tvk_vram_used.restype = ctypes.c_longlong
    l.tvk_dispatch_count.restype = ctypes.c_longlong
    l.tvk_set_max_descriptor_sets.argtypes = [ctypes.c_int]
    l.tvk_set_max_descriptor_sets.restype = None
    l.tvk_buf_device.argtypes = [ctypes.c_longlong]
    l.tvk_buf_device.restype = ctypes.c_void_p
    l.tvk_buf_host.argtypes = [ctypes.c_longlong]
    l.tvk_buf_host.restype = ctypes.c_void_p
    l.tvk_buf_free.argtypes = [ctypes.c_void_p]
    l.tvk_buf_size.argtypes = [ctypes.c_void_p]
    l.tvk_buf_size.restype = ctypes.c_longlong
    l.tvk_buf_vk.argtypes = [ctypes.c_void_p]
    l.tvk_buf_vk.restype = ctypes.c_ulonglong
    l.tvk_buf_map.argtypes = [ctypes.c_void_p]
    l.tvk_buf_map.restype = ctypes.c_void_p
    l.tvk_buf_unmap.argtypes = [ctypes.c_void_p]
    l.tvk_dispatch.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    l.tvk_dispatch.restype = ctypes.c_int
    l.tvk_dispatch_async.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    l.tvk_dispatch_async.restype = ctypes.c_int
    l.tvk_submit.argtypes = []
    l.tvk_copy_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    l.tvk_copy_to_device.restype = ctypes.c_int
    l.tvk_copy_from_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    l.tvk_copy_from_device.restype = ctypes.c_int
    l.tvk_last_error.restype = ctypes.c_char_p
    return _lib


# ---------------------------------------------------------------------------
# Buffer abstraction
# ---------------------------------------------------------------------------

class VkBuffer:
    """A Vulkan buffer with an optional host-mapped view.

    `device` buffers are device-local. Uploading into one is staged through a
    host-visible staging buffer (a raw memmove into device-local memory would
    segfault, since that memory is not host-mapped).
    `host` buffers are host-visible and mapped, for staging in and out.

    Reference counting
    -------------------
    A queued (async) dispatch holds a raw pointer to this buffer's VkBuffer.
    If the Python object is garbage-collected while the work is still in flight,
    the underlying buffer is destroyed and the GPU reads freed memory -- which
    on RADV surfaces as `The CS has been cancelled because the context is lost`
    and corrupted results. `acquire()` / `release()` give async callers an
    explicit hold so the buffer cannot be destroyed under queued work.
    """

    __slots__ = ("_h", "_size", "_mapped", "kind", "_refs")

    def __init__(self, nbytes: int, kind: str = "device"):
        self._size = int(nbytes)
        lib = _load()
        if kind == "device":
            self._h = lib.tvk_buf_device(self._size)
        else:
            self._h = lib.tvk_buf_host(self._size)
        if not self._h:
            raise MemoryError(f"failed to allocate {self._size} bytes ({kind})")
        self.kind = kind
        self._mapped = None
        self._refs = 1

    def acquire(self):
        """Take a reference; the buffer will not be freed until `release()`."""
        self._refs += 1
        return self

    def release(self):
        """Drop a reference, freeing the buffer when the last one goes."""
        self._refs -= 1
        if self._refs <= 0:
            self._free()

    def _free(self):
        h = getattr(self, "_h", None)
        if h:
            try:
                _load().tvk_buf_free(h)
            except Exception:
                pass
            self._h = None

    @property
    def handle(self):
        return self._h

    @property
    def vk_handle(self) -> int:
        return _load().tvk_buf_vk(self._h)

    @property
    def size(self) -> int:
        return self._size

    @property
    def mapped(self):
        if self._mapped is None:
            self._mapped = _load().tvk_buf_map(self._h)
        return self._mapped

    def upload(self, tensor: torch.Tensor) -> None:
        """Copy a CPU tensor into this buffer.

        For a device-local destination the copy is staged through a reusable
        host buffer. Allocating a fresh staging buffer per call dominated layer
        timing (a 12x4096 upload cost ~0.24 ms, of which most was allocation and
        teardown), so a small pool of staging buffers is kept and reused.
        """
        src = tensor.detach().contiguous()
        if src.dtype != torch.float32:
            src = src.float()
        nbytes = src.numel() * 4
        assert nbytes == self._size, f"{nbytes} != {self._size}"
        if self.kind == "host":
            ctypes.memmove(self.mapped, src.numpy().tobytes(), nbytes)
            return
        stage = _staging(nbytes)
        ctypes.memmove(stage.mapped, src.numpy().tobytes(), nbytes)
        if _load().tvk_copy_to_device(self._h, stage.handle) != 0:
            raise RuntimeError("tvk_copy_to_device failed")

    def download_u32(self, shape) -> torch.Tensor:
        """Download as uint32 (for kernels that write indices, not floats)."""
        raw = self._read_u32()
        return torch.from_numpy(raw.copy()).reshape(shape).to(torch.int64)

    def _read_u32(self) -> np.ndarray:
        if self.kind == "host":
            raw = (ctypes.c_char * self._size).from_address(self.mapped)
            return np.frombuffer(bytes(bytearray(raw)), dtype=np.uint32)
        stage = VkBuffer(self._size, "host")
        try:
            if _load().tvk_copy_from_device(stage.handle, self._h) != 0:
                raise RuntimeError("tvk_copy_from_device failed")
            raw = (ctypes.c_char * self._size).from_address(stage.mapped)
            return np.frombuffer(bytes(bytearray(raw)), dtype=np.uint32)
        finally:
            del stage

    def download(self, shape, dtype=torch.float32) -> torch.Tensor:
        """Copy this buffer out into a new CPU tensor."""
        if self.kind == "host":
            raw = (ctypes.c_char * self._size).from_address(self.mapped)
            arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.float32)
            return torch.from_numpy(arr.copy()).reshape(shape).to(dtype)
        stage = _staging(self._size)
        if _load().tvk_copy_from_device(stage.handle, self._h) != 0:
            raise RuntimeError("tvk_copy_from_device failed")
        raw = (ctypes.c_char * self._size).from_address(stage.mapped)
        arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.float32)
        return torch.from_numpy(arr.copy()).reshape(shape).to(dtype)

    def __del__(self):
        # Only frees when the last reference goes; an async dispatch holding a
        # reference keeps the buffer alive until it calls release().
        try:
            self.release()
        except Exception:
            pass


def dispatch(shader: str, bufs, push: bytes = b"", gx=1, gy=1, gz=1,
             sync: bool = True) -> None:
    """Run a compute shader.

    With `sync=False` the work is queued but not waited for, so several
    dispatches can be issued back to back and drained with `submit()`. That
    removes the per-op host round trip, which dominated batched-layer timing.
    """
    with _lock:
        _dispatch(shader, bufs, push, gx, gy, gz, sync=sync)


def _dispatch(shader, bufs, push, gx, gy, gz, sync=True):
    """Queue a compute dispatch.

    Every buffer handed to the GPU is kept alive for the duration of the call.
    A `_Buf`'s `__del__` frees its VkBuffer; if Python collects the wrapper while
    the GPU still reads it, the kernel reads unmapped VA and silently returns
    zeros (or worse). Submission is not completion, so the reference is held
    here rather than at each call site.

    `acquire()` is also a *Python* reference for the duration of this function:
    an inline `_Buf(...)` argument is otherwise garbage-collected the moment the
    calling expression finishes, which returns its memory to the allocator while
    the dispatch is still being set up.
    """
    lib = _load()
    # Take a reference on every buffer for the whole dispatch, sync or async.
    # For sync dispatches the queue drains before we return, so the holds are
    # released immediately below; for async ones they stay until submit().
    #
    # `anchor()` covers the hazardous inline form `_Buf(t).buf`: the temporary
    # wrapper's refcount drops to zero at the end of that expression, BEFORE the
    # dispatch it feeds has been issued. The anchor holds the wrapper (and so its
    # VkBuffer) until the queue has drained.
    holds = [b.acquire() for b in bufs]
    handles = (ctypes.c_ulonglong * len(bufs))(*[b.vk_handle for b in bufs])
    push_buf = ctypes.create_string_buffer(push) if push else None
    fn = lib.tvk_dispatch if sync else lib.tvk_dispatch_async
    try:
        rc = fn(shader.encode(), handles, len(bufs), push_buf, len(push),
                gx, gy, gz)
    except Exception:
        for h in holds:
            h.release()
        raise
    if rc != 0:
        for h in holds:
            h.release()
        err = lib.tvk_last_error() or b"unknown error"
        raise RuntimeError(
            f"Vulkan dispatch failed for shader '{shader}': {err.decode(errors='replace')}")
    if sync:
        # The queue has drained, so the buffers are no longer referenced.
        for h in holds:
            h.release()
    else:
        _pending_holds.extend(holds)


def submit() -> None:
    """Wait for all queued asynchronous dispatches to complete."""
    with _lock:
        _load().tvk_submit()
        # The queue has drained, so the holds taken by async dispatches are no
        # longer needed.
        for h in _pending_holds:
            h.release()
        _pending_holds.clear()


# ---------------------------------------------------------------------------
# Push-constant structs (little-endian, packed to match the shaders)
# ---------------------------------------------------------------------------

import struct

def _pk(*fields) -> bytes:
    return b"".join(fields)

def u32(v) -> bytes: return struct.pack("<I", int(v))
def f32(v) -> bytes: return struct.pack("<f", float(v))


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------

def init(device_index: int = 0) -> None:
    """Create the Vulkan instance/device and load the shader directory."""
    lib = _load()
    if not lib.tvk_init(int(device_index)):
        raise RuntimeError(
            f"Failed to initialise Vulkan device {device_index}. "
            "Check `vulkaninfo` for a RADV/AMD device."
        )
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.environ.get("TVKVK_SHADER_DIR"),
        os.path.join(here, "..", "csrc", "shaders"),
        os.path.join(here, "shaders"),
    ):
        if cand and os.path.isdir(cand) and any(
            f.endswith(".spv") for f in os.listdir(cand)
        ):
            # C++ appends "<name>.spv" to this string, so it must end with "/".
            d = os.path.abspath(cand)
            lib.tvk_set_shader_dir((d.rstrip("/") + "/").encode())
            return
    raise FileNotFoundError("Could not locate compiled .spv shaders.")


def device_name() -> str:
    return (_load().tvk_device_name() or b"").decode()


def last_error() -> str:
    return (_load().tvk_last_error() or b"").decode(errors="replace")


def dispatch_count() -> int:
    return int(_load().tvk_dispatch_count())


def reset_counters() -> None:
    _load().tvk_reset_counters()


def set_max_descriptor_sets(n: int) -> None:
    """Set the descriptor-set pool size per shader (must precede any dispatch)."""
    _load().tvk_set_max_descriptor_sets(int(n))


def vram_total() -> int:
    return int(_load().tvk_vram_total())


def vram_used() -> int:
    return int(_load().tvk_vram_used())


def rmsnorm(x: VkBuffer, weight: VkBuffer, out: VkBuffer,
            rows: int, row_size: int, eps: float, sync: bool = True) -> None:
    push = u32(rows) + u32(row_size) + f32(eps)
    dispatch("rmsnorm", [x, weight, out], push, rows, 1, 1, sync=sync)


def gelu_erf(x: VkBuffer, out: VkBuffer, n: int) -> None:
    dispatch("gelu_erf", [x, out], u32(n), (n + 255) // 256, 1, 1)


def rope(data: VkBuffer, offset: int, head_dim: int, pos: int) -> None:
    push = u32(offset) + u32(head_dim) + u32(pos)
    dispatch("rope", [data], push, head_dim // 2, 1, 1)


def silu_gate(gate_up: VkBuffer, out: VkBuffer, N: int) -> None:
    dispatch("silu_gate", [gate_up, out], u32(N), (N + 255) // 256, 1, 1)


def residual_add(a: VkBuffer, b: VkBuffer, out: VkBuffer, n: int) -> None:
    dispatch("residual_add", [a, b, out], u32(n), (n + 255) // 256, 1, 1)


def broadcast_add(a: VkBuffer, b: VkBuffer, out: VkBuffer,
                  n: int, b_size: int, alpha: float) -> None:
    push = u32(n) + u32(b_size) + f32(alpha)
    dispatch("broadcast_add", [a, b, out], push, (n + 255) // 256, 1, 1)


def slice_(src: VkBuffer, out: VkBuffer, count: int, offset: int) -> None:
    push = u32(count) + u32(offset)
    dispatch("slice", [src, out], push, (count + 255) // 256, 1, 1)


def gather(src: VkBuffer, idx: VkBuffer, out: VkBuffer,
           count: int, offset: int) -> None:
    push = u32(count) + u32(offset)
    dispatch("gather", [src, idx, out], push, (count + 255) // 256, 1, 1)


def kv_store(kv: VkBuffer, k_cache: VkBuffer, v_cache: VkBuffer,
             pos: int, cap: int, head_dim: int, kv_len: int) -> None:
    push = u32(pos) + u32(cap) + u32(head_dim) + u32(kv_len)
    dispatch("kv_store", [kv, k_cache, v_cache], push,
             (2 * kv_len + 255) // 256, 1, 1)


def attention(q: VkBuffer, k_cache: VkBuffer, v_cache: VkBuffer, mask: VkBuffer,
              out: VkBuffer, pos: int, new_len: int, n_heads: int,
              heads_per_kv: int, head_dim: int, scaling: float) -> None:
    push = (u32(pos) + u32(new_len) + u32(n_heads) + u32(heads_per_kv) +
            u32(head_dim) + f32(scaling))
    dispatch("attention", [q, k_cache, v_cache, mask, out], push, n_heads, 1, 1)


def argmax(x: VkBuffer, out: VkBuffer, rows: int, cols: int) -> None:
    """Row-wise argmax over a [rows, cols] matrix; out[0] is the winning column.

    The shader uses a single 256-thread workgroup, so rows must be <= 256.
    """
    if rows > 256:
        raise ValueError(f"argmax supports rows <= 256, got {rows}")
    push = u32(rows) + u32(cols)
    dispatch("argmax", [x, out], push, 1, 1, 1)


def argmax_rows(x: VkBuffer, out: VkBuffer, rows: int, cols: int) -> None:
    """Per-row argmax: out[r] = argmax of row r. Requires rows <= 256."""
    if rows > 256:
        raise ValueError(f"argmax_rows supports rows <= 256, got {rows}")
    push = u32(rows) + u32(cols)
    dispatch("argmax_rows", [x, out], push, 1, 1, 1)




# ---------------------------------------------------------------------------
# Additional kernels (matmul family, layernorm, fused rope+kv, batched attn)
# ---------------------------------------------------------------------------

def layernorm(x: VkBuffer, w: VkBuffer, b: VkBuffer, out: VkBuffer,
              rows: int, row_size: int, eps: float = 1e-5) -> None:
    push = u32(rows) + u32(row_size) + f32(eps)
    dispatch("layernorm", [x, w, b, out], push, rows, 1, 1)


def matmul(a: VkBuffer, b: VkBuffer, out: VkBuffer, M: int, K: int, N: int,
           sync: bool = True) -> None:
    push = u32(M) + u32(K) + u32(N)
    dispatch("matmul", [a, b, out], push, (N + 15) // 16, (M + 15) // 16, 1, sync=sync)


def matmul_t_bias(x: VkBuffer, w: VkBuffer, bias: VkBuffer, out: VkBuffer,
                  M: int, K: int, N: int, sync: bool = True) -> None:
    """out = x @ w.T + bias, with w laid out [N, K] (the nn.Linear layout).

    The grid is flattened to 1-D (RADV ignores dispatch Y/Z); the shader
    reconstructs the tile indices from `tiles_x`.
    """
    tiles_x = (N + 15) // 16
    tiles_y = (M + 15) // 16
    push = u32(M) + u32(K) + u32(N) + u32(tiles_x)
    dispatch("matmul_t_bias", [x, w, bias, out], push, tiles_x * tiles_y, 1, 1,
             sync=sync)


def fused_gate_up(a: VkBuffer, w: VkBuffer, out: VkBuffer,
                  M: int, K: int, N: int, sync: bool = True) -> None:
    """out = silu(a @ Wg^T) * (a @ Wu^T), with Wg/Wu stacked as [K, 2N]."""
    push = u32(M) + u32(K) + u32(N)
    dispatch("fused_gate_up", [a, w, out], push, (N + 15) // 16, (M + 15) // 16, 1, sync=sync)


def rope_qkv(qkv: VkBuffer, pos: int, n_heads: int, n_kv_heads: int,
             head_dim: int, n_tok: int = 1, sync: bool = True) -> None:
    """Apply RoPE in place to Q and K inside a packed [Q|K|V] buffer.

    `n_tok` is the number of tokens packed in the buffer; the stride between
    heads is n_tok * head_dim per region, so it must be passed explicitly
    rather than assumed to be 1.
    """
    q_len = n_heads * head_dim
    kv_len = n_kv_heads * head_dim
    # `pos` is the absolute position of token 0; each token is rotated by
    # (pos + token_index).
    push = (u32(pos) + u32(n_heads) + u32(n_kv_heads) + u32(head_dim) +
            u32(q_len) + u32(kv_len) + u32(n_tok))
    total = n_tok * (n_heads + n_kv_heads) * (head_dim // 2)
    dispatch("rope_qkv", [qkv], push, (total + 63) // 64, 1, 1, sync=sync)


def kv_append(kv: VkBuffer, k_cache: VkBuffer, v_cache: VkBuffer,
              pos: int, cap: int, head_dim: int, kv_len: int,
              q_len: int = 0) -> None:
    """Append K and V from a packed [Q|K|V] buffer into the cache at slot `pos`.

    `q_len` is the offset of K inside `kv` (0 when K is at the start).
    """
    push = u32(pos) + u32(cap) + u32(head_dim) + u32(q_len) + u32(kv_len)
    dispatch("kv_append", [kv, k_cache, v_cache], push,
             (2 * kv_len + 255) // 256, 1, 1)


def rope_kv_store(qkv: VkBuffer, k_cache: VkBuffer, v_cache: VkBuffer,
                  pos: int, cap: int, n_heads: int, n_kv_heads: int,
                  head_dim: int) -> None:
    """Apply RoPE to Q/K then append K and V to the cache.

    Two dispatches: a Vulkan `barrier()` cannot order work *across* workgroups,
    so fusing RoPE with the cache write let some workgroups store stale values.
    """
    q_len = n_heads * head_dim
    rope_qkv(qkv, pos, n_heads, n_kv_heads, head_dim, n_tok=1)
    kv_append(qkv, k_cache, v_cache, pos, cap, head_dim, n_kv_heads * head_dim,
              q_len=q_len)


def attention2(q: VkBuffer, k_cache: VkBuffer, v_cache: VkBuffer, mask: VkBuffer,
               probs: VkBuffer, out: VkBuffer, seq_start: int, seq_len: int,
               n_heads: int, heads_per_kv: int, head_dim: int, cap: int,
               scaling: float, sync: bool = True) -> None:
    """Batched GQA attention over the window [seq_start, seq_start+seq_len).

    q is [batch, n_heads, seq_len, head_dim]; the cache is
    [batch, n_kv_heads, cap, head_dim]; mask is
    [batch, n_heads, seq_len, seq_len]. `probs` is a scratch buffer that also
    receives the normalised scores, which the tests compare directly.
    """
    if seq_len > 32768:
        raise ValueError(f"attention2 supports seq_len <= 32768, got {seq_len}")
    # `n_q` is the number of query positions per head; it is distinct from
    # seq_len, which is the key/value window. The head stride in q/out is
    # n_q * head_dim. Derive it from the q buffer size, which is
    # batch * n_heads * n_q * head_dim.
    q_elems = q.size // 4
    n_q = max(1, q_elems // (n_heads * head_dim))
    push = (u32(seq_start) + u32(seq_len) + u32(n_heads) + u32(heads_per_kv) +
            u32(head_dim) + u32(cap) + u32(n_q) + f32(scaling))
    # one workgroup per (batch, q_head)
    dispatch("attention2", [q, k_cache, v_cache, mask, probs, out], push,
             1, n_heads, 1, sync=sync)


__all__ = [
    "init", "device_name", "dispatch_count", "reset_counters",
    "vram_total", "vram_used", "VkBuffer",
    "rmsnorm", "gelu_erf", "rope", "silu_gate", "residual_add",
    "broadcast_add", "slice_", "gather", "kv_store", "attention", "argmax",
    "layernorm", "matmul", "matmul_t_bias", "fused_gate_up",
    "rope_qkv", "kv_append", "rope_kv_store", "attention2", "argmax_rows",
    "dispatch",
]
