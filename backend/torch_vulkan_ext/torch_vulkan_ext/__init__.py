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
    # Allocations (opaque Allocation*)
    l.tvk_alloc_device.argtypes = [ctypes.c_longlong]
    l.tvk_alloc_device.restype = ctypes.c_void_p
    l.tvk_alloc_host.argtypes = [ctypes.c_longlong]
    l.tvk_alloc_host.restype = ctypes.c_void_p
    l.tvk_alloc_free.argtypes = [ctypes.c_void_p]
    # BufferViews (opaque BufferView*)
    l.tvk_buf_view.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong]
    l.tvk_buf_view.restype = ctypes.c_void_p
    l.tvk_buf_view_free.argtypes = [ctypes.c_void_p]
    l.tvk_buf_vk.argtypes = [ctypes.c_void_p]
    l.tvk_buf_vk.restype = ctypes.c_ulonglong
    l.tvk_buf_view_offset.argtypes = [ctypes.c_void_p]
    l.tvk_buf_view_offset.restype = ctypes.c_longlong
    l.tvk_buf_view_size.argtypes = [ctypes.c_void_p]
    l.tvk_buf_view_size.restype = ctypes.c_longlong
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
# Allocation / BufferView abstraction
# ---------------------------------------------------------------------------

class Allocation:
    """Owning handle to a Vulkan allocation (VkBuffer + VkDeviceMemory).
    
    Reference-counted on the C++ side. Multiple BufferViews can share
    the same Allocation. When the last view is freed, the Allocation
    is destroyed on the C++ side.
    """
    
    __slots__ = ("_h", "_size")
    
    def __init__(self, nbytes: int, kind: str = "device"):
        self._size = int(nbytes)
        lib = _load()
        if kind == "device":
            self._h = lib.tvk_alloc_device(self._size)
        else:
            self._h = lib.tvk_alloc_host(self._size)
        if not self._h:
            raise MemoryError(f"failed to allocate {self._size} bytes ({kind})")
    
    def view(self, offset: int = 0, size: int = None) -> "BufferView":
        """Create a view into this allocation (shared ownership)."""
        lib = _load()
        if size is None:
            size = self._size - offset
        v = lib.tvk_buf_view(self._h, offset, size)
        if not v:
            raise MemoryError(f"failed to create view offset={offset} size={size}")
        return BufferView(v, self)
    
    def free(self):
        """Explicitly free this allocation (decrements refcount)."""
        if self._h:
            lib = _load()
            lib.tvk_alloc_free(self._h)
            self._h = None
    
    @property
    def size(self) -> int:
        return self._size
    
    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class BufferView:
    """View into an Allocation (offset + size). Shared ownership of the Allocation.
    
    The C++ Allocation is reference-counted; this view holds a reference.
    """
    
    __slots__ = ("_h", "_alloc")
    
    def __init__(self, handle: int, alloc: Allocation):
        self._h = handle
        self._alloc = alloc  # keep Allocation alive
    
    @property
    def vk_handle(self) -> int:
        # Return the VkBuffer handle from the underlying allocation, not the view
        return _load().tvk_buf_vk(self._alloc._h)
    
    @property
    def offset(self) -> int:
        return _load().tvk_buf_view_offset(self._h)
    
    @property
    def size(self) -> int:
        return _load().tvk_buf_view_size(self._h)
    
    @property
    def mapped(self):
        """Map the underlying allocation (entire allocation).
        
        Returns the host-mapped pointer if the allocation is host-visible,
        None for device-local allocations.
        """
        return _load().tvk_buf_map(self._alloc._h)
    
    def upload(self, tensor: torch.Tensor) -> None:
        """Upload a tensor to this view's offset in the allocation."""
        src = tensor.detach().contiguous()
        if src.dtype != torch.float32:
            src = src.float()
        nbytes = src.numel() * 4
        assert nbytes <= self.size, f"tensor {nbytes} bytes exceeds view size {self.size}"
        
        # For host allocations (mapped), use direct memmove
        if self.mapped is not None:
            import ctypes
            ctypes.memmove(
                ctypes.c_void_p(self.mapped + self.offset),
                src.numpy().tobytes(), nbytes)
            return
        
        # For device allocations, use staging buffer
        lib = _load()
        stage = _staging(nbytes)
        stage._direct_upload(tensor)
        # Copy from staging to the device allocation at this view's offset
        # For full-allocation views (offset=0, size=allocation size), we can copy directly
        if self.offset == 0 and nbytes == self._alloc.size:
            # Fast path: view covers entire allocation
            if lib.tvk_copy_to_device(self._alloc._h, stage._alloc._h) != 0:
                raise RuntimeError("tvk_copy_to_device failed")
        else:
            # Subregion copy: create a temporary view at the correct offset
            temp_view = lib.tvk_buf_view(self._alloc._h, self.offset, nbytes)
            if not temp_view:
                raise MemoryError(f"failed to create temp view for upload")
            try:
                if lib.tvk_copy_to_device(temp_view, stage._alloc._h) != 0:
                    raise RuntimeError("tvk_copy_to_device failed")
            finally:
                lib.tvk_buf_view_free(temp_view)
    
    def _direct_upload(self, tensor: torch.Tensor) -> None:
        """Upload directly to this host allocation without recursion."""
        src = tensor.detach().contiguous()
        if src.dtype != torch.float32:
            src = src.float()
        nbytes = src.numel() * 4
        assert nbytes <= self.size, f"tensor {nbytes} bytes exceeds view size {self.size}"
        import ctypes
        ctypes.memmove(
            ctypes.c_void_p(self.mapped + self.offset),
            src.numpy().tobytes(), nbytes)
    
    def download(self, shape, dtype=torch.float32) -> torch.Tensor:
        """Download this view's data as a tensor."""
        # For host allocations, the memory is mapped directly
        mapped_ptr = self.mapped
        if mapped_ptr is not None:
            # Host allocation - memory is directly mapped
            raw = (ctypes.c_char * self._alloc.size).from_address(mapped_ptr)
            if dtype == torch.int64 or dtype == torch.uint32:
                arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.uint32)
            else:
                arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.float32)
            start = self.offset // 4
            end = start + self.size // 4
            return torch.from_numpy(arr[start:end].copy()).reshape(shape).to(dtype)
        
        # For device allocations, stage through a PERSISTENT host buffer.
        # Allocating a fresh host allocation per download caused vkAllocateMemory/
        # vkFreeMemory churn on every device->host read (several per layer per
        # step); the pool is keyed by size and reused. Downloads are synchronous,
        # so reuse is safe.
        stage_view = _staging(self.size)
        lib = _load()
        # tvk_copy_* take *Allocation* handles, not BufferView handles.
        if lib.tvk_copy_from_device(stage_view._alloc._h, self._alloc._h) != 0:
            raise RuntimeError("tvk_copy_from_device failed")
        raw = (ctypes.c_char * self.size).from_address(stage_view.mapped)
        if dtype == torch.int64 or dtype == torch.uint32:
            arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.uint32)
        else:
            arr = np.frombuffer(bytes(bytearray(raw)), dtype=np.float32)
        return torch.from_numpy(arr.copy()).reshape(shape).to(dtype)
    
    def download_u32(self, shape) -> torch.Tensor:
        """Download as uint32 (for kernels that write indices, not floats)."""
        return self.download(shape, dtype=torch.int64)
    
    def free(self):
        if self._h:
            lib = _load()
            lib.tvk_buf_view_free(self._h)
            self._h = None
            self._alloc = None
    
    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


_STAGING_ALLOCS: dict[int, Allocation] = {}

def _staging(nbytes: int) -> BufferView:
    """A persistent host-visible allocation of exactly `nbytes` (created once)."""
    alloc = _STAGING_ALLOCS.get(nbytes)
    if alloc is None:
        alloc = Allocation(nbytes, "host")
        _STAGING_ALLOCS[nbytes] = alloc
    return alloc.view()


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def dispatch(shader: str, views, push: bytes = b"", gx=1, gy=1, gz=1,
             sync: bool = True) -> None:
    """Run a compute shader.
    
    With `sync=False` the work is queued but not waited for, so several
    dispatches can be issued back to back and drained with `submit()`.
    """
    with _lock:
        _dispatch(shader, views, push, gx, gy, gz, sync=sync)


def _dispatch(shader, views, push, gx, gy, gz, sync=True):
    """Queue a compute dispatch.
    
    Every BufferView handed to the GPU is kept alive for the duration
    of the call via its internal Allocation reference.
    """
    lib = _load()
    holds = [v._alloc for v in views]  # keep Allocations alive
    handles = (ctypes.c_ulonglong * len(views))(*[v._h for v in views])  # pass BufferView* pointers
    push_buf = ctypes.create_string_buffer(push) if push else None
    fn = lib.tvk_dispatch if sync else lib.tvk_dispatch_async
    try:
        rc = fn(shader.encode(), handles, len(views), push_buf, len(push),
                gx, gy, gz)
    except Exception:
        for a in holds:
            pass  # Allocations are kept alive by BufferViews
        raise
    if rc != 0:
        err = lib.tvk_last_error() or b"unknown error"
        raise RuntimeError(
            f"Vulkan dispatch failed for shader '{shader}': {err.decode(errors='replace')}")
    # No manual release needed - BufferViews keep Allocations alive
    # until their Python lifetime ends


def submit() -> None:
    """Wait for all queued asynchronous dispatches to complete."""
    with _lock:
        _load().tvk_submit()


# ---------------------------------------------------------------------------
# Public ops (allocation-based API)
# ---------------------------------------------------------------------------

def init(device_index: int = 0) -> None:
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
    _load().tvk_set_max_descriptor_sets(int(n))


def vram_total() -> int:
    return int(_load().tvk_vram_total())


def vram_used() -> int:
    return int(_load().tvk_vram_used())


def rmsnorm(x: BufferView, weight: BufferView, out: BufferView,
            rows: int, row_size: int, eps: float, sync: bool = True) -> None:
    push = u32(rows) + u32(row_size) + f32(eps)
    dispatch("rmsnorm", [x, weight, out], push, rows, 1, 1, sync=sync)


def gelu_erf(x: BufferView, out: BufferView, n: int) -> None:
    dispatch("gelu_erf", [x, out], u32(n), (n + 255) // 256, 1, 1)


def rope(data: BufferView, offset: int, head_dim: int, pos: int) -> None:
    push = u32(offset) + u32(head_dim) + u32(pos)
    dispatch("rope", [data], push, head_dim // 2, 1, 1)


def silu_gate(gate_up: BufferView, out: BufferView, N: int) -> None:
    dispatch("silu_gate", [gate_up, out], u32(N), (N + 255) // 256, 1, 1)


def residual_add(a: BufferView, b: BufferView, out: BufferView, n: int) -> None:
    dispatch("residual_add", [a, b, out], u32(n), (n + 255) // 256, 1, 1)


def broadcast_add(a: BufferView, b: BufferView, out: BufferView,
                  n: int, b_size: int, alpha: float) -> None:
    push = u32(n) + u32(b_size) + f32(alpha)
    dispatch("broadcast_add", [a, b, out], push, (n + 255) // 256, 1, 1)


def slice_(src: BufferView, out: BufferView, count: int, offset: int) -> None:
    push = u32(count) + u32(offset)
    dispatch("slice", [src, out], push, (count + 255) // 256, 1, 1)


def gather(src: BufferView, idx: BufferView, out: BufferView,
           count: int, offset: int) -> None:
    push = u32(count) + u32(offset)
    dispatch("gather", [src, idx, out], push, (count + 255) // 256, 1, 1)


def kv_store(kv: BufferView, k_cache: BufferView, v_cache: BufferView,
             pos: int, cap: int, head_dim: int, kv_len: int) -> None:
    push = u32(pos) + u32(cap) + u32(head_dim) + u32(kv_len)
    dispatch("kv_store", [kv, k_cache, v_cache], push,
             (2 * kv_len + 255) // 256, 1, 1)


def attention(q: BufferView, k_cache: BufferView, v_cache: BufferView, mask: BufferView,
              out: BufferView, pos: int, new_len: int, n_heads: int,
              heads_per_kv: int, head_dim: int, scaling: float) -> None:
    push = (u32(pos) + u32(new_len) + u32(n_heads) + u32(heads_per_kv) +
            u32(head_dim) + f32(scaling))
    dispatch("attention", [q, k_cache, v_cache, mask, out], push, n_heads, 1, 1)


def argmax(x: BufferView, out: BufferView, rows: int, cols: int) -> None:
    if rows > 256:
        raise ValueError(f"argmax supports rows <= 256, got {rows}")
    push = u32(rows) + u32(cols)
    dispatch("argmax", [x, out], push, 1, 1, 1, sync=True)


def argmax_rows(x: BufferView, out: BufferView, rows: int, cols: int) -> None:
    if rows > 256:
        raise ValueError(f"argmax_rows supports rows <= 256, got {rows}")
    push = u32(rows) + u32(cols)
    dispatch("argmax_rows", [x, out], push, 1, 1, 1)


# ---------------------------------------------------------------------------
# Additional kernels
# ---------------------------------------------------------------------------

def layernorm(x: BufferView, w: BufferView, b: BufferView, out: BufferView,
              rows: int, row_size: int, eps: float = 1e-5) -> None:
    push = u32(rows) + u32(row_size) + f32(eps)
    dispatch("layernorm", [x, w, b, out], push, rows, 1, 1)


def matmul(a: BufferView, b: BufferView, out: BufferView, M: int, K: int, N: int,
           sync: bool = True) -> None:
    push = u32(M) + u32(K) + u32(N)
    dispatch("matmul", [a, b, out], push, (N + 15) // 16, (M + 15) // 16, 1, sync=sync)


def gemv(a: BufferView, x: BufferView, y: BufferView, M: int, K: int, group_size: int = 128,
           sync: bool = True) -> None:
    """GEMV: y[M] = A[M, K] @ x[K] with symmetric 4-bit quantized x.
    
    A: fp32 matrix [M, K]
    x: packed uint [K] with fp16 scale per group of group_size elements
    y: fp32 output [M]
    
    row_stride = (K // group_size) * (group_size // 2 + 2)  # bytes per row
    """
    tiles_x = (K + 15) // 16  # TILE = 16
    row_stride = (K // group_size) * (group_size // 2 + 2)
    push = u32(M) + u32(K) + u32(tiles_x) + u32(group_size) + u32(row_stride)
    dispatch("gemv_q4", [a, x, y], push, (M + 15) // 16, 1, 1, sync=sync)

def matmul_t_bias(x: BufferView, w: BufferView, bias: BufferView, out: BufferView,
                  M: int, K: int, N: int, sync: bool = True) -> None:
    tiles_x = (N + 15) // 16
    tiles_y = (M + 15) // 16
    push = u32(M) + u32(K) + u32(N) + u32(tiles_x)
    dispatch("matmul_t_bias", [x, w, bias, out], push, tiles_x * tiles_y, 1, 1,
             sync=sync)


def fused_gate_up(a: BufferView, w: BufferView, out: BufferView,
                  M: int, K: int, N: int, sync: bool = True) -> None:
    push = u32(M) + u32(K) + u32(N)
    dispatch("fused_gate_up", [a, w, out], push, (N + 15) // 16, (M + 15) // 16, 1, sync=sync)


def rope_qkv(qkv: BufferView, pos: int, n_heads: int, n_kv_heads: int,
             head_dim: int, n_tok: int = 1, sync: bool = True) -> None:
    q_len = n_heads * head_dim
    kv_len = n_kv_heads * head_dim
    push = (u32(pos) + u32(n_heads) + u32(n_kv_heads) + u32(head_dim) +
            u32(q_len) + u32(kv_len) + u32(n_tok))
    total = n_tok * (n_heads + n_kv_heads) * (head_dim // 2)
    dispatch("rope_qkv", [qkv], push, (total + 63) // 64, 1, 1, sync=sync)


def kv_append(kv: BufferView, k_cache: BufferView, v_cache: BufferView,
              pos: int, cap: int, head_dim: int, kv_len: int,
              q_len: int = 0) -> None:
    push = u32(pos) + u32(cap) + u32(head_dim) + u32(q_len) + u32(kv_len)
    dispatch("kv_append", [kv, k_cache, v_cache], push,
             (2 * kv_len + 255) // 256, 1, 1)


def rope_kv_store(qkv: BufferView, k_cache: BufferView, v_cache: BufferView,
                  pos: int, cap: int, n_heads: int, n_kv_heads: int,
                  head_dim: int) -> None:
    q_len = n_heads * head_dim
    rope_qkv(qkv, pos, n_heads, n_kv_heads, head_dim, n_tok=1)
    kv_append(qkv, k_cache, v_cache, pos, cap, head_dim, n_kv_heads * head_dim,
              q_len=q_len)


def attention2(q: BufferView, k_cache: BufferView, v_cache: BufferView, mask: BufferView,
               probs: BufferView, out: BufferView, seq_start: int, seq_len: int,
               n_heads: int, heads_per_kv: int, head_dim: int, cap: int,
               scaling: float, sync: bool = True) -> None:
    if seq_len > 32768:
        raise ValueError(f"attention2 supports seq_len <= 32768, got {seq_len}")
    q_elems = q.size // 4
    n_q = max(1, q_elems // (n_heads * head_dim))
    push = (u32(seq_start) + u32(seq_len) + u32(n_heads) + u32(heads_per_kv) +
            u32(head_dim) + u32(cap) + u32(n_q) + f32(scaling))
    dispatch("attention2", [q, k_cache, v_cache, mask, probs, out], push,
             1, n_heads, 1, sync=sync)


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
    _load().tvk_set_max_descriptor_sets(int(n))


def vram_total() -> int:
    return int(_load().tvk_vram_total())


def vram_used() -> int:
    return int(_load().tvk_vram_used())


def copy_to_device(dst: Allocation, src: Allocation) -> None:
    """Copy from host allocation to device allocation."""
    _load().tvk_copy_to_device(dst._h, src._h)


def copy_from_device(dst: Allocation, src: Allocation) -> None:
    """Copy from device allocation to host allocation."""
    _load().tvk_copy_from_device(dst._h, src._h)


# ---------------------------------------------------------------------------
# Backward compatibility: VkBuffer class (deprecated, use Allocation/BufferView)
# ---------------------------------------------------------------------------

class VkBuffer:
    """Deprecated: Use Allocation/BufferView instead.
    
    This class provides backward compatibility for code using the old VkBuffer API.
    """
    
    def __init__(self, nbytes: int, kind: str = "device"):
        self._alloc = Allocation(nbytes, kind)
        self._view = self._alloc.view()
        self.kind = kind
        self._size = nbytes
    
    @property
    def handle(self):
        return self._view._h
    
    @property
    def size(self):
        return self._view.size
    
    def upload(self, tensor):
        self._view.upload(tensor)
    
    def download(self, shape, dtype=torch.float32):
        return self._view.download(shape, dtype)
    
    def release(self):
        self._alloc.free()
    
    def acquire(self):
        pass  # No-op for compatibility
    
    @property
    def vk_handle(self):
        return self._view.vk_handle
    
    @property
    def _h(self):
        """Backward compatibility: return the BufferView handle."""
        return self._view._h
    
    def view(self, offset: int = 0, size: int = None) -> "BufferView":
        """Create a view into this buffer."""
        return self._alloc.view(offset, size)
    
    def mapped(self):
        return self._view.mapped
    
    def download_u32(self, shape):
        """Download as uint32 (for kernels that write indices, not floats)."""
        return self._view.download(shape, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "init", "device_name", "dispatch_count", "reset_counters",
    "vram_total", "vram_used", "Allocation", "BufferView", "VkBuffer",
    "copy_to_device", "copy_from_device",
    "rmsnorm", "gelu_erf", "rope", "silu_gate", "residual_add",
    "broadcast_add", "slice_", "gather", "kv_store", "attention", "argmax",
    "layernorm", "matmul", "matmul_t_bias", "fused_gate_up",
    "rope_qkv", "kv_append", "rope_kv_store", "attention2", "argmax_rows",
    "dispatch", "submit",
]