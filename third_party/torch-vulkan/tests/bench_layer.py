"""Phase 4: Benchmark simulating transformer layer ops.

Measures per-layer overhead with and without algorithm cache.
Focuses on the hot ops: mm, add, gelu that repeat across layers.
"""

import os
import sys
import time

# Allow pointing at a custom-built PyTorch without hardcoding a developer's
# path. Set TORCH_VULKAN_PYTORCH_PATH if needed.
_custom_torch = os.environ.get("TORCH_VULKAN_PYTORCH_PATH")
if _custom_torch:
    sys.path.insert(0, _custom_torch)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import torch_vulkan


def bench_mm_repeated(M, K, N, iters=20):
    """Benchmark mm with same-sized tensors (simulates weight * activation)."""
    a_cpu = torch.randn(M, K)
    b_cpu = torch.randn(K, N)
    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")

    # Warmup (first call = cache miss)
    torch.mm(a, b)

    # Benchmark (subsequent calls should hit cache)
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    elapsed = (time.perf_counter() - t0) / iters
    print(f"  mm [{M}x{K}] @ [{K}x{N}]: {elapsed*1000:.2f} ms/call")
    return elapsed


def bench_add_repeated(N, iters=20):
    """Benchmark add with same-sized tensors."""
    a_cpu = torch.randn(N)
    b_cpu = torch.randn(N)
    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")

    torch.add(a, b)  # warmup

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.add(a, b)
    elapsed = (time.perf_counter() - t0) / iters
    print(f"  add [{N}]: {elapsed*1000:.2f} ms/call")
    return elapsed


def bench_gelu_repeated(N, iters=20):
    """Benchmark gelu with same-sized tensors."""
    a_cpu = torch.randn(N)
    a = a_cpu.to("vulkan")

    torch.nn.functional.gelu(a)  # warmup

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.gelu(a)
    elapsed = (time.perf_counter() - t0) / iters
    print(f"  gelu [{N}]: {elapsed*1000:.2f} ms/call")
    return elapsed


def bench_simulated_layer():
    """Simulate a transformer layer's GPU ops.
    
    A typical transformer layer does:
    - 4x mm (Q, K, V projections + output projection)
    - 2x add (bias additions)
    - 1x gelu (feedforward activation)
    
    Total = 7 GPU dispatches per layer.
    """
    hidden = 128  # small for testing
    
    print(f"\nSimulated layer (hidden_dim={hidden}):")
    print("  Per-op timings (cached, after warmup):")
    
    total = 0.0
    
    # 4x mm
    for i in range(4):
        total += bench_mm_repeated(hidden, hidden, hidden)
    
    # 2x add
    for i in range(2):
        total += bench_add_repeated(hidden * hidden)
    
    # 1x gelu
    total += bench_gelu_repeated(hidden * hidden)
    
    print(f"\n  Layer total (7 ops): {total*1000:.2f} ms")
    print(f"  Per-op average: {total/7*1000:.2f} ms")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Vulkan: {torch_vulkan.device_name()}")
    
    bench_simulated_layer()
    
    print("\nCache stats (from VulkanContext):")
    stats = torch_vulkan.cache_stats()
    print(f"  Algorithm cache hits:   {stats['algo_hits']}")
    print(f"  Algorithm cache misses: {stats['algo_misses']}")
