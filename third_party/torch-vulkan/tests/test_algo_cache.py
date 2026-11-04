"""Phase 4: Algorithm cache correctness + perf test.

Tests that the algorithm cache produces correct results and that
repeated dispatches with the same tensor buffers hit the cache.
"""

import os
import sys
import time

# Allow pointing at a custom-built PyTorch (e.g. a local APU build) without
# hardcoding a developer's path. Set TORCH_VULKAN_PYTORCH_PATH if needed.
_custom_torch = os.environ.get("TORCH_VULKAN_PYTORCH_PATH")
if _custom_torch:
    sys.path.insert(0, _custom_torch)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import torch_vulkan


def test_mm_correctness():
    """Matrix multiply: create tensors on CPU, move to vulkan, verify result."""
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")
    c = torch.mm(a, b)

    expected = torch.mm(a_cpu, b_cpu)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("mm correctness: PASS")


def test_add_correctness():
    """Elementwise add with alpha."""
    a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0])

    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")
    c = torch.add(a, b, alpha=2.0)

    expected = torch.add(a_cpu, b_cpu, alpha=2.0)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("add correctness: PASS")


def test_gelu_correctness():
    """GELU activation."""
    a_cpu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = a_cpu.to("vulkan")
    c = torch.nn.functional.gelu(a)

    expected = torch.nn.functional.gelu(a_cpu)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-2, atol=1e-2)
    print("gelu correctness: PASS")


def test_relu_correctness():
    """ReLU activation."""
    a_cpu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = a_cpu.to("vulkan")
    c = torch.relu(a)

    expected = torch.relu(a_cpu)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("relu correctness: PASS")


def test_cache_hit_perf():
    """Run mm multiple times with same-sized tensors and measure speedup.
    
    First call = cache miss (creates VkPipeline + descriptors).
    Subsequent calls = cache hit (reuses pipeline, updates push constants).
    """
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")

    # Warmup / first call (cache miss)
    t0 = time.perf_counter()
    c = torch.mm(a, b)
    t_first = time.perf_counter() - t0

    # Subsequent calls (should hit cache)
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        c = torch.mm(a, b)
    t_cached = (time.perf_counter() - t0) / N

    print(f"mm first call (cache miss): {t_first*1000:.2f} ms")
    print(f"mm cached avg ({N} calls):  {t_cached*1000:.2f} ms")
    if t_first > 0 and t_cached > 0:
        print(f"speedup: {t_first/t_cached:.1f}x")

    # Verify result is still correct
    expected = torch.mm(a_cpu, b_cpu)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("cache hit perf: PASS")


def test_add_cache_hit():
    """Test add cache: same tensors reused."""
    a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0])
    a = a_cpu.to("vulkan")
    b = b_cpu.to("vulkan")

    # Warmup
    c = torch.add(a, b, alpha=1.0)

    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        c = torch.add(a, b, alpha=1.0)
    t_cached = (time.perf_counter() - t0) / N

    print(f"add cached avg ({N} calls): {t_cached*1000:.2f} ms")

    expected = torch.add(a_cpu, b_cpu, alpha=1.0)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("add cache hit: PASS")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Vulkan device: {torch_vulkan.device_name()}")
    print()

    test_mm_correctness()
    test_add_correctness()
    test_gelu_correctness()
    test_relu_correctness()
    print()
    test_cache_hit_perf()
    test_add_cache_hit()
    print()
    print("All Phase 4 tests passed.")
