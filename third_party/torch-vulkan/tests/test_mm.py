"""Smoke test: matrix multiply through the Vulkan backend."""

import torch

import torch_vulkan


def test_vulkan_available():
    assert torch_vulkan.is_available(), "No Vulkan device found"
    print(f"Vulkan device: {torch_vulkan.device_name()}")


def test_mm_square():
    N = 64
    a = torch.randn(N, N, device="vulkan")
    b = torch.randn(N, N, device="vulkan")
    c = torch.mm(a, b)

    # Verify against CPU
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_expected = torch.mm(a_cpu, b_cpu)
    c_actual = c.cpu()

    torch.testing.assert_close(c_actual, c_expected, rtol=1e-3, atol=1e-3)
    print(f"mm {N}x{N}: PASS")


def test_mm_small():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="vulkan")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="vulkan")
    c = torch.mm(a, b)

    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    torch.testing.assert_close(c.cpu(), expected)
    print("mm 2x2: PASS")


def test_relu():
    # relu IS wired to the Vulkan backend (relu.spv); verify both shape and
    # values against the CPU reference.
    a_cpu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = a_cpu.to("vulkan")
    b = torch.relu(a)
    assert b.shape == a.shape
    torch.testing.assert_close(b.cpu(), torch.relu(a_cpu), rtol=1e-3, atol=1e-3)
    print("relu: PASS")


def test_unimplemented_op_falls_back_to_cpu():
    # An op with no Vulkan impl (here: sign) should route through the boxed
    # CPU fallback registered for PrivateUse1 and still return a valid result.
    a = torch.randn(16, device="vulkan")
    b = torch.sign(a)
    assert b.shape == a.shape
    print("CPU fallback: PASS")


if __name__ == "__main__":
    test_vulkan_available()
    test_mm_small()
    test_mm_square()
    test_relu()
    test_unimplemented_op_falls_back_to_cpu()
    print("\nAll tests passed.")
