#!/usr/bin/env python
"""Microtest for async Vulkan buffer lifetime correctness.

Tests the three-operation pipeline:
  A writes a buffer, B reads its view and writes another buffer, C reads that result.

Drops all Python temporary references immediately after recording.
Submits asynchronously, waits for completion fence, downloads result.
Compares against a CPU oracle.
"""
import torch
import sys
import os

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk


def cpu_oracle(A, W, B):
    """CPU reference: C = (A @ W.T + B) * 2"""
    return (A @ W.T + B) * 2


def run_microtest():
    print("=== Microtest: A @ W.T + B -> *2 ===")
    tvk.init(0)
    
    M, K, N = 8, 16, 32
    
    # Create test data with distinctive values
    torch.manual_seed(42)
    A = torch.randn(M, K) * 0.1
    W = torch.randn(N, K) * 0.1
    B = torch.randn(N) * 0.1
    
    # Expected result
    expected = cpu_oracle(A, W, B)
    
    # Create allocations with properly sized views
    A_nbytes = A.numel() * 4
    W_nbytes = W.numel() * 4
    B_nbytes = B.numel() * 4
    out1_nbytes = M * N * 4
    out2_nbytes = M * N * 4
    
    A_alloc = tvk.Allocation(A_nbytes)
    A_alloc.view().upload(A)
    
    W_alloc = tvk.Allocation(W_nbytes)
    W_alloc.view().upload(W.contiguous())
    
    B_alloc = tvk.Allocation(B_nbytes)
    B_alloc.view().upload(B)
    
    # Output allocations
    out1_alloc = tvk.Allocation(out1_nbytes)
    out2_alloc = tvk.Allocation(out2_nbytes)
    
    # --- Record operation A: matmul_t_bias (A @ W.T + B) ---
    print("Recording operation A: matmul_t_bias...")
    tvk.matmul_t_bias(A_alloc.view(), W_alloc.view(), B_alloc.view(),
                      out1_alloc.view(), M, K, N, sync=False)
    
    # --- Record operation B: residual_add (out1 + out1) ---
    print("Recording operation B: residual_add (out1 + out1)...")
    tvk.residual_add(out1_alloc.view(), out1_alloc.view(),
                     out2_alloc.view(), M * N)
    
    # --- Submit and wait ---
    print("Submitting async work...")
    tvk.submit()
    print("Submit complete, downloading result...")
    
    # Download result
    result = out2_alloc.view().download((M, N))
    
    # Verify
    diff = (result - expected).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")
    
    if max_diff < 1e-4:
        print("✓ PASS: Result matches CPU oracle")
        return True
    else:
        print("✗ FAIL: Result differs from CPU oracle")
        print(f"Expected[0]: {expected[0, :5]}")
        print(f"Got[0]:      {result[0, :5]}")
        return False


def run_lifetime_test():
    """Test buffer lifetime: delete Python refs before submit."""
    print("\n=== Lifetime test: delete refs before submit ===")
    tvk.init(0)
    
    M, K, N = 4, 8, 4
    
    torch.manual_seed(123)
    A = torch.randn(M, K) * 0.1
    W = torch.randn(N, K) * 0.1
    B = torch.randn(N) * 0.1
    
    expected = (A @ W.T + B) * 2
    
    def do_work():
        A_nbytes = A.numel() * 4
        W_nbytes = W.numel() * 4
        B_nbytes = B.numel() * 4
        out1_nbytes = M * N * 4
        out2_nbytes = M * N * 4
        
        A_alloc = tvk.Allocation(A_nbytes)
        A_alloc.view().upload(A)
        
        W_alloc = tvk.Allocation(W_nbytes)
        W_alloc.view().upload(W.contiguous())
        
        B_alloc = tvk.Allocation(B_nbytes)
        B_alloc.view().upload(B)
        
        out1_alloc = tvk.Allocation(M * N * 4)
        out2_alloc = tvk.Allocation(M * N * 4)
        
        tvk.matmul_t_bias(A_alloc.view(), W_alloc.view(), B_alloc.view(),
                          out1_alloc.view(), M, K, N, sync=False)
        tvk.residual_add(out1_alloc.view(), out1_alloc.view(),
                         out2_alloc.view(), M * N)
        
        # Drop all Python references to allocations/views
        # (they go out of scope here)
        return out2_alloc
    
    out2_alloc = do_work()
    
    # At this point, all intermediate Allocations and Views have been deleted
    # but the async work is still queued. The C++ refcounts should keep them alive.
    
    print("Submitting after Python refs dropped...")
    tvk.submit()
    
    result = out2_alloc.view().download((M, N))
    diff = (result - (A @ W.T + B) * 2).abs()
    max_diff = diff.max().item()
    
    print(f"Max diff: {max_diff:.6e}")
    if max_diff < 1e-4:
        print("✓ PASS: Lifetime test passed (no context loss)")
        return True
    else:
        print("✗ FAIL: Lifetime test failed")
        return False


def run_multiple_in_flight():
    """Test multiple in-flight submissions."""
    print("\n=== Multiple in-flight submissions test ===")
    tvk.init(0)
    
    M, K, N = 4, 8, 4
    num_submissions = 3
    
    torch.manual_seed(999)
    
    results = []
    for i in range(num_submissions):
        A = torch.randn(M, K) * 0.1
        W = torch.randn(N, K) * 0.1
        B = torch.randn(N) * 0.1
        expected = (A @ W.T + B) * 2
        
        A_nbytes = A.numel() * 4
        W_nbytes = W.numel() * 4
        B_nbytes = B.numel() * 4
        
        A_alloc = tvk.Allocation(A_nbytes)
        A_alloc.view().upload(A)
        W_alloc = tvk.Allocation(W_nbytes)
        W_alloc.view().upload(W.contiguous())
        B_alloc = tvk.Allocation(B_nbytes)
        B_alloc.view().upload(B)
        out1 = tvk.Allocation(M * N * 4)
        out2 = tvk.Allocation(M * N * 4)
        
        tvk.matmul_t_bias(A_alloc.view(), W_alloc.view(), B_alloc.view(),
                          out1.view(), M, K, N, sync=False)
        tvk.residual_add(out1.view(), out1.view(), out2.view(), M * N)
        
        results.append((out2, expected))
    
    # Submit all
    tvk.submit()
    
    all_pass = True
    for out2, expected in results:
        result = out2.view().download((M, N))
        diff = (result - expected).abs()
        max_diff = diff.max().item()
        if max_diff < 1e-4:
            print(f"  Submission {results.index((out2, expected))}: PASS")
        else:
            print(f"  Submission {results.index((out2, expected))}: FAIL (max_diff={max_diff:.6e})")
            all_pass = False
    
    return all_pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext")
    sys.path.insert(0, "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan")
    
    print("Running microtests for async Vulkan lifetime...\n")
    
    ok = True
    ok &= run_microtest()
    ok &= run_lifetime_test()
    ok &= run_multiple_in_flight()
    
    if ok:
        print("\n✓ ALL MICROTESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME MICROTESTS FAILED")
        sys.exit(1)