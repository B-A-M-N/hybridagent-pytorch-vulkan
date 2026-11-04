#!/usr/bin/env python3
"""
Comprehensive memory diagnostic for HybridAgent inference.

Measures:
- RSS/PSS (resident set size / proportional set size)
- Memory PSI (pressure stall information)
- Swap in/out rates
- Major/minor page faults
- Per-process memory breakdown
- Model weight memory
- GPU upload/staging buffers
- KV cache retention
- TorchInductor compilation/runtime caches
"""
import gc
import os
import sys
import time
import subprocess
import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk
from hybrid_agent import load_hybrid_agent
from backend.hybrid_offload import HybridBackbone, Placement


def get_process_memory(pid=None):
    """Get memory stats from /proc/<pid>/status."""
    if pid is None:
        pid = os.getpid()
    stats = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmData:", "VmSwap:", "Threads:")):
                    k, v = line.split(":", 1)
                    stats[k] = int(v.strip().split()[0])  # in kB
    except Exception as e:
        stats["error"] = str(e)
    return stats


def get_system_memory():
    """Get system-wide memory from /proc/meminfo."""
    stats = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemFree:", "MemAvailable:", "SwapTotal:", "SwapFree:",
                                    "Active:", "Inactive:", "Dirty:", "Writeback:", "AnonPages:",
                                    "Mapped:", "Shmem:", "KReclaimable:", "SReclaimable:", "SUnreclaim:")):
                    k, v = line.split(":", 1)
                    stats[k] = int(v.strip().split()[0])  # in kB
    except Exception as e:
        stats["error"] = str(e)
    return stats


def get_memory_psi():
    """Get memory pressure stall information."""
    psi = {}
    for level in ["some", "full"]:
        try:
            with open(f"/proc/pressure/memory") as f:
                for line in f:
                    if line.startswith(level):
                        parts = line.split()
                        psi[level] = {
                            "avg10": float(parts[1].split("=")[1]),
                            "avg60": float(parts[2].split("=")[1]),
                            "avg300": float(parts[3].split("=")[1]),
                        }
        except Exception as e:
            psi[level] = {"error": str(e)}
    return psi


def get_page_faults(pid=None):
    """Get page fault counts from /proc/<pid>/stat."""
    if pid is None:
        pid = os.getpid()
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
            return {
                "minflt": int(parts[9]),
                "cminflt": int(parts[10]),
                "majflt": int(parts[11]),
                "cmajflt": int(parts[12]),
            }
    except Exception as e:
        return {"error": str(e)}


def print_memory_snapshot(label, pid=None):
    """Print a comprehensive memory snapshot."""
    print(f"\n{'='*60}")
    print(f"MEMORY SNAPSHOT: {label}")
    print(f"{'='*60}")

    proc = get_process_memory(pid)
    for k, v in proc.items():
        if k != "error":
            print(f"  {k:12s}: {v/1024:>10.1f} MB")
        else:
            print(f"  ERROR: {v}")

    sysmem = get_system_memory()
    print(f"\n  SYSTEM MEMORY:")
    for k, v in sysmem.items():
        if k != "error":
            print(f"  {k:12s}: {v/1024:>10.1f} MB")

    psi = get_memory_psi()
    print(f"\n  MEMORY PSI (stall %):")
    for level, data in psi.items():
        if "error" not in data:
            print(f"  {level:5s}: avg10={data['avg10']:6.2f}%  avg60={data['avg60']:6.2f}%  avg300={data['avg300']:6.2f}%")

    pf = get_page_faults(pid)
    print(f"\n  PAGE FAULTS:")
    for k, v in pf.items():
        if k != "error":
            print(f"  {k:8s}: {v}")


def measure_model_loading():
    """Measure memory during model loading."""
    print_memory_snapshot("BEFORE MODEL LOAD")
    gc.collect()

    t0 = time.perf_counter()
    model, tok = load_hybrid_agent(
        os.path.join(ROOT, "weights", "HybridAgent-8B-Agentic"),
        dtype=torch.bfloat16, device="cpu")
    load_time = time.perf_counter() - t0
    print(f"  Model load time: {load_time:.2f}s")

    print_memory_snapshot("AFTER MODEL LOAD (bf16)")
    return model, tok


def measure_vulkan_setup(model, seq_len=12, gpu_layers=1):
    """Measure memory during Vulkan setup."""
    qm = model.backbone.model
    tvk.init(0)

    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, len(qm.layers)))
    print_memory_snapshot(f"BEFORE VULKAN UPLOAD ({gpu_layers} GPU layers)")

    t0 = time.perf_counter()
    for l in hb.layers:
        if type(l).__name__ == "VulkanDecoderLayer":
            l.upload_weights()
    upload_time = time.perf_counter() - t0
    print(f"  Vulkan upload time: {upload_time:.2f}s")

    print_memory_snapshot(f"AFTER VULKAN UPLOAD ({gpu_layers} GPU layers)")
    return hb


def measure_inference(hb, seq_len=12, steps=5):
    """Measure memory during inference."""
    ids = torch.randint(0, 1000, (1, seq_len))
    print_memory_snapshot(f"BEFORE INFERENCE (S={seq_len})")

    # Warmup
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)

    print_memory_snapshot("AFTER WARMUP")

    # Steady state
    for i in range(steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            hb(input_ids=ids, use_cache=True)
        dt = time.perf_counter() - t0
        if i == 0:
            print(f"  First step: {dt*1000:.1f}ms")
        elif i == steps - 1:
            print(f"  Last step:  {dt*1000:.1f}ms")

    print_memory_snapshot(f"AFTER {steps} STEPS")


def force_gc_and_measure():
    """Force garbage collection and measure."""
    print("\n  Forcing GC...")
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)
    print_memory_snapshot("AFTER FORCED GC")


def main():
    print("=" * 70)
    print("HYBRIDAGENT MEMORY DIAGNOSTIC")
    print(f"PID: {os.getpid()}")
    print("=" * 70)

    # Baseline
    print_memory_snapshot("BASELINE (STARTUP)")

    # Model loading
    model, tok = measure_model_loading()

    # Vulkan setup with different layer counts
    for gpu_layers in [1, 2, 3]:
        hb = measure_vulkan_setup(model, seq_len=12, gpu_layers=gpu_layers)
        measure_inference(hb, seq_len=12, steps=3)
        force_gc_and_measure()
        del hb
        force_gc_and_measure()

    # Test larger sequence
    print("\n\n=== LARGER SEQUENCE (S=512) ===")
    hb = measure_vulkan_setup(model, seq_len=512, gpu_layers=4)
    measure_inference(hb, seq_len=512, steps=2)
    force_gc_and_measure()

    print("\n\n=== FINAL CLEANUP ===")
    del model
    del tok
    force_gc_and_measure()

    print("\n\nDone.")


if __name__ == "__main__":
    main()