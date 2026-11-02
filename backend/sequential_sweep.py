#!/usr/bin/env python3
"""Sequential INT8+Vulkan sweep runner with memory budget checks."""
import argparse
import json
import os
import subprocess
import sys
import time
import gc

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

from backend.memory_budget import check_memory_budget, estimate_config_memory, get_memory_budget


def run_config(seq_len: int, gpu_layers: int, reps: int = 2, timeout: int = 1800) -> tuple[float | None, str | None]:
    """Run a single configuration in a fresh subprocess."""
    import sys
    
    script_path = os.path.join(os.path.dirname(__file__), "quantized_loader.py")
    
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{ROOT}:{ROOT}/third_party/HybridAgent:{ROOT}/backend/torch_vulkan_ext",
        "OMP_NUM_THREADS": "12",
    })
    
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "quantized_loader.py"),
        "--seq", str(seq_len),
        "--layers", str(gpu_layers),
        "--reps", str(reps),
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        
        # Parse result
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                ms = float(line.split()[1])
                return ms, None
        
        # No RESULT line found
        err = (proc.stderr or proc.stdout)[-500:]
        return None, f"FAILED: {err.strip()}"
        
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"EXCEPTION: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=12)
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    # Parse layers
    if "," in args.layers:
        layer_counts = [int(x) for x in args.layers.split(",")]
    else:
        layer_counts = [int(args.layers)]
    
    # Memory budget check
    budget = get_memory_budget()
    print(f"=== Memory Budget ===")
    print(f"  Total RAM: {budget.total_ram_gb:.1f} GB")
    print(f"  Available RAM: {budget.available_ram_gb:.1f} GB")
    print(f"  Swap: {budget.swap_total_gb:.1f} GB total, {budget.swap_free_gb:.1f} GB free")
    
    seq_len = args.seq
    reps = args.reps
    
    results = []
    print(f"\n=== INT8+Vulkan sweep at S={seq_len} ===", flush=True)
    
    for gpu_layers in [0, 1, 2, 3, 4, 5, 6]:
        # Memory budget check
        est_mem = estimate_config_memory(gpu_layers)
        can_run, status = check_memory_budget(f"S={args.seq} GPU={gpu_layers}", estimate_config_memory(gpu_layers))
        print(f"\n  GPU={gpu_layers}: est {estimate_config_memory(gpu_layers):.1f}GB - {status}")
        
        if not can_run:
            results.append({
                "gpu_layers": gpu_layers,
                "ms": None,
                "error": "INSUFFICIENT_MEMORY",
                "status": "SKIPPED"
            })
            continue
        
        print(f"  Running GPU={gpu_layers}...", flush=True)
        ms, error = run_config(args.seq, gpu_layers, args.reps)
        
        if error:
            results.append({"gpu_layers": gpu_layers, "ms": None, "error": error})
            print(f"  FAILED: {error}")
        else:
            results.append({"gpu_layers": gpu_layers, "ms": ms, "error": None})
            print(f"  GPU={gpu_layers}: {ms:.2f} ms")
        
        # Cleanup
        gc.collect()
        time.sleep(2)  # Let GPU settle
    
    # Summary
    print(f"\n=== SWEEP SUMMARY S={args.seq} ===")
    ok = [r for r in results if r["ms"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["ms"])
        cpu = next((r["ms"] for r in results if r["gpu_layers"] == 0), None)
        print(f"  best: {best['gpu_layers']} GPU layers -> {best['ms']:.2f} ms")
        if cpu:
            speedup = cpu / best['ms']
            print(f"  vs all-CPU {cpu:.2f} ms -> {speedup:.3f}x {'GPU WINS' if best['ms'] < cpu else 'CPU WINS'}")
    
    # Save results
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"seq_len": args.seq, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=12)
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main()