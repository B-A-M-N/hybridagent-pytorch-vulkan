#!/usr/bin/env python3
"""Run INT8+Vulkan sweep at S=12 using the direct loader."""
import subprocess
import json
import time
import gc

R = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"

def run_single_config(seq, layers, reps=2):
    """Run a single sweep configuration via subprocess."""
    cmd = [
        "timeout", "7200",
        ".venv/bin/python", "backend/quantized_loader_direct.py",
        "--seq", str(seq),
        "--layers", str(layers),
        "--reps", str(reps)
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "12",
        "PYTHONPATH": f"{R}:{R}/third_party/HybridAgent:{R}/backend/torch_vulkan_ext"
    }
    result = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=env)
    return result

import os

print("=== INT8+Vulkan sweep at S=12 ===", flush=True)
results = []

for g in [0, 1, 2, 3, 4, 5, 6]:
    print(f"\n  Testing gpu_layers={g}...", flush=True)
    try:
        result = run_single_config(12, g, 2)
        if result.returncode == 0:
            # Parse output for RESULT
            for line in result.stdout.split('\n'):
                if line.startswith("RESULT "):
                    ms = float(line.split()[1])
                    print(f"  gpu_layers={g}: {ms:.2f} ms", flush=True)
                    results.append({"gpu_layers": g, "ms": ms, "error": None})
                    break
            else:
                print(f"  gpu_layers={g}: FAILED - no RESULT in output", flush=True)
                print(f"    stdout: {result.stdout[-500:]}", flush=True)
                print(f"    stderr: {result.stderr[-500:]}", flush=True)
                results.append({"gpu_layers": g, "ms": None, "error": "no RESULT in output"})
        else:
            print(f"  gpu_layers={g}: FAILED (exit {result.returncode})", flush=True)
            print(f"    stderr: {result.stderr[-1000:]}", flush=True)
            results.append({"gpu_layers": g, "ms": None, "error": result.stderr[-500:]})
    except Exception as e:
        print(f"  gpu_layers={g}: FAILED - {e}", flush=True)
        results.append({"gpu_layers": g, "ms": None, "error": str(e)})
    
    gc.collect()
    time.sleep(3)

ok = [r for r in results if r["ms"] is not None]
if ok:
    best = min(ok, key=lambda r: r["ms"])
    cpu = next((r["ms"] for r in results if r["gpu_layers"] == 0), None)
    print(f"\n  best: {best['gpu_layers']} GPU layers -> {best['ms']:.2f} ms")
    if cpu:
        print(f"  vs all-CPU {cpu:.2f} ms -> {cpu/best['ms']:.3f}x {'GPU WINS' if best['ms'] < cpu else 'CPU WINS'}")

# Save results
out_path = "reports/int8_sweep_s12_direct.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"seq_len": 12, "results": results}, f, indent=2)
print(f"\nwrote {out_path}")