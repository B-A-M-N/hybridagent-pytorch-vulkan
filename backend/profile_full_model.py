"""
Full-model placement & performance profiler for HybridAgent.

Evaluates multiple CPU/GPU partitions and reports:
- Model-load memory (RAM + VRAM)
- Prefill latency at 512, 2K, 8K tokens
- Sustained latent decode throughput (100+ steps)
- Wall-clock latency per step
- Correctness verification (logits diff vs CPU)

Usage:
    PYTHONPATH=... .venv/bin/python -m backend.profile_full_model \\
        --checkpoints 8B,14B \\
        --gpu-layers 0,2,4,6,8,12 \\
        --prefill-lens 512,2048,8192 \\
        --decode-steps 100
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "third_party", "HybridAgent"))
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

from backend.hybrid_offload import HybridBackbone, Placement
from backend.placement_controller import PlacementController


@dataclass
class ProfileResult:
    """Result of one profile run."""
    checkpoint: str
    seq_len: int
    gpu_layers: int
    use_q4: bool
    prefill_seconds: float
    decode_steps_per_sec: float
    decode_ms_per_step: float
    ram_gb: float
    vram_gb: float
    logits_max_diff: float
    correct: bool
    error: Optional[str] = None


def get_memory_gb():
    """Get current RAM/VRAM usage in GB."""
    import subprocess
    # RAM
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                ram_mb = int(line.split()[1]) / 1024
                break
    # VRAM via nvidia-smi or radeontop - but we're on AMD
    # Use /sys/class/drm/card0/device/mem_info_vram_used if available
    vram_gb = 0.0
    try:
        with open("/sys/class/drm/card0/device/mem_info_vram_used") as f:
            vram_gb = int(f.read().strip()) / (1024 ** 3)
    except Exception:
        pass
    return ram_mb, vram_gb


def measure_prefill(model, tokenizer, seq_len: int, reps: int = 3, use_q4: bool = False) -> float:
    """Measure prefill latency."""
    ids = torch.randint(0, 1000, (1, seq_len))
    # Warmup
    if use_q4:
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, use_q4=True)
    else:
        with torch.no_grad():
            model(input_ids=ids, use_cache=True)
    # Timed runs
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        if use_q4:
            with torch.no_grad():
                model(input_ids=ids, use_cache=True, use_q4=True)
        else:
            with torch.no_grad():
                model(input_ids=ids, use_cache=True)
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2]


def measure_decode(model, tokenizer, prefill_len: int, steps: int, reps: int = 1, use_q4: bool = False) -> tuple[float, float]:
    """
    Measure sustained decode throughput.
    Returns (steps_per_sec, ms_per_step).
    """
    # Prefill
    ids = torch.randint(0, 1000, (1, prefill_len))
    if use_q4:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, use_q4=True)
    else:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
    
    # Now decode `steps` tokens one at a time
    past_kv = out.past_key_values
    next_token = torch.randint(0, 1000, (1, 1))
    
    # Warmup step
    if use_q4:
        with torch.no_grad():
            model(input_ids=next_token, use_cache=True, past_key_values=past_kv, use_q4=True)
    else:
        with torch.no_grad():
            model(input_ids=next_token, use_cache=True, past_key_values=past_kv)
    
    # Timed decode
    times = []
    for _ in range(steps):
        t = time.perf_counter()
        if use_q4:
            with torch.no_grad():
                out = model(input_ids=next_token, use_cache=True, past_key_values=past_kv, use_q4=True)
        else:
            with torch.no_grad():
                out = model(input_ids=next_token, use_cache=True, past_key_values=past_kv)
        past_kv = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1:], dim=-1)
        times.append(time.perf_counter() - t)
    
    times.sort()
    median_step = times[len(times) // 2]
    return 1.0 / median_step, median_step * 1000


def run_profile(
    checkpoint_name: str,
    checkpoint_dir: str,
    gpu_layers: int,
    use_q4: bool,
    prefill_len: int,
    decode_steps: int,
    cpu_only: bool = False,
) -> ProfileResult:
    """Run a full profile for one configuration."""
    print(f"  Profiling: {checkpoint_name}, gpu_layers={gpu_layers}, q4={use_q4}, S={prefill_len}, steps={decode_steps}")
    
    try:
        import torch_vulkan_ext as tvk
        from hybrid_agent import load_hybrid_agent
        
        torch.manual_seed(0)
        model, tok = load_hybrid_agent(checkpoint_dir, dtype=torch.bfloat16, device="cpu")
        qm = model.backbone.model
        lm_head = model.backbone.lm_head
        
        # Build backbone
        if cpu_only or gpu_layers == 0:
            backbone = model.backbone
            placement = Placement.first_n(0, len(qm.layers))
        else:
            placement = Placement.first_n(gpu_layers, len(qm.layers))
            backbone = HybridBackbone(qm, placement)
        
        # Memory after load
        ram_after, vram_after = get_memory_gb()
        
        # Prefill
        prefill_time = measure_prefill(backbone, tok, prefill_len, use_q4=(use_q4 and not cpu_only and gpu_layers > 0))
        
        # Decode
        steps_per_sec, ms_per_step = measure_decode(backbone, tok, prefill_len, decode_steps, use_q4=(use_q4 and not cpu_only and gpu_layers > 0))
        
        # Correctness: compare logits at end of prefill
        ids = torch.randint(0, 1000, (1, prefill_len))
        with torch.no_grad():
            cpu_out = model.backbone(input_ids=ids, use_cache=True)
            if cpu_only or gpu_layers == 0:
                gpu_out = backbone(input_ids=ids, use_cache=True)
            else:
                gpu_out = backbone(input_ids=ids, use_cache=True, use_q4=use_q4)
        
        cpu_logits = cpu_out.logits
        gpu_logits = lm_head(gpu_out.last_hidden_state)
        max_diff = float((cpu_logits - gpu_logits).abs().max().item())
        correct = max_diff < 5.0  # tolerance for accumulated Vulkan+Q4 error
        
        return ProfileResult(
            checkpoint=checkpoint_name,
            seq_len=prefill_len,
            gpu_layers=gpu_layers,
            use_q4=use_q4,
            prefill_seconds=prefill_time,
            decode_steps_per_sec=steps_per_sec,
            decode_ms_per_step=ms_per_step,
            ram_gb=ram_after,
            vram_gb=vram_after,
            logits_max_diff=max_diff,
            correct=correct,
        )
    
    except Exception as e:
        return ProfileResult(
            checkpoint=checkpoint_name,
            seq_len=prefill_len,
            gpu_layers=gpu_layers,
            use_q4=use_q4,
            prefill_seconds=0,
            decode_steps_per_sec=0,
            decode_ms_per_step=0,
            ram_gb=0,
            vram_gb=0,
            logits_max_diff=0,
            correct=False,
            error=str(e),
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="8B", help="Comma-separated: 8B,14B")
    ap.add_argument("--gpu-layers", default="0,2,4,6,8,12", help="Comma-separated GPU layer counts")
    ap.add_argument("--prefill-lens", default="512,2048", help="Comma-separated prefill lengths")
    ap.add_argument("--decode-steps", type=int, default=100, help="Latent decode steps to measure")
    ap.add_argument("--use-q4", action="store_true", help="Test Q4 quantization")
    ap.add_argument("--out", default="reports/full_profile.json")
    args = ap.parse_args()
    
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]
    gpu_layers = [int(x) for x in args.gpu_layers.split(",")]
    prefill_lens = [int(x) for x in args.prefill_lens.split(",")]
    
    checkpoint_dirs = {
        "8B": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic",
        "14B": "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-14B-Agentic",
    }
    
    # Initialize Vulkan once
    import torch_vulkan_ext as tvk
    tvk.init(0)
    
    results = []
    
    for ckpt in checkpoints:
        if ckpt not in checkpoint_dirs:
            print(f"Unknown checkpoint: {ckpt}")
            continue
        ckpt_dir = checkpoint_dirs[ckpt]
        if not os.path.exists(ckpt_dir):
            print(f"Checkpoint not found: {ckpt_dir}")
            continue
        
        print(f"\n=== Checkpoint: {ckpt} ===")
        
        # CPU baseline
        for S in prefill_lens:
            res = run_profile(ckpt, ckpt_dir, 0, False, S, args.decode_steps, cpu_only=True)
            results.append(asdict(res))
            print(f"  CPU baseline S={S}: prefill={res.prefill_seconds:.2f}s, "
                  f"decode={res.decode_steps_per_sec:.2f} steps/s, correct={res.correct}")
        
        # GPU configs
        for ng in gpu_layers:
            if ng == 0:
                continue
            for use_q4 in [False, args.use_q4]:
                for S in prefill_lens:
                    res = run_profile(ckpt, ckpt_dir, ng, use_q4, S, args.decode_steps)
                    results.append(asdict(res))
                    status = "OK" if res.correct else "DIVERGENT"
                    err = f" ({res.error})" if res.error else ""
                    print(f"  GPU={ng} Q4={use_q4} S={S}: prefill={res.prefill_seconds:.2f}s, "
                          f"decode={res.decode_steps_per_sec:.2f} steps/s, {status}{err}")
                    
                    # If context loss, don't try more layers
                    if res.error and "context" in res.error.lower():
                        break
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()