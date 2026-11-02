#!/usr/bin/env python3
"""Single subprocess INT8+Vulkan sweep - runs each GPU config in fresh process."""
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
from backend.quantized_loader_final import create_hybrid_backbone_with_int8


def run_one_config(seq_len: int, gpu_layers: int, reps: int = 2) -> tuple[float | None, str | None]:
    """Run a single sweep config in THIS process (called by subprocess)."""
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    import time

    tvk.init(0)
    torch.manual_seed(0)
    
    # Load original BF16 model
    model, tok = load_hybrid_agent(
        '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
        dtype=torch.bfloat16, device='cpu'
    )
    
    # Create HybridBackbone with BF16 weights (Vulkan layers get correct weights)
    qm = model.backbone.model  # Qwen3Model - has .layers
    hb = HybridBackbone(model.backbone.model, Placement.first_n(gpu_layers, 36))
    
    # CRITICAL: Delete original model to free memory
    del model
    gc.collect()
    
    # Quantize ONLY CPU layers with INT8 from safetensors
    SAFETENSORS_PATH = '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors'
    
    with safe_open(SAFETENSORS_PATH, framework='pt', device='cpu') as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                        base = f'model.layers.{i}.mlp.{name}'
                        weight_int8 = f.get_tensor(f'{base}.weight_int8')
                        weight_scales = f.get_tensor(f'{base}.weight_scales')
                        bias = None
                        try:
                            bias = f.get_tensor(f'{base}.bias')
                        except:
                            pass
                        orig = getattr(mlp, name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(mlp, name, quantized)
                        del orig
                        gc.collect()
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f'model.layers.{i}.self_attn.{proj_name}'
                        weight_int8 = f.get_tensor(f'{base}.weight_int8')
                        weight_scales = f.get_tensor(f'{base}.weight_scales')
                        bias = None
                        try:
                            bias = f.get_tensor(f'{base}.bias')
                        except:
                            pass
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)
                        del orig
                        gc.collect()
            # Layer norms
            try:
                ln1_weight = f.get_tensor(f'model.layers.{i}.input_layernorm.weight')
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except: pass
            try:
                ln2_weight = f.get_tensor(f'model.layers.{i}.post_attention_layernorm.weight')
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except: pass
            try:
                q_norm = f.get_tensor(f'model.layers.{i}.self_attn.q_norm.weight')
                if hasattr(attn, 'q_norm'):
                    attn.q_norm.weight.data = q_norm.to(attn.q_norm.weight.dtype)
            except: pass
            try:
                k_norm = f.get_tensor(f'model.layers.{i}.self_attn.k_norm.weight')
                if hasattr(attn, 'k_norm'):
                    attn.k_norm.weight.data = k_norm.to(attn.k_norm.weight.dtype)
            except: pass
    
    gc.collect()
    
    # Run inference
    ids = torch.randint(0, 1000, (1, seq_len))
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            with torch.no_grad():
                hb(input_ids=torch.randint(0, 1000, (1, seq_len)), use_cache=True)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def run_subprocess(seq_len: int, gpu_layers: int, reps: int = 2, timeout: int = 1800) -> tuple[float | None, str | None]:
    """Run a single configuration in a FRESH subprocess."""
    script = f"""
import sys
sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')
sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent')
sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext')

import torch
import torch_vulkan_ext as tvk
from hybrid_agent import load_hybrid_agent
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import gc
import time
import sys

tvk.init(0)
torch.manual_seed(0)

model, tok = load_hybrid_agent('/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic', dtype=torch.bfloat16, device='cpu')
qm = model.backbone.model  # Qwen3Model
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
hb = HybridBackbone(qm, Placement.first_n({gpu_layers}, 36))
del model

SAFETENSORS_PATH = '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors'
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
from backend.hybrid_offload import VulkanDecoderLayer
import gc
import torch

with safe_open('/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors', framework='pt', device='cpu') as f:
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            continue
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                    base = f'model.layers.{{i}}.{name}'
                    weight_int8 = f.get_tensor(f'{{base}}.weight_int8')
                    weight_scales = f.get_tensor(f'{{base}}.weight_scales')
                    bias = None
                    try:
                        bias = f.get_tensor(f'{{base}}.bias')
                    except:
                        pass
                    orig = getattr(mlp, name)
                    from int8_linear_v3 import Int8Linear
                    quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                    quantized.weight_int8.data.copy_(weight_int8)
                    quantized.scales.data.copy_(weight_scales)
                    if bias is not None:
                        quantized.bias.data.copy_(bias)
                    setattr(mlp, name, quantized)
                    del orig
                    gc.collect()

        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                    base = f'model.layers.{{i}}.self_attn.{{proj_name}}'
                    weight_int8 = f.get_tensor(f'{{base}}.weight_int8')
                    weight_scales = f.get_tensor(f'{{base}}.weight_scales')
                    bias = None
                    try:
                        bias = f.get_tensor(f'{{base}}.bias')
                    except:
                        pass
                    orig = getattr(attn, proj_name)
                    from int8_linear_v3 import Int8Linear
                    quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                    quantized.weight_int8.data.copy_(weight_int8)
                    quantized.scales.data.copy_(weight_scales)
                    if bias is not None:
                        quantized.bias.data.copy_(bias)
                    setattr(attn, proj_name, quantized)
                    del orig
                    gc.collect()

            try:
                ln1_weight = f.get_tensor(f'model.layers.{{i}}.input_layernorm.weight')
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except: pass
            try:
                ln2_weight = f.get_tensor(f'model.layers.{{i}}.post_attention_layernorm.weight')
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except: pass
            try:
                q_norm = f.get_tensor(f'model.layers.{{i}}.self_attn.q_norm.weight')
                if hasattr(attn, 'q_norm'):
                    attn.q_norm.weight.data = q_norm.to(attn.q_norm.weight.dtype)
            except: pass
            try:
                k_norm = f.get_tensor(f'model.layers.{{i}}.self_attn.k_norm.weight')
                if hasattr(attn, 'k_norm'):
                    attn.k_norm.weight.data = k_norm.to(attn.k_norm.weight.dtype)
            except: pass

gc.collect()
ids = torch.randint(0, 1000, (1, {seq_len}))
with torch.no_grad():
    hb(input_ids=ids, use_cache=True)
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        with torch.no_grad():
            hb(input_ids=torch.randint(0, 1000, (1, {seq_len})), use_cache=True)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f'RESULT {{ts[len(ts) // 2] * 1000:.2f}}')
"""
    
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{ROOT}:{ROOT}/third_party/HybridAgent:{ROOT}/backend/torch_vulkan_ext",
        "OMP_NUM_THREADS": "12",
    })
    
    cmd = [sys.executable, "-c", script.format(seq_len=seq_len, gpu_layers=gpu_layers)]
    
    try:
        proc = subprocess.run(
            cmd,
            env={**os.environ, **{"PYTHONPATH": f"{ROOT}:{ROOT}/third_party/HybridAgent:{ROOT}/backend/torch_vulkan_ext", "OMP_NUM_THREADS": "12"}},
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=ROOT,
        )
        
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                ms = float(line.split()[1])
                return ms, None
        
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
    
    layer_counts = [int(x) for x in args.layers.split(",")]
    
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
        est_mem = estimate_config_memory(gpu_layers)
        can_run, status = check_memory_budget(f"S={args.seq} GPU={gpu_layers}", estimate_config_memory(gpu_layers))
        print(f"\n  GPU={gpu_layers}: est {estimate_config_memory(gpu_layers):.1f}GB - {status}")
        
        if not can_run:
            results.append({"gpu_layers": gpu_layers, "ms": None, "error": "INSUFFICIENT_MEMORY", "status": "SKIPPED"})
            continue
        
        print(f"  Running GPU={gpu_layers}...", flush=True)
        ms, error = run_subprocess(args.seq, gpu_layers, args.reps)
        
        if error:
            results.append({"gpu_layers": gpu_layers, "ms": None, "error": error})
            print(f"  FAILED: {error}")
        else:
            results.append({"gpu_layers": gpu_layers, "ms": ms, "error": None})
            print(f"  GPU={gpu_layers}: {ms:.2f} ms")
        
        gc.collect()
        time.sleep(2)
    
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
    
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"seq_len": args.seq, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    import json
    import gc
    import time
    import torch
    
    from backend.memory_budget import check_memory_budget, estimate_config_memory, get_memory_budget
    
    main()