#!/usr/bin/env python3
"""Create HybridBackbone with INT8 CPU layers - WORKING VERSION."""
import torch
import gc
import os
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear

# Paths
MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

def create_hybrid_backbone_with_int8(gpu_layers: int = 0):
    """
    Create HybridBackbone with INT8 CPU layers and Vulkan GPU layers.
    1. Load original BF16 model
    2. Create HybridBackbone with placement
    3. Quantize CPU layers with INT8 from safetensors
    """
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc

    tvk.init(0)
    torch.manual_seed(0)
    
    # Step 1: Load original BF16 model
    print('Loading original BF16 model...')
    model, tok = load_hybrid_agent(
        '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
        dtype=torch.bfloat16, device='cpu'
    )
    print('Model loaded')
    
    # Step 2: Create HybridBackbone with original BF16 weights (Vulkan layers get correct weights)
    qm = model.backbone.model  # Qwen3Model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    print(f'HybridBackbone created with {gpu_layers} GPU layers')
    
    # Step 3: Quantize ONLY CPU layers (non-Vulkan) with INT8 from safetensors
    print('Quantizing CPU layers with INT8...')
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

            # Layer norms (already bfloat16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f'model.layers.{i}.input_layernorm.weight')
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except: pass
            try:
                ln2_weight = f.get_tensor(f'model.layers.{i}.post_attention_layernorm.weight')
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except: pass
            # QKV norms
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
    print('INT8 quantization done')
    return hb


def run_sweep_config(seq_len: int, gpu_layers: int, reps: int = 2) -> float:
    """Run a single sweep configuration and return latency in ms."""
    import time
    import gc
    import torch
    
    hb = create_hybrid_backbone_with_int8(gpu_layers)
    gc.collect()
    
    ids = torch.randint(0, 1000, (1, seq_len))
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)  # warmup
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                hb(input_ids=ids, use_cache=True)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    if not args.driver:
        import gc
        secs = run_sweep_config(args.seq, args.layers, args.reps)
        print(f"RESULT {secs:.2f}")
        sys.exit(0)
    
    # driver mode
    seq = args.seq
    counts = [0, 1, 2, 3, 4, 5, 6]
    results = []
    print(f"=== INT8+Vulkan sweep at S={seq} ===", flush=True)
    for g in counts:
        try:
            ms = run_sweep_config(seq, g, args.reps)
            print(f"  gpu_layers={g}: {ms:.2f} ms", flush=True)
            results.append({"gpu_layers": g, "ms": ms, "error": None})
        except Exception as e:
            print(f"  gpu_layers={g}: FAILED - {e}", flush=True)
            results.append({"gpu_layers": g, "ms": None, "error": str(e)})
        
        gc.collect()
        time.sleep(2)
    
    ok = [r for r in results if r["ms"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["ms"])
        cpu = next((r["ms"] for r in results if r["gpu_layers"] == 0), None)
        print(f"\n  best: {best['gpu_layers']} GPU layers -> {best['ms']:.2f} ms")
        if cpu:
            print(f"  vs all-CPU {cpu:.2f} ms -> {cpu/best['ms']:.3f}x {'GPU WINS' if best['ms'] < cpu else 'CPU WINS'}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"seq_len": seq, "results": results}, f, indent=2)
        print(f"wrote {args.out}")