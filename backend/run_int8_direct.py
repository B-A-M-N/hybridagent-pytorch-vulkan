#!/usr/bin/env python3
"""Direct INT8 model loader - loads quantized model WITHOUT BF16 intermediate."""
import os
import sys
import torch
import gc
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

def load_int8_model_direct(gpu_layers: int = 0):
    """
    Load INT8 model directly from checkpoint WITHOUT BF16 intermediate.
    1. Create model architecture on meta device (NO weights)
    2. Replace parameters with empty int8 tensors on meta device
    3. Materialize on CPU (creates int8 tensors directly)
    4. Load INT8 weights directly into int8 tensors
    4. Create HybridBackbone with proper placement
    """
    import torch
    import torch_vulkan_ext as tvk
    from transformers import AutoTokenizer, AutoConfig
    from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
    from transformers import AutoModelForCausalLM
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc

    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

    tvk.init(0)
    torch.manual_seed(0)

    # Step 1: Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)

    # Step 2: Create model architecture on meta device (NO weights)
    print("Creating model architecture on meta device...")
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    # CRITICAL: Replace ALL parameter data with EMPTY int8 tensors on meta device
    print("Replacing meta tensors with empty int8 tensors on meta device...")
    for name, param in backbone.named_parameters():
        # Create empty int8 tensor on meta device with correct shape
        param.data = torch.empty(param.shape, dtype=torch.int8, device='meta')

    print('All parameters replaced with empty int8 tensors on meta device')

    # Materialize on CPU - this will create int8 tensors directly!
    backbone = backbone.to_empty(device="cpu")
    print('Backbone materialized on CPU with int8 tensors')

    # Verify tensors are int8
    for name, param in backbone.named_parameters():
        if param.dtype != torch.int8:
            print(f"WARNING: {name} is {param.dtype}, expected int8")

    print('Backbone created with int8 tensors - NO BF16 ever allocated')

    # Step 2: Wrap in HybridAgentModel
    from hybrid_agent import HybridAgentModel, HybridLink
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    # Step 3: Load HybridLink (lightweight)
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    print('HybridLink loaded')

    # Step 3: Load INT8 weights directly into the int8 tensors
    print('Loading INT8 weights directly into int8 tensors...')
    from int8_linear_v3 import Int8Linear

    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i in range(36):
            layer = model.backbone.model.layers[i]

            # MLP
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), Int8Linear):
                        base = f"model.layers.{i}.mlp.{name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass

                        orig = getattr(mlp, name)
                        orig.weight_int8.data.copy_(weight_int8)
                        orig.scales.data.copy_(weight_scales)
                        if bias is not None:
                            orig.bias.data.copy_(bias)

            # Attention
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), Int8Linear):
                        base = f"model.layers.{i}.self_attn.{proj_name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass

                        orig = getattr(attn, proj_name)
                        orig.weight_int8.data.copy_(weight_int8)
                        orig.scales.data.copy_(weight_scales)
                        if bias is not None:
                            orig.bias.data.copy_(bias)

            # Layer norms (bf16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except: pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except: pass

            # QKV norms
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                if hasattr(layer.self_attn, 'q_norm'):
                    layer.self_attn.q_norm.weight.data = q_norm.to(layer.self_attn.q_norm.weight.dtype)
            except: pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                if hasattr(layer.self_attn, 'k_norm'):
                    layer.self_attn.k_norm.weight.data = k_norm.to(layer.self_attn.k_norm.weight.dtype)
            except: pass

        # Embeddings
        embed = f.get_tensor("model.embed_tokens.weight")
        model.backbone.model.embed_tokens.weight.data = embed.to(model.backbone.model.embed_tokens.weight.dtype)

        # Final norm
        norm_weight = f.get_tensor("model.norm.weight")
        model.backbone.model.norm.weight.data = norm_weight.to(model.backbone.model.norm.weight.dtype)

        # LM head (bf16 in safetensors)
        lm_head = f.get_tensor("lm_head.weight")
        model.backbone.lm_head.weight.data = lm_head.to(model.backbone.lm_head.weight.dtype)

    print('INT8 weights loaded')

    # Load HybridLink
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    print('HybridLink loaded')

    # Create HybridBackbone with proper placement
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    gc.collect()
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
    import json
    import os

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
    for g in [0, 1, 2, 3, 4, 5, 6]:
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