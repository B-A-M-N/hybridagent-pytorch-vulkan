#!/usr/bin/env python3
"""Production INT8 model loader - CPU layers INT8, Vulkan layers BF16, NO BF16 intermediate."""
import torch
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear
import os
import gc

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"


def create_hybrid_backbone_with_int8(gpu_layers: int = 0):
    """
    Create HybridBackbone with INT8 CPU layers and Vulkan GPU layers.
    CRITICAL: Load INT8 weights DIRECTLY - NO BF16 intermediate.
    CPU layers: Int8Linear | Vulkan layers: standard BF16 Linear (loaded from original model)
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

    tvk.init(0)
    torch.manual_seed(0)

    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

    print("=== Building INT8 model directly (NO BF16 intermediate) ===")

    # Step 1: Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)

    # Determine which layers go to GPU
    placement = Placement.first_n(gpu_layers, config.num_hidden_layers)
    gpu_layer_indices = set()
    for i in range(config.num_hidden_layers):
        if placement.device_of(i) == "vulkan":
            gpu_layer_indices.add(i)

    # Step 2: Create model architecture on meta device (NO weights)
    print("Creating model architecture on meta device...")
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    # CRITICAL: Replace CPU Linear layers with Int8Linear WHILE STILL ON META DEVICE
    print("Replacing CPU Linear layers with Int8Linear on meta device...")
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Linear):
            # Skip lm_head - keep as bfloat16 Linear
            if name == 'lm_head' or name.endswith('.lm_head'):
                continue
            
            # Check if this is a Vulkan layer - keep as BF16
            is_vulkan_layer = False
            if 'model.layers.' in name:
                try:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    if layer_idx in gpu_layer_indices:
                        is_vulkan_layer = True
                except:
                    pass
            
            if is_vulkan_layer:
                continue  # Keep as standard BF16 Linear for Vulkan
            
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            dtype = module.weight.dtype
            quantized = Int8Linear(in_features, out_features, bias=bias, dtype=dtype)
            setattr(parent, attr_name, quantized)

    print(f'CPU Linear layers replaced with Int8Linear, {len(gpu_layer_indices)} Vulkan layers kept as BF16')

    # Step 3: Build COMPLETE state_dict from INT8 checkpoint (CPU) + original (Vulkan)
    print('Building COMPLETE state_dict...')
    state_dict = {}

    # First load Vulkan layer weights from original model
    if gpu_layer_indices:
        print(f'Loading BF16 weights for {len(gpu_layer_indices)} Vulkan layers from original...')
        with safe_open(ORIGINAL_SAFETENSORS_PATH, framework="pt", device="cpu") as f:
            for i in gpu_layer_indices:
                # MLP
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    base = f"model.layers.{i}.mlp.{name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        state_dict[f"model.layers.{i}.mlp.{name}.weight"] = weight
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
                    except: pass

                # Attention
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    base = f"model.layers.{i}.self_attn.{proj_name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight"] = weight
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
                    except: pass

    # Then load CPU INT8 weights + all norms from INT8 checkpoint
    print('Loading INT8 weights for CPU layers + norms from INT8 checkpoint...')
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Layer weights
        for i in range(config.num_hidden_layers):
            is_vulkan = i in gpu_layer_indices
            
            if not is_vulkan:
                # MLP - INT8
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    base = f"model.layers.{i}.mlp.{name}"
                    weight_int8 = f.get_tensor(f"{base}.weight_int8")
                    weight_scales = f.get_tensor(f"{base}.weight_scales")
                    state_dict[f"model.layers.{i}.mlp.{name}.weight_int8"] = weight_int8
                    state_dict[f"model.layers.{i}.mlp.{name}.scales"] = weight_scales
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
                    except:
                        pass

                # Attention - INT8
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    base = f"model.layers.{i}.self_attn.{proj_name}"
                    weight_int8 = f.get_tensor(f"{base}.weight_int8")
                    weight_scales = f.get_tensor(f"{base}.weight_scales")
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
                    except:
                        pass

            # Layer norms (bf16 in safetensors) - load for all layers
            try:
                ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                state_dict[f"model.layers.{i}.input_layernorm.weight"] = ln1
            except: pass
            try:
                ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = ln2
            except: pass

            # QKV norms
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                state_dict[f"model.layers.{i}.self_attn.q_norm.weight"] = q_norm
            except: pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                state_dict[f"model.layers.{i}.self_attn.k_norm.weight"] = k_norm
            except: pass

        # Embeddings, final norm, lm_head
        embed = f.get_tensor("model.embed_tokens.weight")
        state_dict["model.embed_tokens.weight"] = embed

        norm_weight = f.get_tensor("model.norm.weight")
        state_dict["model.norm.weight"] = norm_weight

        lm_head = f.get_tensor("lm_head.weight")
        state_dict["lm_head.weight"] = lm_head

    print(f'State dict built with {len(state_dict)} tensors (Vulkan BF16 + CPU INT8 + norms)')

    # Step 4: Load COMPLETE state_dict directly with assign=True - materializes with loaded data!
    print('Loading state_dict with assign=True...')
    backbone.load_state_dict(state_dict, assign=True)

    # Step 5: Materialize remaining meta tensors (rotary embeddings, etc.)
    print('Materializing remaining meta tensors...')
    backbone = backbone.to_empty(device="cpu")

    # Step 6: Wrap in HybridAgentModel
    from hybrid_agent import HybridAgentModel, HybridLink
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    # Load HybridLink
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    print('HybridLink loaded')

    # Step 7: Create HybridBackbone with proper placement
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    qm = model.backbone.model
    hb = HybridBackbone(qm, placement)
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