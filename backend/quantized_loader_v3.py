"""Direct INT8 checkpoint loader — loads quantized model without BF16 intermediate."""
import os
import torch
import gc
from typing import Optional
from safetensors import safe_open
from safetensors.torch import save_file

from backend.model_access import get_qwen3_model, get_qwen3_layers, create_hybrid_backbone
from int8_linear_v3 import Int8Linear
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
from hybrid_agent import load_hybrid_agent


def create_hybrid_backbone_with_int8(gpu_layers: int = 0):
    """
    Create HybridBackbone with INT8 CPU layers and Vulkan GPU layers.
    Loads INT8 weights DIRECTLY from checkpoint without BF16 intermediate.
    Single entry point for sweep runner.
    """
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    INT8_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    TOTAL_LAYERS = 36
    
    tvk.init(0)
    torch.manual_seed(0)
    
    # Load ONLY tokenizer and HybridLink from original model (lightweight)
    from transformers import AutoTokenizer, AutoConfig
    from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
    from transformers import AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = AutoConfig.from_pretrained(MODEL_DIR)
    
    # Build model architecture on meta device (NO weights allocated)
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    
    # Materialize on target device with empty tensors
    backbone = backbone.to_empty(device="cpu")
    
    # Create HybridAgentModel wrapper
    from hybrid_agent import HybridAgentModel, HybridLink
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)
    
    # Load HybridLink adapter (lightweight)
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    
    # NOW load INT8 weights directly into the meta-initialized model
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Load all layer weights (INT8 + scales + norms) directly into meta tensors
        for i, layer in enumerate(model.backbone.model.layers):
            # Load layer norm weights (bf16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass
            # QKV norms
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                if hasattr(layer.self_attn, 'q_norm'):
                    layer.self_attn.q_norm.weight.data = q_norm.to(layer.self_attn.q_norm.weight.dtype)
            except:
                pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                if hasattr(layer.self_attn, 'k_norm'):
                    layer.self_attn.k_norm.weight.data = k_norm.to(layer.self_attn.k_norm.weight.dtype)
            except:
                pass
        
        # Embeddings
        try:
            embed = f.get_tensor("model.embed_tokens.weight")
            model.backbone.model.embed_tokens.weight.data = embed.to(model.backbone.model.embed_tokens.weight.dtype)
        except:
            pass
        # Final norm
        try:
            norm_weight = f.get_tensor("model.norm.weight")
            model.backbone.model.norm.weight.data = norm_weight.to(model.backbone.model.norm.weight.dtype)
        except:
            pass
        # LM head
        try:
            lm_head = f.get_tensor("lm_head.weight")
            model.backbone.lm_head.weight.data = lm_head.to(model.backbone.lm_head.weight.dtype)
        except:
            pass
    
    gc.collect()
    
    # Create HybridBackbone with INT8 model (all layers are INT8 at this point)
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(0, 36))  # Start with all CPU
    
    # NOW set up Vulkan layers and quantize CPU layers
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                # Vulkan layers need BF16 weights - they'll upload from the INT8 model's tensors
                # The VulkanDecoderLayer constructor will handle uploading
                continue
            
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                        base = f"model.layers.{i}.mlp.{name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(mlp, name)
                        quantized = Int8Linear(
                            orig.in_features, orig.out_features, 
                            bias=orig.bias is not None, dtype=orig.weight.dtype
                        )
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(mlp, name, quantized)
            
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f"model.layers.{i}.self_attn.{proj_name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(
                            orig.in_features, orig.out_features, 
                            bias=orig.bias is not None, dtype=orig.weight.dtype
                        )
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)
            
            # Layer norms (already bfloat16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass
    
    gc.collect()
    
    # For GPU layers, create proper HybridBackbone with Vulkan layers
    if gpu_layers > 0:
        # Create HybridBackbone with original model (which has INT8 weights)
        # Then replace Vulkan layers with VulkanDecoderLayer that will use the INT8 weights
        qm = model.backbone.model
        hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
        
        # Re-quantize CPU layers (non-Vulkan) - they're already INT8 in model
        # But we need to ensure the HybridBackbone's CPU layers are also INT8
        # The HybridBackbone constructor copies the original model's layers
        # So we need to re-quantize the CPU layers in the new HybridBackbone
        with safe_open("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors", framework="pt", device="cpu") as f:
            for i, layer in enumerate(hb.layers):
                if isinstance(layer, VulkanDecoderLayer):
                    continue
                
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    for name in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                            base = f"model.layers.{i}.mlp.{name}"
                            weight_int8 = f.get_tensor(f"{base}.weight_int8")
                            weight_scales = f.get_tensor(f"{base}.weight_scales")
                            bias = None
                            try:
                                bias = f.get_tensor(f"{base}.bias")
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
                            base = f"model.layers.{i}.self_attn.{proj_name}"
                            weight_int8 = f.get_tensor(f"{base}.weight_int8")
                            weight_scales = f.get_tensor(f"{base}.weight_scales")
                            bias = None
                            try:
                                bias = f.get_tensor(f"{base}.bias")
                            except:
                                pass
                            
                            orig = getattr(attn, proj_name)
                            quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                            quantized.weight_int8.data.copy_(weight_int8)
                            quantized.scales.data.copy_(weight_scales)
                            if bias is not None:
                                quantized.bias.data.copy_(bias)
                            setattr(attn, proj_name, quantized)
                
                # Layer norms
                try:
                    ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                    layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
                except:
                    pass
                try:
                    ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                    layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
                except:
                    pass
        
        gc.collect()
        return hb
    
    gc.collect()
    return hb


def run_sweep_config(seq_len: int, gpu_layers: int, reps: int = 2):
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
    return ts[len(ts) // 2]


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
        print(f"RESULT {secs * 1000:.2f}")  # ms
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
        
        # Clean up between runs
        gc.collect()
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    ok = [r for r in results if r["ms"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["ms"])
        cpu = next((r["ms"] for r in results if r["gpu_layers"] == 0), None)
        print(f"\n  best: {best['gpu_layers']} GPU layers -> {best['ms']:.2f} ms")
        if cpu:
            print(f"  vs all-CPU {cpu:.2f} ms -> {cpu/best['ms']:.3f}x {'GPU WINS' if best['ms'] < cpu else 'CPU WINS'}")