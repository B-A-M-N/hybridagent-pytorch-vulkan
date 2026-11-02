#!/usr/bin/env python3
"""Test 100+ sustained latent steps with full HybridAgentModel wrapper."""
import torch
import torch.nn as nn
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, DeviceStats
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import os
import gc
import time

PROFILE = os.environ.get("PROFILE_LATENT") == "1"
# Pin CPU thread count identically for both baselines so the CPU-INT8 columns are
# comparable. Ryzen 9 5900X = 12 physical cores / 24 SMT; use physical cores and
# leave SMT + headroom for the shared desktop.
_CPU_THREADS = int(os.environ.get("HYBRID_CPU_THREADS", "12"))
torch.set_num_threads(_CPU_THREADS)
print("=== 100+ Step Heterogeneous Latent Generation Test (Full Model) ===")
print(f"  CPU threads (torch): {_CPU_THREADS}")
if PROFILE:
    print("  PROFILE_LATENT=1: detailed step profiling enabled")

gpu_layers = int(os.environ.get("HYBRID_GPU_LAYERS", "5"))

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
add_anchor_tokens(tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(MODEL_DIR)

tvk.init(0)
torch.manual_seed(0)

# Ensure GPU KV cache is enabled
os.environ["DISABLE_GPU_KV"] = "0"

placement = Placement.first_n(gpu_layers, config.num_hidden_layers)
gpu_layer_indices = set()
for i in range(config.num_hidden_layers):
    if placement.device_of(i) == "vulkan":
        gpu_layer_indices.add(i)

# Create model on meta device
with torch.device("meta"):
    backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

# Replace CPU Linear layers with Int8Linear
for name, module in list(backbone.named_modules()):
    if isinstance(module, nn.Linear):
        if name == 'lm_head' or name.endswith('.lm_head'):
            continue
        is_vulkan_layer = False
        if 'model.layers.' in name:
            try:
                layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                if layer_idx in gpu_layer_indices:
                    is_vulkan_layer = True
            except:
                pass
        if is_vulkan_layer:
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        dtype = module.weight.dtype
        quantized = Int8Linear(in_features, out_features, bias=bias, dtype=dtype)
        setattr(parent, attr_name, quantized)

# Build state_dict from INT8 checkpoint (CPU) + original (Vulkan)
state_dict = {}

if gpu_layer_indices:
    with safe_open(ORIGINAL_SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i in gpu_layer_indices:
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

with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    for i in range(config.num_hidden_layers):
        is_vulkan = i in gpu_layer_indices
        if not is_vulkan:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                base = f"model.layers.{i}.mlp.{name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                state_dict[f"model.layers.{i}.mlp.{name}.weight_int8"] = weight_int8
                state_dict[f"model.layers.{i}.mlp.{name}.scales"] = weight_scales
                try:
                    bias = f.get_tensor(f"{base}.bias")
                    state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
                except: pass
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
                try:
                    bias = f.get_tensor(f"{base}.bias")
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
                except: pass
        try:
            ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
            state_dict[f"model.layers.{i}.input_layernorm.weight"] = ln1
        except: pass
        try:
            ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
            state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = ln2
        except: pass
        try:
            q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
            state_dict[f"model.layers.{i}.self_attn.q_norm.weight"] = q_norm
        except: pass
        try:
            k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
            state_dict[f"model.layers.{i}.self_attn.k_norm.weight"] = k_norm
        except: pass
    embed = f.get_tensor("model.embed_tokens.weight")
    state_dict["model.embed_tokens.weight"] = embed
    norm_weight = f.get_tensor("model.norm.weight")
    state_dict["model.norm.weight"] = norm_weight
    lm_head = f.get_tensor("lm_head.weight")
    state_dict["lm_head.weight"] = lm_head

backbone.load_state_dict(state_dict, assign=True)
backbone = backbone.to_empty(device="cpu")

# Load Vulkan layer weights
if gpu_layer_indices:
    with safe_open(ORIGINAL_SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i in gpu_layer_indices:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                base = f"model.layers.{i}.mlp.{name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    if hasattr(backbone.model.layers[i].mlp, name):
                        getattr(backbone.model.layers[i].mlp, name).weight.data.copy_(weight)
                except: pass
                try:
                    bias = f.get_tensor(f"{base}.bias")
                    if hasattr(backbone.model.layers[i].mlp, name) and getattr(backbone.model.layers[i].mlp, name).bias is not None:
                        getattr(backbone.model.layers[i].mlp, name).bias.data.copy_(bias)
                except: pass
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    if hasattr(backbone.model.layers[i].self_attn, proj_name):
                        getattr(backbone.model.layers[i].self_attn, proj_name).weight.data.copy_(weight)
                except: pass
                try:
                    bias = f.get_tensor(f"{base}.bias")
                    if hasattr(backbone.model.layers[i].self_attn, proj_name) and getattr(backbone.model.layers[i].self_attn, proj_name).bias is not None:
                        getattr(backbone.model.layers[i].self_attn, proj_name).bias.data.copy_(bias)
                except: pass
            try:
                ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                backbone.model.layers[i].input_layernorm.weight.data.copy_(ln1)
            except: pass
            try:
                ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                backbone.model.layers[i].post_attention_layernorm.weight.data.copy_(ln2)
            except: pass
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                if hasattr(backbone.model.layers[i].self_attn, 'q_norm'):
                    backbone.model.layers[i].self_attn.q_norm.weight.data.copy_(q_norm)
            except: pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                if hasattr(backbone.model.layers[i].self_attn, 'k_norm'):
                    backbone.model.layers[i].self_attn.k_norm.weight.data.copy_(k_norm)
            except: pass

gc.collect()

# Create HybridAgentModel
model = HybridAgentModel(backbone)
model.resize_to_tokenizer(tokenizer)

# Load HybridLink
adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
if os.path.exists(adapter_path):
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    model.hybridlink.load_state_dict(state, strict=True)
print('HybridLink loaded')

# Create HybridBackbone with placement
qm = model.backbone.model
hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
gc.collect()

print("=== Full HybridAgentModel with Heterogeneous Backbone ===")

def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float("nan")

def _swapped_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSwap:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")

print(f"  RSS before upload: {_rss_mb():.0f} MB")

# Upload Vulkan weights
print("Uploading Vulkan weights...")
for i, layer in enumerate(hb.layers):
    if hasattr(layer, 'upload_weights'):
        layer.upload_weights()
gc.collect()

# Release the duplicated CPU-side FP32 parameters now that the VRAM copies exist.
# Leaves the packed/device buffers (self._w / self._w_q4_buf) intact.
if os.environ.get("HYBRID_KEEP_CPU_WEIGHTS") != "1":
    for layer in hb.layers:
        if hasattr(layer, "release_cpu_weights"):
            layer.release_cpu_weights()
    gc.collect()
    print(f"  RSS after upload + release_cpu_weights: {_rss_mb():.0f} MB "
          f"(swap {_swapped_mb():.0f} MB)")
else:
    print(f"  RSS after upload (weights retained): {_rss_mb():.0f} MB")

# Attach GPU KV cache manager to Vulkan layers (same as p2_gpu_kv_test.py)
if gpu_layers > 0:
    from p2_gpu_kv_test import GPUKVCacheManager, _Buf
    kv_manager = GPUKVCacheManager(config, set(range(gpu_layers)))
    for layer in hb.layers:
        if isinstance(layer, VulkanDecoderLayer):
            layer._gpu_kv_cache = kv_manager

# Now test 100 latent steps with full model
print("\n--- Testing 100 sequential latent steps ---")

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    # Prefill through full model
    out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_hidden = out.hidden_states[-1][:, -1:, :]
    next_logits = model.backbone.lm_head(last_hidden)[:, -1, :]
    
    # Seed GPU KV cache with prefill data from DynamicCache
    if gpu_layers > 0:
        for layer in hb.layers:
            if isinstance(layer, VulkanDecoderLayer) and hasattr(layer, '_gpu_kv_cache'):
                kv_mgr = layer._gpu_kv_cache
                cached_layer = past_kv.layers[layer.layer_idx]
                # Get K/V from DynamicCache: [1, n_kv_heads, seq_len, head_dim]
                cached_k = cached_layer.keys  # [1, n_kv_heads, seq_len, head_dim]
                cached_v = cached_layer.values
                # Convert to [1, n_kv_heads * seq_len * head_dim] format for GPU
                S = cached_k.shape[2]
                KVD = layer.num_kv_heads * layer.head_dim
                # Permute to [1, seq_len, n_kv_heads, head_dim] then flatten
                k_gpu = cached_k.permute(0, 2, 1, 3).contiguous().view(1, S * KVD)
                v_gpu = cached_v.permute(0, 2, 1, 3).contiguous().view(1, S * KVD)
                # Upload to GPU KV cache
                kv_mgr.append_kv(layer.layer_idx, _Buf(k_gpu), _Buf(v_gpu), S, KVD)
                print(f"  Seeded GPU KV cache layer {layer.layer_idx} with prefill data (S={S})")

    # latent steps (HYBRID_STEPS lets a smoke test run a short trajectory)
    _STEPS = int(os.environ.get("HYBRID_STEPS", "100"))
    latent_times = []
    profile_totals = {
        "hybridlink_ms": [],
        "backbone_ms": [],
        "step_total_ms": [],
        "cpu_layer_ms": [],
        "vk_layer_ms": [],
        "gpu_layer_count": [],
        "cpu_layer_count": [],
    }
    for step in range(_STEPS):
        # In LATENT mode, run HybridLink on last_hidden
        hlink_t0 = time.perf_counter()
        injected = model.hybridlink(last_hidden)
        hlink_t1 = time.perf_counter()

        stats = DeviceStats() if PROFILE else None
        backbone_t0 = time.perf_counter()
        out = hb(
            inputs_embeds=injected,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            use_q4=False,
            stats=stats,
        )
        backbone_t1 = time.perf_counter()

        step_ms = (backbone_t1 - hlink_t0) * 1000
        latent_times.append(step_ms)
        if PROFILE:
            profile_totals["hybridlink_ms"].append((hlink_t1 - hlink_t0) * 1000)
            profile_totals["backbone_ms"].append((backbone_t1 - backbone_t0) * 1000)
            profile_totals["step_total_ms"].append(step_ms)
            if stats is not None:
                summary = stats.summary()
                profile_totals["cpu_layer_ms"].append(summary["cpu_layer_time_s"] * 1000)
                profile_totals["vk_layer_ms"].append(summary["vulkan_layer_time_s"] * 1000)
                profile_totals["gpu_layer_count"].append(summary["layers_on_gpu"])
                profile_totals["cpu_layer_count"].append(summary["layers_on_cpu"])

        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        next_logits = model.backbone.lm_head(last_hidden)[:, -1, :]

        if step % 20 == 0:
            print(f"  Step {step}: {step_ms:.1f} ms")

    latent_times.sort()
    print(f"\n  Median latent step: {latent_times[len(latent_times)//2]:.1f} ms")
    print(f"  Min: {latent_times[0]:.1f} ms, Max: {latent_times[-1]:.1f} ms")
    print(f"  Total {_STEPS} steps: {sum(latent_times):.1f} ms ({sum(latent_times)/_STEPS:.1f} ms/step avg)")
    print(f"  KV cache length: {past_kv.get_seq_length()}")

    if PROFILE:
        def avg_ms(vals):
            return sum(vals) / len(vals) if vals else 0.0

        n = len(profile_totals["step_total_ms"])
        print("\n  === PROFILING BREAKDOWN ===")
        print(f"  Steps recorded: {n}")
        print(f"  HybridLink avg: {avg_ms(profile_totals['hybridlink_ms']):.2f} ms")
        print(f"  Backbone avg: {avg_ms(profile_totals['backbone_ms']):.2f} ms")
        print(f"  CPU-layer avg total/step: {avg_ms(profile_totals['cpu_layer_ms']):.2f} ms")
        print(f"  Vulkan-layer avg total/step: {avg_ms(profile_totals['vk_layer_ms']):.2f} ms")
        print(f"  GPU-layer count/step: {avg_ms(profile_totals['gpu_layer_count']):.1f}")
        print(f"  CPU-layer count/step: {avg_ms(profile_totals['cpu_layer_count']):.1f}")
        # Warm window: drop the first 10 steps (cold-start / lazy init / compile).
        warm = sorted(profile_totals["step_total_ms"][10:])
        if warm:
            print(f"  Warm step (drop first 10) median: {warm[len(warm)//2]:.1f} ms")
            print(f"  Warm step mean: {sum(warm)/len(warm):.1f} ms")
            print(f"  Warm step p95: {warm[int(len(warm)*0.95)]:.1f} ms")
        # Warm per-device
        warm_cpu = profile_totals["cpu_layer_ms"][10:]
        warm_vk = profile_totals["vk_layer_ms"][10:]
        if warm_cpu:
            print(f"  Warm CPU-layer total/step median: {sorted(warm_cpu)[len(warm_cpu)//2]:.1f} ms")
        if warm_vk:
            print(f"  Warm Vulkan-layer total/step median: {sorted(warm_vk)[len(warm_vk)//2]:.1f} ms")

# Numerical sanity: the decode output must be finite and non-degenerate.
_lh = last_hidden.detach().float()
print(f"\n  Output sanity: finite={bool(torch.isfinite(_lh).all())} "
      f"abs_sum={float(_lh.abs().sum()):.4f} "
      f"mean={float(_lh.mean()):.6f} std={float(_lh.std()):.6f}")

# Verify KV cache continuity
print(f"\n  KV cache integrity: {past_kv is not None}")
print(f"  Number of cached layers: {len(past_kv.layers)}")
print(f"  Cache shape per layer: {past_kv.layers[0].keys.shape}")