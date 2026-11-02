#!/usr/bin/env python3
"""
P3: Placement sweep with GPU-local KV cache for 0, 1, 2, 4, 8 GPU layers
"""

import torch
import torch.nn as nn
import torch_vulkan_ext as tvk
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, _Buf, u32
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import os
import gc
import time
import psutil
import json

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

# GPU KV Cache classes (copy from p2_gpu_kv_test.py)
class GPUKVCache:
    def __init__(self, layer_idx: int, num_kv_heads: int, head_dim: int, max_capacity: int = 4096):
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_capacity = max_capacity
        self.k_cache = None
        self.v_cache = None
        self.valid_len = 0
        self._allocated = False
        
    def _allocate(self, initial_cap: int):
        cap = min(self.max_capacity, max(initial_cap, 256))
        k_elements = 1 * self.num_kv_heads * cap * self.head_dim
        v_elements = 1 * self.num_kv_heads * cap * self.head_dim
        self.k_cache = tvk.Allocation(max(1, k_elements) * 4, "device")
        self.v_cache = tvk.Allocation(max(1, v_elements) * 4, "device")
        self._allocated = True
        
    def _grow(self, required_cap: int):
        new_cap = min(self.max_capacity, max(required_cap, self.valid_len * 2))
        if new_cap <= self.get_capacity():
            return
        k_elements = 1 * self.num_kv_heads * new_cap * self.head_dim
        v_elements = 1 * self.num_kv_heads * new_cap * self.head_dim
        new_k_cache = tvk.Allocation(max(1, k_elements) * 4, "device")
        new_v_cache = tvk.Allocation(max(1, v_elements) * 4, "device")
        if self.k_cache is not None:
            old_cap = self.get_capacity()
            old_k_elements = 1 * self.num_kv_heads * old_cap * self.head_dim
            old_v_elements = 1 * self.num_kv_heads * old_cap * self.head_dim
            tvk.dispatch("slice", [self.k_cache.view(), new_k_cache.view()],
                        u32(old_k_elements) + u32(0),
                        (old_k_elements + 255) // 256, 1, 1, sync=False)
            tvk.dispatch("slice", [self.v_cache.view(), new_v_cache.view()],
                        u32(old_v_elements) + u32(0),
                        (old_v_elements + 255) // 256, 1, 1, sync=False)
            tvk.submit()
        if self.k_cache:
            self.k_cache.free()
        if self.v_cache:
            self.v_cache.free()
        self.k_cache = new_k_cache
        self.v_cache = new_v_cache
        
    def get_capacity(self):
        if not self._allocated:
            return 0
        return self.k_cache.size // (4 * self.num_kv_heads * self.head_dim)
        
    def append_kv(self, k_new: _Buf, v_new: _Buf, S: int, KVD: int):
        if not self._allocated:
            self._allocate(S + 256)
        required_cap = self.valid_len + S
        if required_cap > self.get_capacity():
            self._grow(required_cap)
        k_write_offset = self.valid_len * self.num_kv_heads * self.head_dim
        v_write_offset = self.valid_len * self.num_kv_heads * self.head_dim
        k_elements = S * KVD
        v_elements = S * KVD
        k_dest_view = self.k_cache.view()
        v_dest_view = self.v_cache.view()
        tvk.dispatch("slice", [k_new.buf, k_dest_view], 
                    u32(k_elements) + u32(k_write_offset), 
                    (k_elements + 255)//256, 1, 1, sync=False)
        tvk.dispatch("slice", [v_new.buf, v_dest_view], 
                    u32(v_elements) + u32(v_write_offset), 
                    (v_elements + 255)//256, 1, 1, sync=False)
        tvk.submit()
        self.valid_len += S
        
    def get_kv_views(self, cap: int):
        if not self._allocated or self.valid_len == 0:
            return None, None
        return self.k_cache, self.v_cache
        
    def to_cpu(self, cap: int):
        if not self._allocated or self.valid_len == 0:
            return None, None
        actual_cap = min(cap, self.valid_len)
        elements = 1 * self.num_kv_heads * actual_cap * self.head_dim
        nbytes = elements * 4
        k_cpu = self.k_cache.view(0, nbytes).download((1, self.num_kv_heads, actual_cap, self.head_dim))
        v_cpu = self.v_cache.view(0, nbytes).download((1, self.num_kv_heads, actual_cap, self.head_dim))
        return k_cpu, v_cpu
        
    def get_capacity(self):
        if not self._allocated:
            return 0
        return self.k_cache.size // (4 * self.num_kv_heads * self.head_dim)
        
    def reset(self):
        self.valid_len = 0
        
    def free(self):
        if self.k_cache:
            self.k_cache.free()
            self.k_cache = None
        if self.v_cache:
            self.v_cache.free()
            self.v_cache = None
        self._allocated = False
        self.valid_len = 0


class GPUKVCacheManager:
    def __init__(self, config, gpu_layer_indices: set):
        self.caches = {}
        self.config = config
        for i in gpu_layer_indices:
            self.caches[i] = GPUKVCache(
                layer_idx=i,
                num_kv_heads=config.num_key_value_heads,
                head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                max_capacity=4096
            )
    def get_cache(self, layer_idx: int):
        return self.caches.get(layer_idx)
    def append_kv(self, layer_idx: int, k_new: _Buf, v_new: _Buf, S: int, KVD: int):
        cache = self.caches.get(layer_idx)
        if cache:
            cache.append_kv(k_new, v_new, S, KVD)
    def get_kv_views(self, layer_idx: int, cap: int):
        cache = self.caches.get(layer_idx)
        if cache:
            return cache.get_kv_views(cap)
        return None, None
    def export_to_cpu(self, layer_idx: int, cap: int):
        cache = self.caches.get(layer_idx)
        if cache:
            return cache.to_cpu(cap)
        return None, None
    def free_all(self):
        for cache in self.caches.values():
            cache.free()
        self.caches.clear()
    def reset_all(self):
        for cache in self.caches.values():
            cache.reset()


def build_model(gpu_layers=1):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)
    tvk.init(0)
    torch.manual_seed(0)

    placement = Placement.first_n(gpu_layers, config.num_hidden_layers)
    gpu_layer_indices = set()
    for i in range(config.num_hidden_layers):
        if placement.device_of(i) == "vulkan":
            gpu_layer_indices.add(i)

    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Linear):
            if name == 'lm_head' or name.endswith('.lm_head'):
                continue
            is_vulkan = False
            if 'model.layers.' in name:
                try:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    if layer_idx in gpu_layer_indices:
                        is_vulkan = True
                except:
                    pass
            if is_vulkan:
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

    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)

    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    gc.collect()

    print("Uploading Vulkan weights (Q4)...")
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            layer.upload_weights()
            layer.upload_weights_q4()
    gc.collect()

    # Create GPU KV cache manager
    kv_manager = GPUKVCacheManager(config, {i for i in range(gpu_layers)})

    return model, hb, tokenizer, config, kv_manager


# Monkey-patch VulkanDecoderLayer to use GPU KV cache
def gpu_kv_forward(self, hidden_states, position_embeddings, attention_mask,
            past_key_values=None, cache_position=None, stats=None,
            use_q4: bool = False, **_kwargs):
    self.upload_weights()
    if use_q4:
        self.upload_weights_q4()

    t0 = time.time()
    layer_dtype = hidden_states.dtype
    h = hidden_states.detach().to(torch.float32)
    _keep_alive = []

    def note(op):
        if stats is not None:
            stats.record(op, "vulkan")

    def keep(buf):
        _keep_alive.append(buf)
        return buf

    # ---- input RMSNorm (device-resident)
    note("rmsnorm.input")
    _h_in = keep(_Buf(h.reshape(-1, h.shape[-1])))
    h = self._rms_b(_h_in, self._w["input_ln"], sync=False)
    keep(h)

    # ---- q/k/v projections + QK-norm, all on device (async)
    note("matmul.q_proj")
    if use_q4:
        q = keep(self._lin_b_q4(h, self._w_q4_buf["q"], self._zero_hd,
                            self.num_heads * self.head_dim, self.hidden_size, sync=False))
    else:
        q = keep(self._lin_b(h, self._w["q"], self._zero_hd,
                        self.num_heads * self.head_dim, sync=False))
    note("rmsnorm.q_norm")
    M = q.shape[0]
    q = keep(self._rms_b(q.view(M * self.num_heads, self.head_dim),
                     self._w["q_norm"], sync=False))

    note("matmul.k_proj")
    if use_q4:
        k = keep(self._lin_b_q4(h, self._w_q4_buf["k"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim, self.hidden_size, sync=False))
    else:
        k = keep(self._lin_b(h, self._w["k"], self._zero_kvd,
                        self.num_kv_heads * self.head_dim, sync=False))
    note("rmsnorm.k_norm")
    k = keep(self._rms_b(k.view(M * self.num_kv_heads, self.head_dim),
                     self._w["k_norm"], sync=False))

    note("matmul.v_proj")
    if use_q4:
        v = keep(self._lin_b_q4(h, self._w_q4_buf["v"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim, self.hidden_size, sync=False))
    else:
        v = keep(self._lin_b(h, self._w["v"], self._zero_kvd,
                        self.num_kv_heads * self.head_dim, sync=False))

    # ---- RoPE on the packed [Q|K|V] buffer (async)
    cos, sin = position_embeddings
    S = h.shape[0]
    pos_start = int(cache_position[0].item()) if cache_position is not None else 0
    HD = self.num_heads * self.head_dim
    KVD = self.num_kv_heads * self.head_dim
    q_t = q.view(S, HD)
    k_t = k.view(S, KVD)
    v_t = v.view(S, KVD)
    qkv = keep(tvk.VkBuffer(max(1, S * (HD + 2 * KVD)) * 4, "device"))
    tvk.dispatch(
        "pack3", [q_t.buf, k_t.buf, v_t.buf, qkv],
        u32(S) + u32(HD) + u32(KVD) + u32(KVD),
        (S * (HD + 2 * KVD) + 63) // 64, 1, 1, sync=False)
    note("rope")
    tvk.rope_qkv(qkv, pos_start, self.num_heads,
                 self.num_kv_heads, self.head_dim, n_tok=S, sync=False)

    # Submit RoPE + pack3
    tvk.submit()

    qkv2 = qkv.download((S * (HD + 2 * KVD),))
    q = qkv2[:S * HD].view(S, self.num_heads, self.head_dim)
    k = qkv2[S * HD:S * HD + S * KVD].view(S, self.num_kv_heads, self.head_dim)
    v = qkv2[S * HD + S * KVD:S * HD + 2 * S * KVD].view(S, self.num_kv_heads,
                                                         self.head_dim)

    # ---- GPU KV Cache: Append K/V to GPU cache
    if hasattr(self, '_gpu_kv_cache'):
        kv_manager = self._gpu_kv_cache
        kv_manager.append_kv(self.layer_idx, k_t, v_t, S, KVD)
        
        # Get GPU views for attention
        cap = kv_manager.get_cache(self.layer_idx).valid_len
        k_view, v_view = kv_manager.get_kv_views(self.layer_idx, cap)
        if k_view is not None and v_view is not None:
            kb = keep(k_view)
            vb = keep(v_view)
        else:
            # Fallback to CPU path
            pass

    # Fallback to original CPU path
    if past_key_values is not None and hasattr(past_key_values, 'layers'):
        cached_layer = past_key_values.layers[self.layer_idx]
        cached_k = cached_layer.keys
        cached_v = cached_layer.values
        k_new = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        v_new = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        k_c = torch.cat([cached_k, k_new], dim=2)
        v_c = torch.cat([cached_v, v_new], dim=2)
        cached_layer.keys = k_c
        cached_layer.values = v_c
        cap = k_c.shape[2]
    else:
        k_c = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        v_c = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        cap = k_c.shape[2]
    n_q = S
    first_pos = pos_start
    q_pos = torch.arange(first_pos, first_pos + S)
    k_pos = torch.arange(cap)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    mask = torch.zeros(1, self.num_heads, n_q, cap)
    mask = mask.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
    if os.environ.get("OFFLOAD_ZERO_MASK"):
        mask = torch.zeros(1, self.num_heads, n_q, cap)
    probs = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * cap) * 4, "device"))
    ob = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * self.head_dim) * 4, "host"))
    q_in = q.reshape(n_q, self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
    kb = keep(_Buf(k_c[:, :cap, :].contiguous()))
    vb = keep(_Buf(v_c[:, :cap, :].contiguous()))
    if stats is not None:
        stats.transfers += 2
        stats.transfer_bytes += (kb.buf.size + vb.buf.size)
    note("attention")
    _q_in = keep(_Buf(q_in))
    _mask = keep(_Buf(mask))
    tvk.attention2(_q_in.buf, kb.buf, vb.buf, _mask.buf, probs, ob,
                   0, cap, self.num_heads,
                   self.num_heads // self.num_kv_heads, self.head_dim,
                   cap, self.scaling, sync=False)

    tvk.submit()
    attn = ob.download((self.num_heads, n_q, self.head_dim))
    attn = attn.permute(1, 0, 2).reshape(n_q, HD)
    if stats is not None:
        stats.transfers += 1
        stats.transfer_bytes += ob.size

    # ---- output projection + residual (async)
    note("matmul.o_proj")
    attn_t = attn.contiguous()
    if use_q4:
        out = self._lin_b_q4(attn_t, self._w_q4_buf["o"], self._zero_h,
                              self.hidden_size, self.hidden_size, sync=False).cpu().reshape(
            attn_t.shape[:-1] + (self.hidden_size,))
    else:
        out = self._lin(attn_t, self._w["o"], self._zero_h, self.hidden_size)
    h = hidden_states + out.to(layer_dtype)
    residual = h

    # ---- post-attention RMSNorm + SwiGLU MLP (async)
    note("rmsnorm.post_attn")
    h2 = self._rms(h.detach().to(torch.float32), self._w["post_ln"], sync=False)
    note("fused_gate_up")
    gu = keep(_Buf.out((S, self.config.intermediate_size)))
    _h2 = keep(_Buf(h2))
    tvk.fused_gate_up(_h2.buf, self._w["gate_up"].buf, gu.buf,
                      S, self.hidden_size, self.config.intermediate_size,
                      sync=False)

    tvk.submit()
    gu_t = gu.cpu()
    note("matmul.down_proj")
    if use_q4:
        mlp = self._lin_b_q4(gu_t, self._w_q4_buf["down"], self._zero_h,
                              self.hidden_size, self.config.intermediate_size).cpu().reshape(
            gu_t.shape[:-1] + (self.hidden_size,))
    else:
        mlp = self._lin(gu_t, self._w["down"], self._zero_h, self.hidden_size)
    h = residual + mlp.to(layer_dtype)

    if stats is not None:
        stats.vk_time_s += time.time() - t0
    return h

VulkanDecoderLayer.forward = gpu_kv_forward


def run_latent_benchmark(model, hb, tokenizer, config, kv_manager, num_steps=100, use_q4=True, warmup_steps=5):
    results = {
        "prefill_ms": 0,
        "steps": [],
        "warmup_steps": [],
        "total_ms": 0,
        "peak_rss_mb": 0,
        "peak_vram_mb": 0,
        "final_cache_len": 0,
    }

    ids = torch.randint(0, 1000, (1, 12))
    torch.manual_seed(42)
    
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
    t_prefill_end = time.perf_counter()
    results["prefill_ms"] = (t_prefill_end - t_prefill_start) * 1000
    print(f"  Prefill: {results['prefill_ms']:.1f} ms, cache_len={cache_len}")

    # Attach GPU KV manager to the Vulkan layer
    for layer in hb.layers:
        if isinstance(layer, VulkanDecoderLayer):
            layer._gpu_kv_cache = kv_manager

    times = []
    hybridlink_times = []
    backbone_times = []
    
    with torch.no_grad():
        for step in range(num_steps):
            t_hl_start = time.perf_counter()
            injected = model.hybridlink(last_hidden)
            next_input = injected
            cache_position = torch.tensor([cache_len], device=next_input.device)
            t_hl_end = time.perf_counter()
            hybridlink_times.append((t_hl_end - t_hl_start) * 1000)
            
            t_bb_start = time.perf_counter()
            out = hb(
                inputs_embeds=next_input,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=use_q4,
                cache_position=cache_position,
                output_hidden_states=True
            )
            t_bb_end = time.perf_counter()
            backbone_times.append((t_bb_end - t_bb_start) * 1000)
            
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            cache_len = past_kv.get_seq_length()
            
            step_time = (t_bb_end - t_hl_start) * 1000
            times.append(step_time)
            
            if step < warmup_steps:
                results["warmup_steps"].append(step_time)
            else:
                results["steps"].append(step_time)
            
            if step % 20 == 0:
                rss = get_memory_mb()
                vram = tvk.vram_used() / 1024 / 1024
                results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
                results["peak_vram_mb"] = max(results["peak_vram_mb"], vram)
                print(f"  Step {step}: {step_time:.1f} ms (HL: {hybridlink_times[-1]:.1f}, BB: {backbone_times[-1]:.1f}), cache_len={cache_len}, RSS={rss:.0f}MB, VRAM={vram:.0f}MB")

    results["total_ms"] = sum(times)
    results["final_cache_len"] = cache_len
    
    sustained = results["steps"]
    if sustained:
        sustained_sorted = sorted(sustained)
        results["median_ms"] = sustained_sorted[len(sustained_sorted)//2]
        results["min_ms"] = min(sustained)
        results["max_ms"] = max(sustained)
        results["avg_ms"] = sum(sustained) / len(sustained)
        results["steps_per_sec"] = 1000 / results["avg_ms"]
        results["hybridlink_avg_ms"] = sum(hybridlink_times[warmup_steps:]) / len(sustained)
        results["backbone_avg_ms"] = sum(backbone_times[warmup_steps:]) / len(sustained)
    
    warmup_sorted = sorted(results["warmup_steps"])
    results["warmup_avg_ms"] = sum(results["warmup_steps"]) / len(results["warmup_steps"]) if results["warmup_steps"] else 0
    results["warmup_median_ms"] = warmup_sorted[len(warmup_sorted)//2] if warmup_sorted else 0
    
    return results


def test_placement(gpu_layers):
    print(f"\n{'='*60}")
    print(f"TESTING: {gpu_layers} GPU layer(s) {'(Vulkan Q4)' if gpu_layers > 0 else '(CPU-only)'}")
    print(f"{'='*60}")
    
    use_vulkan = gpu_layers > 0
    model, hb, tokenizer, config, kv_manager = build_model(gpu_layers=gpu_layers)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB" + (f", VRAM: {tvk.vram_used()/1024/1024:.0f} MB" if use_vulkan else ""))
    
    res = run_latent_benchmark(model, hb, tokenizer, config, kv_manager, num_steps=50, use_q4=use_vulkan)
    
    print(f"  Results: avg={res['avg_ms']:.1f} ms, median={res['median_ms']:.1f} ms, {res['steps_per_sec']:.2f} steps/s")
    if use_vulkan:
        print(f"  Peak VRAM: {res['peak_vram_mb']:.0f} MB")
    
    del model, hb
    gc.collect()
    if use_vulkan:
        tvk.submit()  # drain
    
    return res


def main():
    print("=" * 70)
    print("P3: PLACEMENT SWEEP WITH GPU-LOCAL KV CACHE")
    print("=" * 70)
    
    # Test different GPU layer counts
    layer_counts = [0, 1, 2, 4, 8]
    all_results = {}
    
    for gpu_layers in layer_counts:
        all_results[f"gpu_layers_{gpu_layers}"] = test_placement(gpu_layers)
    
    # Summary
    print("\n" + "=" * 70)
    print("P3 PLACEMENT SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'GPU Layers':<12} {'Avg (ms)':<10} {'Median (ms)':<12} {'Steps/s':<10} {'Speedup vs CPU':<15} {'VRAM (MB)':<10}")
    print("-" * 70)
    
    cpu_avg = all_results["gpu_layers_0"]["avg_ms"]
    for name, r in all_results.items():
        gpu_l = int(name.split("_")[-1])
        speedup = cpu_avg / r["avg_ms"] if r["avg_ms"] > 0 else 0
        vram = r.get("peak_vram_mb", 0)
        print(f"{gpu_l:<12} {r['avg_ms']:<10.1f} {r['median_ms']:<12.1f} {r['steps_per_sec']:<10.2f} {speedup:.2f}x             {vram:<10.0f}")
    
    best = max(all_results.items(), key=lambda x: x[1]["steps_per_sec"])
    print(f"\n>>> BEST: {best[0]} with {best[1]['steps_per_sec']:.2f} steps/s")
    if best[0] == "gpu_layers_0":
        print("    CPU-only wins - heterogeneous adds overhead without benefit")
    else:
        print(f"    Heterogeneous ({best[0].split('_')[-1]} GPU layers) is optimal")
    
    with open("p3_placement_sweep_gpu_kv.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to p3_placement_sweep_gpu_kv.json")

if __name__ == "__main__":
    main()