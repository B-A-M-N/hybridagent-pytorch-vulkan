#!/usr/bin/env python3
"""Test 100-step latent with K/V fix applied."""
import torch
import torch.nn as nn
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, _Buf, u32
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import os
import gc
import time

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

def get_memory_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024

print("=== 100-STEP LATENT TEST WITH K/V FIX ===")

# Apply the K/V fix first
def patch_vulkan_decoder_layer():
    original_forward = VulkanDecoderLayer.forward
    
    def fixed_forward(self, hidden_states, position_embeddings, attention_mask,
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

        # ---- input RMSNorm
        note("rmsnorm.input")
        _h_in = keep(_Buf(h.reshape(-1, h.shape[-1])))
        h = self._rms_b(_h_in, self._w["input_ln"])
        keep(h)

        # ---- q/k/v projections + QK-norm
        note("matmul.q_proj")
        if use_q4:
            q = keep(self._lin_b_q4(h, self._w_q4_buf["q"], self._zero_hd,
                                self.num_heads * self.head_dim, self.hidden_size))
        else:
            q = keep(self._lin_b(h, self._w["q"], self._zero_hd,
                            self.num_heads * self.head_dim))
        note("rmsnorm.q_norm")
        M = q.shape[0]
        q = keep(self._rms_b(q.view(M * self.num_heads, self.head_dim),
                         self._w["q_norm"]))

        note("matmul.k_proj")
        if use_q4:
            k = keep(self._lin_b_q4(h, self._w_q4_buf["k"], self._zero_kvd,
                                self.num_kv_heads * self.head_dim, self.hidden_size))
        else:
            k = keep(self._lin_b(h, self._w["k"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim))
        note("rmsnorm.k_norm")
        k = keep(self._rms_b(k.view(M * self.num_kv_heads, self.head_dim),
                         self._w["k_norm"]))

        note("matmul.v_proj")
        if use_q4:
            v = keep(self._lin_b_q4(h, self._w_q4_buf["v"], self._zero_kvd,
                                self.num_kv_heads * self.head_dim, self.hidden_size))
        else:
            v = keep(self._lin_b(h, self._w["v"], self._zero_kvd,
                            self.num_kv_heads * self.head_dim))

        # ---- RoPE
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
        tvk.submit()
        
        qkv2 = qkv.download((S * (HD + 2 * KVD),))
        q = qkv2[:S * HD].view(S, self.num_heads, self.head_dim)
        k = qkv2[S * HD:S * HD + S * KVD].view(S, self.num_kv_heads, self.head_dim)
        v = qkv2[S * HD + S * KVD:S * HD + 2 * S * KVD].view(S, self.num_kv_heads,
                                                             self.head_dim)

        # ---- attention with FIXED K/V concatenation
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            cached_k = past_key_values.key_cache[self.layer_idx]
            cached_v = past_key_values.value_cache[self.layer_idx]
            k_c = torch.cat([cached_k, k], dim=1)
            v_c = torch.cat([cached_v, v], dim=1)
            past_key_values.key_cache[self.layer_idx] = k_c
            past_key_values.value_cache[self.layer_idx] = v_c
            cap = k_c.shape[1]
        else:
            k_c = k.permute(1, 0, 2).contiguous()
            v_c = v.permute(1, 0, 2).contiguous()
            cap = k_c.shape[1]
        
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

        # ---- output projection + residual
        note("matmul.o_proj")
        attn_t = attn.contiguous()
        if use_q4:
            out = self._lin_b_q4(attn_t, self._w_q4_buf["o"], self._zero_h,
                                  self.hidden_size, self.hidden_size).cpu().reshape(
                attn_t.shape[:-1] + (self.hidden_size,))
        else:
            out = self._lin(attn_t, self._w["o"], self._zero_h, self.hidden_size)
        h = hidden_states + out.to(layer_dtype)
        residual = h

        # ---- post-attention RMSNorm + SwiGLU MLP
        note("rmsnorm.post_attn")
        h2 = self._rms(h.detach().to(torch.float32), self._w["post_ln"])
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

    VulkanDecoderLayer.forward = fixed_forward
    print("✓ VulkanDecoderLayer.forward patched with K/V concatenation fix")

# Apply the patch
patch_vulkan_decoder_layer()

# ============================================================
# MAIN TEST - 100 STEP LATENT
# ============================================================
def get_memory_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024

print("=== 100-STEP LATENT TEST WITH K/V FIX ===")

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

# Build model with 1 GPU layer (Q4)
gpu_layers = 1

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
        quantized = Int8Linear(module.in_features, module.out_features, bias=module.bias is not None, dtype=module.weight.dtype)
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight"] = weight
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    if hasattr(backbone.model.layers[i].self_attn, proj_name):
                        getattr(backbone.model.layers[i].self_attn, proj_name).weight.data.copy_(weight)
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

# Upload Vulkan weights (Q4)
print("Uploading Vulkan weights (Q4)...")
for i, layer in enumerate(hb.layers):
    if isinstance(layer, VulkanDecoderLayer):
        layer.upload_weights()
        layer.upload_weights_q4()
gc.collect()

print(f"Model built. RSS: {get_memory_mb():.0f} MB")

# ============================================================
# 100-STEP LATENT TEST
# ============================================================
print(f"\n=== 100-STEP LATENT TEST ===")

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_hidden = out.hidden_states[-1][:, -1:, :]
    cache_len = past_kv.get_seq_length()
    print(f"Initial cache length: {cache_len}")
    
    times = []
    for step in range(100):
        injected = model.hybridlink(last_hidden)
        next_input = injected
        cache_position = torch.tensor([cache_len], device=next_input.device)
        
        t0 = time.perf_counter()
        out = hb(
            inputs_embeds=next_input,
            past_key_values=past_kv,
            use_cache=True,
            use_q4=True,
            cache_position=cache_position,
            output_hidden_states=True
        )
        t1 = time.perf_counter()
        
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
        
        times.append((t1 - t0) * 1000)
        
        if step % 20 == 0:
            print(f"  Step {step}: {(t1-t0)*1000:.1f} ms, cache_len={cache_len}")

times.sort()
print(f"\n  Median: {times[50]:.1f} ms")
print(f"  Min: {times[0]:.1f} ms, Max: {times[-1]:.1f} ms")
print(f"  Avg: {sum(times)/100:.1f} ms")
print(f"  Final cache: {past_kv.get_seq_length()}")
print(f"  KV cache layers: {len(past_kv.key_cache)}")
print(f"  Cache shape: {past_kv.key_cache[0].shape}")

print(f"\n=== TEST COMPLETE ===")
print(f"100-step latent: COMPLETED ({sum(times)/100:.1f} ms/step avg)")
print(f"Latent steps/sec: {1000/(sum(times)/100):.1f}")