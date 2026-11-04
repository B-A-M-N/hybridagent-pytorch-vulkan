#!/usr/bin/env python3
"""Verify K/V concatenation fix with deterministic cache test."""
import torch
import torch_vulkan_ext as tvk
from backend.hybrid_offload import VulkanDecoderLayer, _Buf, u32
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
import time

print("=== VERIFYING K/V CONCATENATION FIX ===")

# Create a single Vulkan layer
config = AutoConfig.from_pretrained('/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic')
layer = VulkanDecoderLayer(config, 0)
layer.upload_weights()
layer.upload_weights_q4()

# Monkey-patch the forward with our fix
def fixed_forward(self, hidden_states, position_embeddings, attention_mask,
            past_key_values=None, cache_position=None, stats=None,
            use_q4: bool = False, **_kwargs):
    self.upload_weights()
    self.upload_weights_q4()

    t0 = time.time()
    layer_dtype = hidden_states.dtype
    h = hidden_states.detach().to(torch.float32)
    _keep_alive = []

    def note(op):
        pass

    def keep(buf):
        return buf

    note('rmsnorm.input')
    _h_in = _Buf(h.reshape(-1, h.shape[-1]))
    h = self._rms_b(_h_in, self._w['input_ln'])
    keep(h)

    note('matmul.q_proj')
    q = keep(self._lin_b_q4(h, self._w_q4_buf['q'], self._zero_hd,
                        self.num_heads * self.head_dim, self.hidden_size))
    note('rmsnorm.q_norm')
    M = q.shape[0]
    q = keep(self._rms_b(q.view(M * self.num_heads, self.head_dim),
                     self._w['q_norm']))

    note('matmul.k_proj')
    k = keep(self._lin_b_q4(h, self._w_q4_buf['k'], self._zero_kvd,
                        self.num_kv_heads * self.head_dim, self.hidden_size))
    note('rmsnorm.k_norm')
    k = keep(self._rms_b(k.view(M * self.num_kv_heads, self.head_dim),
                     self._w['k_norm']))

    note('matmul.v_proj')
    v = keep(self._lin_b_q4(h, self._w_q4_buf['v'], self._zero_kvd,
                        self.num_kv_heads * self.head_dim, self.hidden_size))

    cos, sin = position_embeddings
    S = h.shape[0]
    pos_start = int(cache_position[0].item()) if cache_position is not None else 0
    HD = self.num_heads * self.head_dim
    KVD = self.num_kv_heads * self.head_dim
    q_t = q.view(S, HD)
    k_t = k.view(S, KVD)
    v_t = v.view(S, KVD)
    qkv = _Buf(tvk.VkBuffer(max(1, S * (HD + 2 * KVD)) * 4, 'device'))
    tvk.dispatch(
        'pack3', [q_t.buf, k_t.buf, v_t.buf, qkv],
        u32(S) + u32(HD) + u32(KVD) + u32(KVD),
        (S * (HD + 2 * KVD) + 63) // 64, 1, 1, sync=False)
    tvk.rope_qkv(qkv, pos_start, self.num_heads,
                 self.num_kv_heads, self.head_dim, n_tok=S, sync=False)
    tvk.submit()
    
    qkv2 = qkv.download((S * (HD + 2 * KVD),))
    q = qkv2[:S * HD].view(S, self.num_heads, self.head_dim)
    k = qkv2[S * HD:S * HD + S * KVD].view(S, self.num_kv_heads, self.head_dim)
    v = qkv2[S * HD + S * KVD:S * HD + 2 * S * KVD].view(S, self.num_kv_heads,
                                                         self.head_dim)

    # FIXED K/V concatenation
    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
        cached_k = past_key_values.key_cache[self.layer_idx]
        cached_v = past_key_values.value_cache[self.layer_idx]
        # Verify cached values are preserved
        print(f"  Cached K shape: {cached_k.shape}, V shape: {cached_v.shape}")
        print(f"  New K shape: {k.shape}, V shape: {v.shape}")
        k_c = torch.cat([cached_k, k], dim=1)
        v_c = torch.cat([cached_v, v], dim=1)
        past_key_values.key_cache[self.layer_idx] = k_c
        past_key_values.value_cache[self.layer_idx] = v_c
        cap = k_c.shape[1]
        print(f"  After concat: K shape: {k_c.shape}, V shape: {v_c.shape}")
    else:
        k_c = k.permute(1, 0, 2).contiguous()
        v_c = v.permute(1, 0, 2).contiguous()
        cap = k_c.shape[1]
        print(f"  No cache: K shape: {k_c.shape}, V shape: {v_c.shape}")
    
    n_q = S
    first_pos = pos_start
    q_pos = torch.arange(first_pos, first_pos + S)
    k_pos = torch.arange(cap)
    allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    mask = torch.zeros(1, self.num_heads, n_q, cap)
    mask = mask.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * cap) * 4, 'device'))
    ob = keep(tvk.VkBuffer(max(1, self.num_heads * n_q * self.head_dim) * 4, 'host'))
    q_in = q.reshape(n_q, self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
    kb = keep(_Buf(k_c[:, :cap, :].contiguous()))
    vb = keep(_Buf(v_c[:, :cap, :].contiguous()))
    note('attention')
    _q_in = keep(_Buf(q_in))
    _mask = keep(_Buf(mask))
    tvk.attention2(_q_in.buf, kb.buf, vb.buf, _mask.buf, probs, ob,
                   0, cap, self.num_heads,
                   self.num_heads // self.num_kv_heads, self.head_dim,
                   cap, self.scaling, sync=False)
    tvk.submit()
    attn = ob.download((self.num_heads, n_q, self.head_dim))
    attn = attn.permute(1, 0, 2).reshape(n_q, HD)

    note('matmul.o_proj')
    attn_t = attn.contiguous()
    out = self._lin_b_q4(attn_t, self._w_q4_buf['o'], self._zero_h,
                          self.hidden_size, self.hidden_size).cpu().reshape(
        attn_t.shape[:-1] + (self.hidden_size,))
    h = hidden_states + out.to(layer_dtype)
    return h

layer.forward = fixed_forward.__get__(layer, type(layer))

# Test the fix with deterministic cache values
print("=== Testing K/V concatenation with deterministic values ===")

# Create a DynamicCache with known values
past_kv = DynamicCache()
# Pre-populate with known values
batch, n_kv, seq_len, head_dim = 1, 8, 5, 128
cached_k = torch.arange(batch * n_kv * seq_len * head_dim, dtype=torch.float32).view(batch, n_kv, seq_len, head_dim)
cached_v = torch.arange(batch * n_kv * seq_len * head_dim, 1000, dtype=torch.float32).view(batch, n_kv, seq_len, head_dim)
past_kv.key_cache.append(cached_k)
past_kv.value_cache.append(cached_v)

print(f"Initial cache K shape: {past_kv.key_cache[0].shape}")
print(f"Initial cache V shape: {past_kv.value_cache[0].shape}")

# Create test input
hidden = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
pos_emb = (torch.randn(1, 4096), torch.randn(1, 4096))
cache_pos = torch.tensor([5])  # cache length = 5
cache_position = torch.tensor([5])

# Test forward
print("\n--- First forward (should append to cache) ---")
out = layer(hidden, position_embeddings=(torch.randn(1, 4096), torch.randn(1, 4096)), 
            attention_mask=None, past_key_values=past_kv, 
            cache_position=torch.tensor([5]), use_q4=True)
print(f"Output shape: {out.shape}")

# Check cache was updated correctly
print(f"After step 0 - Cache K shape: {past_kv.key_cache[0].shape}")
print(f"After step 0 - Cache V shape: {past_kv.value_cache[0].shape}")
print(f"Cache length: {past_kv.get_seq_length()}")

# Verify values are preserved
expected_seq_len = 6  # 5 + 1
assert past_kv.key_cache[0].shape[2] == expected_seq_len, f"Expected seq_len {expected_seq_len}, got {past_kv.key_cache[0].shape[2]}"
assert past_kv.value_cache[0].shape[2] == expected_seq_len

# Check first 5 positions match original
orig_k = cached_k[0, 0, :5, 0]
new_k = past_kv.key_cache[0][0, 0, :5, 0]
assert torch.allclose(orig_k, new_k), "Historical K values not preserved!"
print("✓ Historical K values preserved")

# Second forward - should append again
print("\n--- Second forward (should append again) ---")
out2 = layer(out2 if 'out2' in locals() else hidden, position_embeddings=(torch.randn(1, 4096), torch.randn(1, 4096)), 
             attention_mask=None, past_key_values=past_kv,
             cache_position=torch.tensor([6]), use_q4=True)

print(f"After step 1 - Cache K shape: {past_kv.key_cache[0].shape}")
print(f"After step 1 - Cache V shape: {past_kv.value_cache[0].shape}")
print(f"Cache length: {past_kv.get_seq_length()}")

assert past_kv.key_cache[0].shape[2] == 7, f"Expected 7, got {past_kv.key_cache[0].shape[2]}"
assert past_kv.value_cache[0].shape[2] == 7

# Check first 6 positions still match
assert torch.allclose(cached_k[0, 0, :5, 0], past_kv.key_cache[0][0, 0, :5, 0]), "Historical K values corrupted!"
print("✓ Historical K values preserved after second step")

print("\n=== K/V Concatenation Fix VERIFIED ===")
print("✓ Cache correctly appended")
print("✓ Historical values preserved")
print("✓ No duplicate appends")
print("✓ Cache grows correctly")