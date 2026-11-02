#!/usr/bin/env python3
"""Test single Vulkan layer with K/V fix."""
import torch
import torch_vulkan_ext as tvk
from backend.hybrid_offload import VulkanDecoderLayer, _Buf, u32
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
import time

print("=== TESTING SINGLE VULKAN LAYER WITH K/V FIX ===")

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

# Test forward
hidden = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
pos_emb = (torch.randn(1, 4096), torch.randn(1, 4096))
cache_pos = torch.tensor([0])
past_kv = DynamicCache()

print('Testing forward...')
out = layer(hidden, position_embeddings=(torch.randn(1,4096), torch.randn(1,4096)), 
            attention_mask=None, past_key_values=DynamicCache(), 
            cache_position=torch.tensor([0]), use_q4=True)
print('SUCCESS: Forward passed')
print(f'Output shape: {out.shape}')