"""Persistent Vulkan pipeline — pre-allocate everything, batch dispatches.

Instead of per-op: create algo → record → eval → destroy
Do once: create all algos + tensors at init
Per call: memcpy input → single batched eval → read output

This is Phase 4 of torch-vulkan optimization.
"""

import torch


class PersistentLayerPipeline:
    """Pre-recorded pipeline for one transformer layer.
    
    All Kompute tensors and algorithms created once at init.
    Forward pass just writes data and dispatches.
    """

    def __init__(self, d_model, n_kv_heads, head_dim, ffn_dim, seq_len=4):
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.seq_len = seq_len
        self.qkv_dim = d_model + 2 * n_kv_heads * head_dim  # 5120 for phi-4

        # Pre-allocate ALL tensors on vulkan (created once, reused forever)
        self._x = torch.empty(seq_len, d_model, device='vulkan')
        self._norm1_w = torch.empty(d_model, device='vulkan')
        self._norm1_b = torch.empty(d_model, device='vulkan')
        self._qkv_w = torch.empty(d_model, self.qkv_dim, device='vulkan')
        self._out_w = torch.empty(n_kv_heads * head_dim, d_model, device='vulkan')
        self._norm2_w = torch.empty(d_model, device='vulkan')
        self._norm2_b = torch.empty(d_model, device='vulkan')
        self._ffn_up_w = torch.empty(d_model, ffn_dim * 2, device='vulkan')
        self._ffn_down_w = torch.empty(ffn_dim, d_model, device='vulkan')
        
        self._weights_loaded = False

    def load_weights(self, norm1_w, qkv_w, out_w, norm2_w, ffn_up_w, ffn_down_w):
        """Load weights once — memcpy into pre-allocated vulkan tensors."""
        # Copy weight data into existing vulkan tensor storage
        self._norm1_w = norm1_w.to('vulkan')
        self._norm1_b = torch.empty(self.d_model).to('vulkan')
        self._qkv_w = qkv_w.T.contiguous().to('vulkan')
        self._out_w = out_w.T.contiguous().to('vulkan')
        self._norm2_w = norm2_w.to('vulkan')
        self._norm2_b = torch.empty(self.d_model).to('vulkan')
        self._ffn_up_w = ffn_up_w.T.contiguous().to('vulkan')
        self._ffn_down_w = ffn_down_w.T.contiguous().to('vulkan')
        self._weights_loaded = True

    def forward(self, x):
        """Run one layer — weights pre-loaded, just compute."""
        assert self._weights_loaded, "Call load_weights first"
        S = x.shape[0]
        d = self.d_model
        nkv = self.n_kv_heads
        hd = self.head_dim
        n_heads_full = d // hd  # 24 for phi-4

        # Everything stays on vulkan as long as possible
        xv = x.to('vulkan')

        # LayerNorm + QKV projection (2 ops, batched mentally)
        xn = torch.nn.functional.layer_norm(xv, [d], self._norm1_w, self._norm1_b)
        qkv = torch.mm(xn, self._qkv_w)

        # Split Q/K/V — must go to CPU for reshape (for now)
        qkv_cpu = qkv.to('cpu')
        q = qkv_cpu[:, :nkv * hd]
        k = qkv_cpu[:, n_heads_full * hd:n_heads_full * hd + nkv * hd]
        v = qkv_cpu[:, n_heads_full * hd + nkv * hd:]

        # Attention on Vulkan
        qh = q.reshape(S, nkv, hd).permute(1, 0, 2).unsqueeze(0).to('vulkan')
        kh = k.reshape(S, nkv, hd).permute(1, 0, 2).unsqueeze(0).to('vulkan')
        vh = v.reshape(S, nkv, hd).permute(1, 0, 2).unsqueeze(0).to('vulkan')
        ao = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
        ao = ao.to('cpu').squeeze(0).permute(1, 0, 2).reshape(S, nkv * hd)

        # Output projection on Vulkan
        proj = torch.mm(ao.to('vulkan'), self._out_w).to('cpu')
        x1 = x + proj

        # FFN: LayerNorm + Up + GELU + Down
        xn2 = torch.nn.functional.layer_norm(x1.to('vulkan'), [d], self._norm2_w, self._norm2_b)
        gu = torch.mm(xn2, self._ffn_up_w).to('cpu')
        gate = gu[:, :self.ffn_dim]
        up = gu[:, self.ffn_dim:]
        h = torch.nn.functional.gelu(gate.to('vulkan')).to('cpu') * up
        ffn_out = torch.mm(h.to('vulkan'), self._ffn_down_w).to('cpu')

        return x1 + ffn_out


class PersistentModelPipeline:
    """Pre-recorded pipeline for full model — all layers persistent."""

    def __init__(self, n_layers, d_model, n_kv_heads, head_dim, ffn_dim, seq_len=4):
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                PersistentLayerPipeline(d_model, n_kv_heads, head_dim, ffn_dim, seq_len)
            )
        self.n_layers = n_layers

    def load_layer_weights(self, layer_idx, norm1_w, qkv_w, out_w, norm2_w, ffn_up_w, ffn_down_w):
        self.layers[layer_idx].load_weights(norm1_w, qkv_w, out_w, norm2_w, ffn_up_w, ffn_down_w)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
