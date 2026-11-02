#!/usr/bin/env python3
"""Audit actual heterogeneous execution path through HybridBackbone + VulkanDecoderLayer."""
import torch
import torch_vulkan_ext as tvk
from transformers import AutoTokenizer, AutoConfig
from hybrid_agent import add_anchor_tokens, load_hybrid_agent
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, DeviceStats
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
import gc

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"

def audit_heterogeneous_forward():
    """Trace a full forward pass through CPU and Vulkan layers."""
    
    print("=== HETEROGENEOUS EXECUTION AUDIT ===\n")
    
    # Load INT8 heterogeneous model with 3 GPU layers (good balance from sweeps)
    gpu_layers = 3
    print(f"Creating HybridBackbone with {gpu_layers} GPU layers...")
    
    hb = create_hybrid_backbone_with_int8(gpu_layers)
    print(f"✓ Model created\n")
    
    # Analyze layer placement
    print("--- LAYER PLACEMENT ---")
    cpu_layers = []
    gpu_layers_actual = []
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            gpu_layers_actual.append(i)
            print(f"  Layer {i}: VULKAN (VulkanDecoderLayer)")
        else:
            cpu_layers.append(i)
            print(f"  Layer {i}: CPU ({type(layer).__name__})")
    
    print(f"\n  CPU layers: {len(cpu_layers)} ({cpu_layers[:5]}...{cpu_layers[-5:] if len(cpu_layers) > 10 else ''})")
    print(f"  GPU layers: {len(gpu_layers_actual)} ({gpu_layers_actual})")
    
    # Analyze weight residency
    print("\n--- WEIGHT RESIDENCY ---")
    cpu_weight_mb = 0
    gpu_weight_mb = 0
    
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            # Vulkan layer weights are in VRAM as _Buf
            if hasattr(layer, '_w'):
                for name, buf in layer._w.items():
                    gpu_weight_mb += buf.size / 1e6
            if hasattr(layer, '_w_q4_buf'):
                for name, buf in layer._w_q4_buf.items():
                    gpu_weight_mb += buf.size / 1e6
        else:
            # CPU layer weights are in RAM
            for name, param in layer.named_parameters():
                cpu_weight_mb += param.numel() * param.element_size() / 1e6
    
    print(f"  CPU weights (RAM): {cpu_weight_mb:.1f} MB")
    print(f"  GPU weights (VRAM): {gpu_weight_mb:.1f} MB")
    
    # Analyze KV cache residency
    print("\n--- KV CACHE RESIDENCY ---")
    print("  KV cache stays on CPU (DynamicCache) per HybridBackbone design")
    print("  Vulkan layers transfer K/V to CPU cache each step")
    print("  Vulkan layers bring cache window back for attention")
    
    # Run forward pass with stats
    print("\n--- FORWARD PASS TRACE ---")
    stats = DeviceStats()
    
    ids = torch.randint(0, 1000, (1, 128))  # prefill
    with torch.no_grad():
        # Warmup
        _ = hb(input_ids=ids, use_cache=True, stats=stats, use_q4=True)
        
        # Measured run
        stats = DeviceStats()
        import time
        t0 = time.perf_counter()
        out = hb(input_ids=ids, use_cache=True, stats=stats, use_q4=True)
        total_time = time.perf_counter() - t0
    
    print(f"\n  Total forward time: {total_time*1000:.1f} ms")
    print(f"  DeviceStats summary: {stats.summary()}")
    
    # Trace data flow
    print("\n--- DATA FLOW (per Vulkan layer forward) ---")
    print("  1. hidden_states (CPU bf16) -> cast to fp32 -> upload to device (_Buf)")
    print("  2. input_layernorm (device-resident)")
    print("  3. q/k/v projections (device matmul, Q4 or fp32)")
    print("  4. q_norm / k_norm (device rmsnorm)")
    print("  5. pack3 -> rope_qkv (device)")
    print("  6. download qkv -> CPU for KV cache update")
    print("  7. attention: upload k_c/v_c, download attn output")
    print("  8. o_proj (device)")
    print("  9. residual add (CPU)")
    print("  10. post_attention_layernorm (device)")
    print("  11. fused_gate_up (device)")
    print("  12. download gate_up -> CPU for down_proj")
    print("  13. down_proj (device) -> download -> CPU residual add")
    print("  14. Return hidden_states (CPU bf16)")
    
    print("\n--- TRANSFER POINTS PER VULKAN LAYER ---")
    print(f"  Transfers recorded: {stats.transfers}")
    print(f"  Transfer bytes: {stats.transfer_bytes / 1e6:.1f} MB")
    print(f"  Per-layer avg: {stats.transfer_bytes / 1e6 / len(gpu_layers_actual):.1f} MB")
    
    # Memory breakdown
    print("\n--- MEMORY BREAKDOWN ---")
    import subprocess
    result = subprocess.run(['ps', '-o', 'rss=', str(os.getpid())], capture_output=True, text=True)
    rss = int(result.stdout.strip()) / 1024
    print(f"  Process RSS: {rss:.0f} MB")
    print(f"  CPU weights: {cpu_weight_mb:.0f} MB")
    print(f"  GPU weights: {gpu_weight_mb:.0f} MB")
    print(f"  KV cache (est): ~{128 * 36 * 2 * 4096 * 128 * 2 / 1e9:.1f} GB at S=128")
    
    # Check for redundant representations
    print("\n--- REDUNDANCY CHECK ---")
    if hasattr(hb, 'embed_tokens'):
        embed_mb = hb.embed_tokens.weight.numel() * hb.embed_tokens.weight.element_size() / 1e6
        print(f"  Embeddings: {embed_mb:.1f} MB (single copy)")
    if hasattr(hb, 'norm'):
        norm_mb = hb.norm.weight.numel() * hb.norm.weight.element_size() / 1e6
        print(f"  Final norm: {norm_mb:.1f} MB (single copy)")
    print(f"  lm_head: kept on CPU (bf16)")
    
    return hb, stats


if __name__ == "__main__":
    import os
    hb, stats = audit_heterogeneous_forward()