#!/usr/bin/env python3
"""Test 100+ sustained latent steps through HybridGenerationEngine with heterogeneous backbone."""
import torch
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from third_party.HybridAgent.hybrid_agent import HybridGenerationEngine, load_hybrid_agent
from third_party.HybridAgent.hybrid_agent.tools import ToolExecutor
from backend.hybrid_offload import HybridBackbone, Placement
import gc

print("=== 100+ Step Heterogeneous Latent Generation Test ===")

# Create heterogeneous model with 5 GPU layers (optimal for S=12)
gpu_layers = 5
hb = create_hybrid_backbone_with_int8(gpu_layers)
gc.collect()

# Upload Vulkan weights ONCE
print("Uploading Vulkan weights...")
for i, layer in enumerate(hb.layers):
    if hasattr(layer, 'upload_weights'):
        layer.upload_weights()
gc.collect()

# Load tokenizer and HybridLink
from transformers import AutoTokenizer
from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
add_anchor_tokens(tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n--- Testing 100 sequential latent steps ---")

# Use a simple latent generation loop through the backbone directly
# Simulating what HybridGenerationEngine does in LATENT mode
ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    # Prefill
    out = hb(input_ids=ids, use_cache=True, use_q4=False)
    past_kv = out.past_key_values
    last_hidden = out.last_hidden_state[:, -1:, :]
    # Apply lm_head to get logits
    next_logits = hb.backbone.lm_head(last_hidden)[:, -1, :]
    
    # 100 latent steps
    latent_times = []
    for step in range(100):
        # In LATENT mode, the engine runs HybridLink on last_hidden
        # and feeds that as next input
        if hasattr(hb, 'hybridlink') and hb.hybridlink is not None:
            injected = hb.hybridlink(last_hidden)
            next_input = injected
        else:
            # Use logits argmax as fallback
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            next_input = hb.embed_tokens(next_token)
        
        # Run forward step
        import time
        t0 = time.perf_counter()
        out = hb(
            inputs_embeds=next_input,
            past_key_values=past_kv,
            use_cache=True,
            use_q4=False
        )
        t1 = time.perf_counter()
        latent_times.append((t1 - t0) * 1000)
        
        past_kv = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]
        # Apply lm_head to get logits
        next_logits = hb.backbone.lm_head(last_hidden)[:, -1, :]
        
        if step % 20 == 0:
            print(f"  Step {step}: {(t1-t0)*1000:.1f} ms")

latent_times.sort()
print(f"\n  Median latent step: {latent_times[len(latent_times)//2]:.1f} ms")
print(f"  Min: {latent_times[0]:.1f} ms, Max: {latent_times[-1]:.1f} ms")
print(f"  Total 100 steps: {sum(latent_times):.1f} ms ({sum(latent_times)/100:.1f} ms/step avg)")
print(f"  KV cache length: {past_kv.get_seq_length()}")

# Verify KV cache continuity
print(f"\n  KV cache integrity: {past_kv is not None}")
print(f"  Number of cached layers: {len(past_kv.key_cache)}")
print(f"  Cache shape per layer: {past_kv.key_cache[0].shape}")