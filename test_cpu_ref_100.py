#!/usr/bin/env python3
"""Stock CPU reference - 100 step latent test (fixed)."""
import torch
import torch.nn as nn
import time
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens, load_hybrid_agent
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM

print("=== STOCK CPU REFERENCE - 100 STEP LATENT ===")

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
add_anchor_tokens(tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(MODEL_DIR)

print("Loading stock BF16 model...")
model, tokenizer = load_hybrid_agent(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device='cpu'
)

# Get backbone
backbone = model.backbone

# Test 100-step latent
print("\n=== 100-STEP LATENT TEST (STOCK CPU) ===")

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    # Prefill
    out = backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_hidden = out.hidden_states[-1][:, -1:, :]
    cache_len = past_kv.get_seq_length()
    print(f"Initial cache length: {cache_len}")

    times = []
    for step in range(100):
        # HybridLink feedback
        injected = model.hybridlink(last_hidden)
        next_input = injected
        cache_position = torch.tensor([cache_len], device=next_input.device)
        
        t0 = time.perf_counter()
        out = backbone(
            inputs_embeds=next_input,
            past_key_values=past_kv,
            use_cache=True,
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

print(f"\nLatent steps/sec: {1000/(sum(times)/100):.1f}")
print("\n=== STOCK CPU REFERENCE COMPLETE ===")