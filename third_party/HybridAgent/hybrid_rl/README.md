# HybridRL

Hybrid reinforcement learning scales the agent's reasoning by learning when to think in
latent space and when to act through text. Each training step samples a group of hybrid
trajectories per question (**hybrid policy rollout**), scores them with
r = r_correct + r_format + r_tool, computes group-normalized advantages, and updates the
policy over the sampled text positions by replaying the recorded trajectories in embedding
space (**exact rollout replay**). HybridLink stays frozen. See the paper for the method.

## Reward

| Term | Math | Search |
|---|---|---|
| `r_correct` | Math-Verify equivalence of the boxed answer, 0 / 1 | token-level F1 of the boxed answer, in [0, 1] |
| `r_format` | 0, or a penalty when the trajectory is malformed (unpaired `<think>` or `<latent>` tags, missing `<answer>` with `\boxed{}`, a tool call without its `<result>`) | same |
| `r_tool` | a small bonus when the answer is correct and both tools were used | same |

## Data

A parquet file with the columns `ability` (`math` or `qa`), `reward_model` (`{"ground_truth": ...}`) and `extra_info` (`{"question": ...}`). `--data` accepts a local path or `hf://<dataset repo>/<file>`.

## Training

```bash
TASK=search MODEL=<HybridSFT checkpoint dir or HF id> DATA=hf://togethercomputer/HybridAgent-HybridRL/search.parquet \
    NNODES=4 NODE_RANK=<this node> MASTER_ADDR=<rank-0 host> bash scripts/run_hybrid_rl.sh
```

Run the same command on every node with its `NODE_RANK`. Rollouts are data-parallel over questions. Both tools are live during training, so `TAVILY_API_KEY` is required.

Recommended hardware: 32 GPUs (4 nodes × 8), which is the launcher default (`NNODES=4`, `NPROC_PER_NODE=8`); each rank then rolls out 2 prompts × 8 trajectories per step. Other world sizes work as long as `--prompts_per_step` is divisible by the number of GPUs, with proportionally more rollouts per GPU.

`TASK=math` selects the deep-reasoning setting (`math.parquet`). The per-task defaults (training length, tool-call cap, per-trajectory budget, latent-span bounds, …) live in `hybrid_rl/config.py`, can be overridden on the command line, and are listed by `python -m hybrid_rl.train --help`.

## Output

`train.log` and `metrics.jsonl` record per-step rewards, format and tool-use rates, latent and text usage, the loss, the clip fraction and the replay-vs-rollout agreement of the log-probs before the first update of each step. `rollouts.jsonl` keeps a few sampled trajectories per step. Checkpoints (`checkpoint-N/`) are complete HybridAgent checkpoints ready for inference.
