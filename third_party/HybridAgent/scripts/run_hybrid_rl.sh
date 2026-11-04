#!/usr/bin/env bash
# HybridRL launcher: NNODES x NPROC_PER_NODE GPUs with a torchrun c10d rendezvous.
#
#   TASK=math|search MODEL=<HybridSFT checkpoint dir or HF id> DATA=<parquet or hf://repo/file> \
#     OUT=outputs/hybrid_rl/math NNODES=4 NODE_RANK=<this node> MASTER_ADDR=<rank-0 host> \
#     bash scripts/run_hybrid_rl.sh [extra hybrid_rl.train args]
#
# Run the same command on every node with its own NODE_RANK. Both tools are live during
# training: TAVILY_API_KEY must be set (for example in .env).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${TASK:?set TASK to math or search}"
MODEL="${MODEL:?set MODEL to a HybridSFT checkpoint dir or HF model id}"
DATA="${DATA:?set DATA to a parquet path or hf://<dataset repo>/<file>}"
OUT="${OUT:-outputs/hybrid_rl/$TASK}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id="hybrid-rl-$TASK" \
  -m hybrid_rl.train --model "$MODEL" --data "$DATA" --task "$TASK" --output_dir "$OUT" "$@"
