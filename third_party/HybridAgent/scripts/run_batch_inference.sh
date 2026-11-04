#!/usr/bin/env bash
# Batch inference on N GPUs: one process per GPU, questions sharded round-robin, shards merged.
#
#   bash scripts/run_batch_inference.sh <num_gpus> --model <ckpt> --input <questions.jsonl> --output <out.jsonl> [args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
N="${1:?number of GPUs}"; shift
OUTPUT=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [ "${ARGS[$i]}" = "--output" ]; then OUTPUT="${ARGS[$((i + 1))]}"; fi
done
[ -n "$OUTPUT" ] || { echo "--output is required"; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"
PIDS=()
for ((g = 0; g < N; g++)); do
  SHARD_ARGS=()
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--output" ]; then SHARD_ARGS+=("--output" "$OUTPUT.shard$g"); i=$((i + 1)); else SHARD_ARGS+=("${ARGS[$i]}"); fi
  done
  CUDA_VISIBLE_DEVICES="$g" python -m hybrid_inference.run_batch "${SHARD_ARGS[@]}" --shard_id "$g" --num_shards "$N" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
cat "$OUTPUT".shard* > "$OUTPUT"
echo "merged $(wc -l < "$OUTPUT") records into $OUTPUT"
