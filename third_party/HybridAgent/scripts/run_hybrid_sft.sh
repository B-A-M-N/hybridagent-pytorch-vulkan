#!/usr/bin/env bash
# HybridSFT: format activation -> HybridLink activation -> self-conditioned HybridSFT.
#
#   MODEL=Qwen/Qwen3-8B-Base DATA=<jsonl path or HF dataset id> OUT=outputs/hybrid_sft \
#     bash scripts/run_hybrid_sft.sh [format_activation|hybridlink_activation|self_conditioned|all] [extra args]
#
# CONFIG selects the accelerate config (default: ZeRO-2 on 8 GPUs); set
# CONFIG_SELF_CONDITIONED=hybrid_sft/configs/accelerate_zero2_offload.yaml for the 14B model.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-8B-Base}"
DATA="${DATA:?set DATA to a local JSONL file or a Hugging Face dataset repo id}"
DATA_FILE="${DATA_FILE:-train.jsonl}"
OUT="${OUT:-outputs/hybrid_sft}"
CONFIG="${CONFIG:-hybrid_sft/configs/accelerate_zero2.yaml}"
CONFIG_SELF_CONDITIONED="${CONFIG_SELF_CONDITIONED:-$CONFIG}"
STAGE="${1:-all}"
shift || true
EXTRA=("$@")   # extra hybrid_sft.train arguments

run_stage() {  # config stage init_model
  accelerate launch --config_file "$1" -m hybrid_sft.train \
    --stage "$2" --model "$3" --data "$DATA" --data_file "$DATA_FILE" \
    --output_dir "$OUT/$2" "${EXTRA[@]}"
}

case "$STAGE" in
  format_activation)     run_stage "$CONFIG" format_activation "$MODEL" ;;
  hybridlink_activation) run_stage "$CONFIG" hybridlink_activation "$OUT/format_activation" ;;
  self_conditioned)      run_stage "$CONFIG_SELF_CONDITIONED" self_conditioned "$OUT/hybridlink_activation" ;;
  all)
    run_stage "$CONFIG" format_activation "$MODEL"
    run_stage "$CONFIG" hybridlink_activation "$OUT/format_activation"
    run_stage "$CONFIG_SELF_CONDITIONED" self_conditioned "$OUT/hybridlink_activation"
    ;;
  *) echo "usage: bash scripts/run_hybrid_sft.sh [format_activation|hybridlink_activation|self_conditioned|all]"; exit 1 ;;
esac
