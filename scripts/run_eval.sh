#!/usr/bin/env bash
# =============================================================================
#  Run lmms-eval benchmarks after fine-tuning
#  Usage: bash scripts/run_eval.sh <model_path_or_repo>
#
#  Examples:
#    bash scripts/run_eval.sh outputs/lora_vit/merged
#    bash scripts/run_eval.sh llava-hf/llava-v1.6-mistral-7b-hf  # baseline
# =============================================================================
set -euo pipefail

MODEL="${1:-outputs/lora_vit/merged}"
TASKS="mmbench_en,pope,mme,seedbench"
OUT="results/lmms_eval/$(basename $MODEL)"

echo "Model : $MODEL"
echo "Tasks : $TASKS"
echo "Output: $OUT"
echo ""

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args "pretrained=${MODEL}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUT}" \
    --log_samples

echo ""
echo "Results written to: $OUT"
echo "To compare baseline vs fine-tuned, run for both and diff results.json"
