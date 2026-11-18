#!/bin/bash
# Paper evaluation on the PReMind-Bench test split (main 1,028 generated / 999 scored requests; 152 conflict rows,
# 140 scored interdependency cases). Needs the evaluation bundle unpacked into premind/ (docs/EVALUATION.md).
#   GPU=0 scripts/eval_premind.sh infer premind4b main        # vLLM inference (PREMIND_4B_LORA = local adapter dir)
#   GPU=0 scripts/eval_premind.sh infer premind4b conflict
#   scripts/eval_premind.sh judge premind4b main --execute    # GPT-6-Luna judging (paid; OPENAI_API_KEY)
#   scripts/eval_premind.sh judge premind4b conflict --execute
#   scripts/eval_premind.sh tables premind4b base4b           # metrics from stored outputs (CPU, no API)
set -euo pipefail
cd "$(dirname "$0")/../premind"
export PREMIND_EVAL_OUT=${PREMIND_EVAL_OUT:-$PWD/../paper_eval}
CMD=${1:?usage: infer|judge|tables ...}; shift
case $CMD in
  infer) KEY=$1 COH=$2; shift 2
         exec "${PYTHON:-python}" -B evaluation/paper/infer.py "$KEY" "$COH" ondemand --gpu "${GPU:?set GPU}" "$@" ;;
  judge) KEY=$1 COH=$2; shift 2
         exec "${PYTHON:-python}" -B evaluation/paper/judge.py "$KEY" "$COH" ondemand "$@" ;;
  tables) exec "${PYTHON:-python}" -B evaluation/paper/tables.py "$@" ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
