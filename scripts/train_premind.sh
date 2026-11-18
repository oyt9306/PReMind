#!/bin/bash
# Train PReMind-4B with the paper recipe (Phase I 125 steps, then Phase II 25 steps from the Phase I adapter).
# Needs: the training-data bundle unpacked into premind/ (see docs/TRAINING.md), two GPUs, OPENAI_API_KEY (GPT-6-Luna
# hazard judge used for rewards), and the pinned environment (pyproject.toml).
#   GPUS=0,1 scripts/train_premind.sh phase1
#   GPUS=0,1 scripts/train_premind.sh phase2       # after phase1; uses runs/cprime_p1_4b_s0/checkpoints/step_000125
set -euo pipefail
cd "$(dirname "$0")/../premind"
PHASE=${1:?usage: scripts/train_premind.sh phase1|phase2}
: "${GPUS:?set GPUS to two comma-separated GPU ids}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (hazard judge)}"
export PREMIND_PHASE1_RUN=${PREMIND_PHASE1_RUN:-$PWD/runs/cprime_p1_4b_s0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1 \
       DDP_TIMEOUT_MIN=180 OMP_NUM_THREADS=8 PIN_WORLD=2 CUR_ACUTE_LADDER=2,2
case $PHASE in
  phase1) export PREMIND_OUT=${PREMIND_OUT:-$PREMIND_PHASE1_RUN} FORCED_P1_EXECUTE=1
          ENTRY=training/cprime/phase1/train_entry_p1.py ;;
  phase2) export PREMIND_OUT=${PREMIND_OUT:-$PWD/runs/cprime_p2_4b_s0} P2E_EXECUTE=1
          ENTRY=training/cprime/phase2/train_entry.py ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
exec "${PYTHON:-python}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$ENTRY"
