# Training PReMind-4B

PReMind-4B is trained from `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) with group-relative,
DAPO-style reinforcement learning and LoRA (rank 32, alpha 64, dropout 0, language-model attention and MLP).

## Recipe

- **Interaction regimes.** *On-demand* episodes sample the memory-call decision, so both the decision and the answer are
  trained. *Forced-call* episodes fix the call and mask the first-turn decision tokens, so evidence use is trained even
  when the policy would skip retrieval (forced probability 0.5 on safe rows, 0.4 on unsafe target-present and 0.4 on
  target-absent rows). Both phases use this mixture; at inference the model decides on demand.
- **Phase I** (125 steps from the base model) trains memory access, grounding and hazard response, with a penalty
  against stock instructions repeated regardless of the scene.
- **Phase II** (25 steps from the Phase I adapter, fresh optimizer) focuses on difficult memory-dependent hazards: a fact
  bonus (W_F = 0.15) for stating and citing the decisive fact, curriculum filtering from Phase I difficulty, experience
  replay of cases where the relevant fact was not used, and action-prioritized hazard credit (stock-phrase penalty
  λ_p = 0.2).
- **Batch.** 4 prompts × 8 completions per step on two GPUs, KL β 0.02, seed 0.
- **Judge.** GPT-6-Luna (medium effort) supplies the hazard reward. The trainer calibrates the judge at start-up and
  enforces per-run spending caps (see `phase*/args_4b.json`).

The exact arguments of the paper run are in `premind/training/cprime/phase{1,2}/args_4b.json`; the original pin blocks
(source and data hashes of the paper run) are in `premind/training/cprime/provenance/`.

## Data

The training data are not public; they are available from the authors on request for research use. The training
bundle contains the train/dev releases (`gt_retrieved_drop` contexts), the hazard key actions, the curriculum and MIE
inventories and the other inputs the trainer reads, laid out relative to `premind/`. Unpack it into `premind/`:

```bash
tar -xf premind_train_data_v1.tar -C premind/      # creates premind/bv3/... and premind/bv3/PATHMAP_train.json
```

Data files are byte-identical to the paper run, so their integrity checks pass. Some of them store absolute paths of
the original machine; `premind/pathmap.py` maps those to your copy at run time (`PATHMAP_*.json`).

## Running

```bash
export OPENAI_API_KEY=...                          # GPT-6-Luna hazard judge
GPUS=0,1 scripts/train_premind.sh phase1           # -> premind/runs/cprime_p1_4b_s0
GPUS=0,1 scripts/train_premind.sh phase2           # -> premind/runs/cprime_p2_4b_s0; final adapter: checkpoints/step_000025/lora
```

`PREMIND_PHASE1_RUN` selects the Phase I run that Phase II starts from (its `checkpoints/step_000125/lora` adapter and
its `.curriculum` state). `PREMIND_PIN_CHECK=1` additionally compares the run's pin block with the original pins; this
only matches on the original machine layout.

Verified on CPU: both phases load all code and data from a copy of this repository and the bundle without touching the
original paths, and stop at the judge calibration when no API access is available. Full training needs two GPUs and has not
been re-run from this copy.
