# Evaluation

## Protocol

Each request includes an image. The assistant first decides, from the image and request alone, whether to call
personal memory. After a call it receives the top-4 entries retrieved online by SigLIP2 (natural retrieval misses are
kept); otherwise it answers without memory. Decoding is greedy (vLLM).

## Populations

| Population | n | Used for |
|---|---|---|
| main test, generated | 1,028 | inference |
| main test, scored | 999 | memory-access, abstention and hazard metrics (Table 2) |
| grounding episodes | 511 (349 unique requests) | Cite F1, Fact use, Misattr. in Table 2 (target-present requests that need memory, after retrieval relabeling) |
| acute requests | 378 | adequacy, critical harm |
| MIE requests | 107 | memory-informed emergency response |
| interdependency cases | 140 (63 seen / 77 unseen hazard types) | joint success rate (JSR, Table 3); the 12 safe controls of the conflict split are scored separately |

The public dataset has 996 of the 999 scored main requests; the other three show a copyrighted cartoon character and
cannot be distributed.

## Metrics

- **Call BA:** balanced accuracy of the memory-call decision.
- **Cite F1 / Fact use:** end-to-end target citation F1 and decisive-fact use (answers that do not call memory score 0).
- **Misattr.:** using a look-alike's memory.
- **Correct / Needless:** correct abstention when the target's fact is missing; abstention where evidence is present or
  not needed.
- **Adequacy / Crit.:** a specific, non-contraindicated acute response; a judged likely-serious harmful instruction.
- **MIE:** the target's emergency-relevant fact changes the response.
- **JSR:** an adequate, non-critical hazard response that applies the target's fact without misattribution. The 140
  interdependency responses are judged by GPT-6-Luna with full context (full retrieved memory, the decisive fact and the
  clarified reference; prompts in `premind/evaluation/paper/interdependency_judge_prompts.json`). MIE, adequacy and
  severity are judged separately, and a standard emergency call alone does not count as using the fact. Unjudgeable
  responses are excluded from the denominator.

Adequacy, critical harm and MIE are judged by GPT-6-Luna (medium effort) with the frozen rubrics in
`premind/bv3/final_construction/v4_build/`. The same judge supplies the training reward.

## Running

Unpack the evaluation inputs (test release in the evaluation format, retrieval embeddings, judge references) next to
the code:

```bash
tar -xf premind_eval_inputs_v1.tar
cp -r premind_eval_inputs_v1/bv3/. premind/bv3/
export PREMIND_EVAL_OUT=$PWD/runs/paper_eval       # inference outputs and judgments are written here
```

Evaluate a model:

```bash
export PREMIND_4B_LORA=/path/to/PReMind-4B         # local adapter directory
GPU=0 scripts/eval_premind.sh infer premind4b main
GPU=0 scripts/eval_premind.sh infer premind4b conflict
scripts/eval_premind.sh judge premind4b main --execute        # paid GPT-6-Luna calls (OPENAI_API_KEY); shared cap in budget.py
scripts/eval_premind.sh judge premind4b conflict --execute
scripts/eval_premind.sh tables premind4b
```

Without `--execute`, `judge` prints the number of items it would judge. `base4b` runs the zero-shot base model.

`tables` reads the full-context interdependency verdicts from `bv3/eval_inputs/interdependency_rejudge_rows.jsonl`
(rows: model, condition, row_id, seen, `new_JSR_all_axes_raw`); produce them for a new model with the prompts in
`premind/evaluation/paper/interdependency_judge_prompts.json`.

Verified on CPU with the paper's outputs and judgments (not distributed): with the original paths blocked, `tables`
reproduces the Qwen3.5-4B and PReMind-4B rows of Table 2 and Table 3.
