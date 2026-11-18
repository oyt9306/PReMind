# Reproducing the paper

| Step | Command | Needs |
|---|---|---|
| CPU tests | `bash scripts/run_tests.sh` | this repository |
| Inference on the test split | `GPU=0 scripts/eval_premind.sh infer <key> main` and `... conflict` | evaluation inputs, one GPU, vLLM |
| Judging | `scripts/eval_premind.sh judge <key> main --execute` and `... conflict --execute` | `OPENAI_API_KEY` (GPT-6-Luna, paid) |
| Metrics | `scripts/eval_premind.sh tables <key>` | your outputs and judgments |
| Training | `GPUS=0,1 scripts/train_premind.sh phase1`, then `phase2` | training bundle (on request), two GPUs, `OPENAI_API_KEY` |

Environment: Python 3.12 with the versions pinned in `pyproject.toml` (the paper's conda environment). Base models are
loaded at pinned revisions: Qwen/Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Judged metrics depend on the judge model; re-judging with another model or a later version of GPT-6-Luna can change
adequacy, critical harm, MIE and JSR. 
