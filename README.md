# PReMind: When Personal Memory Meets Visual Hazards

**Authors**: Yeongtak Oh, Donghyeok Lee, Joonhee Oh, Jaewon Jeong, Sungroh Yoon

<p align="center">
  <a href="https://oyt9306.github.io/premind.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/🌐%20Project%20Page-PReMind-blue?style=for-the-badge" alt="Project Page"/>
  </a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PReMind--4B-yellow)](https://huggingface.co/Yeongtak/PReMind-4B)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PReMind--Bench-yellow)](https://huggingface.co/datasets/Yeongtak/PReMind-Bench)

> **Content warning:** the benchmark contains synthetic images of hazardous situations, such as fire and injuries.

---

## Abstract

Personalized vision-language assistants use visual observations and personal memories to provide advice tailored to the user. When the observed scene contains a physical hazard, personal context, such as allergies, medical conditions, or mobility constraints, can change what help is appropriate. Responding appropriately therefore requires recognizing the visible hazard and applying the relevant person's information.

We introduce **PReMind-Bench**, a diagnostic benchmark that evaluates this combination across diverse hazards and personal-memory conditions. It separately measures memory access, grounding, abstention, and hazard response, and assesses joint success on interdependency cases. Existing models often satisfy one requirement while failing the other, and training only for safety or personalization leaves a gap in joint success. We therefore propose **PReMind**, a two-phase reinforcement learning framework that combines on-demand and forced-call interactions (memory always retrieved) to jointly train grounded memory use and hazard response. At 4B, PReMind raises joint success from 12.9% to 28.6% over its base on the interdependency set and improves hazard adequacy and correct abstention on the main test.

---

## Benchmark

Every request comes with an image. The model first decides whether it needs the user's personal memory; if it calls memory, it receives the top-4 retrieved entries and then answers.

<p align="center">
  <img src="./imgs/scenario_map.jpg" alt="Scenario map: one scene supports requests that need a personal fact, safety knowledge, both, or only the image." width="80%">
</p>

| | |
|---|---|
| Public test split | 1,020 main requests + 112 test-only rows |
| Interdependency set | 140 cases (63 seen / 77 unseen hazard groups) + 12 safe controls |
| Rule-based metrics | Call BA, Cite F1, Fact use, Misattribution, correct / needless abstention |
| Judged metrics (GPT-6-Luna) | hazard adequacy, critical harm, memory-informed emergency response (MIE), joint success rate (JSR) |

The test split is on [Hugging Face](https://huggingface.co/datasets/Yeongtak/PReMind-Bench) (gated, CC BY-NC 4.0). The training data are not public; they are shared for non-commercial research after an approved request to the authors.

---

## Results

| | Call BA ↑ | Cite F1 ↑ | Fact use ↑ | Misattr. ↓ | Correct ↑ | Needless ↓ | Adequacy ↑ | Crit. ↓ | MIE ↑ | JSR ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 80.4 | 56.2 | 41.7 | 3.1 | 26.1 | 7.9 | 34.4 | 9.5 | 24.3 | 12.9 |
| **PReMind-4B** | **85.9** | **83.2** | **58.5** | **1.2** | **55.8** | **1.6** | **56.9** | **8.7** | **54.2** | **28.6** |

Main-test metrics (Table 2) and on-demand JSR on the interdependency set (Table 3). Over three Phase II seeds, JSR is 27.4 ± 2.1%. See the paper for all baselines.

---

## Project Structure

```
PReMind/
├── premind/
│   ├── evaluation/paper/        # Evaluation used in the paper
│   │   ├── infer.py             # vLLM inference on the test split (main / interdependency)
│   │   ├── judge.py             # GPT-6-Luna judging (adequacy, severity, MIE)
│   │   ├── tables.py            # Table 2 / Table 3 metrics
│   │   ├── grounding511.py      # Cite F1 / Fact use / Misattr. on the 511 grounding episodes
│   │   └── interdependency_judge_prompts.json
│   ├── training/cprime/         # PReMind training
│   │   ├── phase1/              # Phase I (125 steps from Qwen3.5-4B)
│   │   └── phase2/              # Phase II (25 steps from the Phase I adapter)
│   ├── core/                    # Shared code: data loaders, retrieval, rewards, judges, trainer
│   ├── construction/            # Benchmark construction pipeline
│   ├── prompts/  rewards/       # Prompt templates and rule-based scorers
│   └── pathmap.py               # Maps paths stored in the data packages to your local copy
├── scripts/
│   ├── eval_premind.sh          # infer / judge / tables
│   ├── train_premind.sh         # phase1 / phase2
│   └── run_tests.sh
├── docs/                        # EVALUATION.md, TRAINING.md, REPRODUCE.md, BENCHMARK.md, CONSTRUCTION.md
└── imgs/
```

---

## Installation

Tested with **Python 3.12**. Exact versions are pinned in `pyproject.toml`.

```bash
git clone https://github.com/oyt9306/PReMind && cd PReMind
pip install -e ".[vllm,test]"
bash scripts/run_tests.sh        # CPU tests
```

Judging uses the OpenAI API (GPT-6-Luna). Set the key as an environment variable:
```bash
export OPENAI_API_KEY="your_key_here"
```

---

## Evaluation

Unpack the evaluation inputs (test release in the evaluation format, retrieval embeddings and judge references; download link coming soon):

```bash
tar -xf premind_eval_inputs_v1.tar
cp -r premind_eval_inputs_v1/core/. premind/core/
export PREMIND_EVAL_OUT=$PWD/runs/paper_eval
```

Then run inference, judging and metrics:

```bash
export PREMIND_4B_LORA=/path/to/PReMind-4B            # local copy of the adapter
GPU=0 scripts/eval_premind.sh infer premind4b main
GPU=0 scripts/eval_premind.sh infer premind4b conflict
scripts/eval_premind.sh judge premind4b main --execute
scripts/eval_premind.sh judge premind4b conflict --execute
scripts/eval_premind.sh tables premind4b
```

Use `base4b` instead of `premind4b` for the zero-shot base model. Populations and metric definitions are in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Training

PReMind-4B is trained from Qwen3.5-4B with LoRA in two phases. Both phases mix on-demand episodes (the model decides whether to call memory) with forced-call episodes (memory is always retrieved and only the answer is trained).

```bash
GPUS=0,1 scripts/train_premind.sh phase1     # 125 steps
GPUS=0,1 scripts/train_premind.sh phase2     # 25 steps from the Phase I adapter
```

The training data are available on request (see above). Details in [docs/TRAINING.md](docs/TRAINING.md).

---

## Using the model

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

rev = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B", revision=rev)
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", revision=rev, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "Yeongtak/PReMind-4B").eval()
```

---

## Citation

```bibtex
@misc{oh2026premind,
  title  = {PReMind: When Personal Memory Meets Visual Hazards},
  author = {Oh, Yeongtak and Lee, Donghyeok and Oh, Joonhee and Jeong, Jaewon and Yoon, Sungroh},
  year   = {2026}
}
```

## License

Code: Apache 2.0. Dataset and model: CC BY-NC 4.0 (see their cards).
