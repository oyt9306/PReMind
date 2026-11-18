# Changelog

## v1.0 — 2026-11-18

First public release of PReMind-Bench and the PReMind code, matching the paper "PReMind: When Personal Memory Meets
Visual Hazards".

- **Data:** public test split (1,020 main requests + 112 test-only rows; 140 scored interdependency cases), online
  top-4 retrieval. Train/dev available on request.
- **Model:** PReMind-4B (Qwen3.5-4B LoRA; Phase I 125 + Phase II 25 steps, seed 0).
- **Code:** two-phase training (on-demand + forced-call episodes), paper evaluation with GPT-6-Luna judging, and metric
  recomputation from stored outputs; benchmark construction pipeline.
- Earlier internal releases (K = 8 retrieval, rule-only scoring, earlier train/test splits) are superseded; their
  results are not comparable.
