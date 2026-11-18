"""Frozen configuration for the Qwen3.5 baseline POC. Everything that must stay fixed across the
eight baselines lives here; only `SETTINGS` varies capability flags."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
SPLITS_ROOT = BENCH_ROOT / "final_splits_v4"  # 2026-09-25 query pool v2 + risk_level (ids preserved from v3)
CONTEXT_ROOT = BENCH_ROOT / "context_generation"
DRAFTS_DIR = CONTEXT_ROOT / "final" / "eval"  # 2026-09-25: finalized GT (Stage-1 v10 + Stage-2 accepted); old unsuffixed files are the superseded run
GT_VARIANT = "v11"  # 2026-09-25: query pool v2 + risk_level, rule fixes + local Qwen check (v10 = Stage-2 reviewed)
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
# 2026-09-25 RL sweep: evaluate other id sets (e.g. the train-internal validation slice) with the same harness
EVAL_SET_PATH = Path(os.environ.get("EVAL_SET", OUT_ROOT / "eval_set.jsonl"))
RETRIEVAL_PATH = Path(os.environ.get("RETRIEVAL_SET", OUT_ROOT / "retrieval_top5.jsonl"))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    revision: str


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3.5-9b": ModelSpec("qwen3.5-9b", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}


@dataclass(frozen=True)
class Setting:
    name: str
    rag: bool
    tool: bool


SETTINGS: dict[str, Setting] = {
    "direct": Setting("direct", rag=False, tool=False),
    "tool_only": Setting("tool_only", rag=False, tool=True),
    "rag_only": Setting("rag_only", rag=True, tool=False),
    "rag_tool": Setting("rag_tool", rag=True, tool=True),
}


@dataclass(frozen=True)
class InferenceConfig:
    seed: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 1024
    max_model_len: int = 32768
    enable_thinking: bool = False
    # Same pixel budget for the query image and every crop (Qwen3.5 processor min/max pixels).
    min_pixels: int = 256 * 32 * 32
    max_pixels: int = 1024 * 32 * 32
    tool_budget: int = 3  # max crop calls per sample; the next turn is forced to answer
    redundant_iou: float = 0.9  # a crop with IoU >= this vs an earlier crop is "redundant"
    top_k: int = 5
    # env override (2026-09-25): GPUs 4-7 are shared with other users; value is recorded in run_meta via asdict(INFER)
    gpu_memory_utilization: float = float(os.environ.get("BASELINE_GPU_UTIL", "0.85"))
    backend: str = "vllm==0.18.0, bf16"


INFER = InferenceConfig()
RETRIEVER_NAME = "lexical_bow_cosine_v1+scene_caption"  # stage1 scorer; query = query_text + caption_scenes.py caption
INDEX_VERSION = "per_user_store_v11final_2026-09-25"
