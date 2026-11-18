"""Frozen run configuration for the final training arms (T3 passive RAG, T5/T6 memory call).

PROVISIONAL (Phase C prep). Every value carries a source in ``SOURCES`` or is marked TBD there.
Values with no owner decision (steps, r) have no default: passing ``None`` raises.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
INTERFACE_DIR = V3 / "final_construction" / "train_interface_v1"

CONFIG_VERSION = "trainer_v1.config_v1"
LEGACY = "rl_sweep/train_rl.py"
LEGACY_LAUNCH = "rl_sweep/launch_tool_train.sh"

# Seeds s0-s2 (PREREG §3). Integer mapping s_k -> k follows the legacy `--seed <int>` (train_rl.py:408).
SEEDS = {"s0": 0, "s1": 1, "s2": 2}
PURPOSES = ("dry_run", "c2_smoke", "final")
REWARD_VERSIONS = {"passive_rag": "reward_v4.0-passive", "memory_call": "reward_v4.0-call"}
MIN_FINAL_STEPS = 100      # R26 minimum
C2_SMOKE_STEPS = 20        # C2 plumbing smoke definition (Phase C brief)
DRY_RUN_STEPS = 2
MAX_MODEL_LEN_ROUND = 128  # round the computed vLLM max_model_len up to a multiple of this


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the PREREG §3 matrix."""
    arm_id: str
    fp_ref: str
    interface_arm: str     # train_interface_v1 Adapter arm: passive_rag | memory_call
    retriever: str
    reward_version: str


ARMS: dict[str, ArmSpec] = {
    "new_passive_rag_training": ArmSpec(
        "T3_new_passive_rag_training", "#3", "passive_rag", "legacy_lexical_bow", REWARD_VERSIONS["passive_rag"]),
    "new_memory_call_legacy_retriever": ArmSpec(
        "T5_new_memory_call_legacy_retriever", "#5", "memory_call", "legacy_lexical_bow", REWARD_VERSIONS["memory_call"]),
    "new_memory_call_improved_retriever": ArmSpec(
        "T6_new_memory_call_improved_retriever", "#6", "memory_call", "improved_dev_selected",
        REWARD_VERSIONS["memory_call"]),
}
ARM_ALIASES = {"T3": "new_passive_rag_training", "T5": "new_memory_call_legacy_retriever",
               "T6": "new_memory_call_improved_retriever"}


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainerConfig:
    """All run settings for one (arm, seed). Build with ``build_config``; validation runs on construction."""
    arm: ArmSpec
    seed_label: str
    seed: int
    purpose: str
    steps: Optional[int]
    r: Optional[float]
    vllm_max_model_len: int
    lora: LoraSpec = field(default_factory=LoraSpec)
    model_id: str = "Qwen/Qwen3.5-4B"
    model_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    algo: str = "grpo"
    loss_type: str = "grpo"
    learning_rate: float = 1e-5
    beta: float = 0.0
    epsilon: float = 0.2
    num_generations: int = 8
    per_device_bs: int = 8
    grad_accum: int = 1
    train_ranks: int = 3
    temperature: float = 1.0
    scale_rewards: str = "group"
    max_completion_length: int = 512
    stage1_cap: int = 512
    importance_sampling_level: str = "token"
    vllm_is_mode: str = "token_truncate"
    vllm_is_cap: float = 3.0
    vllm_mode: str = "server"
    shuffle_dataset: bool = False
    enable_thinking: bool = False
    max_prompt_length: None = None   # no prompt truncation exists in TRL 1.9.2; kept None on purpose
    reward_version: str = ""

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def prompts_per_step(self) -> int:
        """P in R26: train ranks x per-device batch x grad accum / group size."""
        return self.train_ranks * self.per_device_bs * self.grad_accum // self.num_generations

    def to_json(self) -> dict:
        return {**asdict(self), "prompts_per_step": self.prompts_per_step, "config_version": CONFIG_VERSION,
                "sources": SOURCES}


SOURCES: dict[str, str] = {
    "arm": "docs/PREREG_FINAL_v1_DRAFT_2026-10-01.md §3 (T3 #3, T5 #5 primary, T6 #6 ablation)",
    "arm.retriever[T3]": "TBD: PREREG §3 says T3 is 'top-5 always, as Z2' and does not name the ranker; legacy "
                         "lexical BoW assumed; the per-row retriever_version manifest is authoritative",
    "arm.retriever[T6]": "TBD: dev-selected candidate (OD-8 gate); T6 'not run' if none passes",
    "seed": "PREREG §3 seeds s0-s2; s_k -> int k as legacy train_rl.py:408 (TBD-confirm mapping)",
    "steps": "TBD-PhaseC: R26 (docs/PREREG_FINAL_v1_TBD_RESOLUTION_DRAFT_2026-10-01.md §4); no default",
    "r": "TBD-PhaseC: DF-01 / PREREG §8; required for memory-call arms, no default",
    "reward_version": "Phase C brief names reward_v4.0-passive / reward_v4.0-call; PREREG §8 still lists TBD-B",
    "lora": f"{LEGACY}:507-508 (r=16, alpha=32, dropout=0.0, language_model q/k/v/o/gate/up/down)",
    "model_id/model_revision": "train_interface_v1/length_budget.py:28-29 (baseline_eval/config.py pin)",
    "algo": "Phase C brief: GRPO; O-9 (GSPO vs GRPO) TBD",
    "loss_type": f"{LEGACY}:506 loss_type='grpo'",
    "learning_rate": f"{LEGACY}:414 default 1e-5; O-9 TBD",
    "beta": "TRL 1.9.2 GRPOConfig.beta default 0.0 (introspected in qwen35_trl); MC prereg L352 / O-9 TBD",
    "epsilon": "TRL 1.9.2 GRPOConfig.epsilon default 0.2 (legacy did not set it)",
    "num_generations": f"{LEGACY}:412 and {LEGACY_LAUNCH}:67 (8); O-9 group size TBD",
    "per_device_bs": f"{LEGACY_LAUNCH}:12 (4b BS=8)",
    "grad_accum": f"{LEGACY}:413 default 1",
    "train_ranks": f"{LEGACY_LAUNCH}:65 (4-GPU topology: 3 train ranks + 1 server GPU)",
    "temperature": f"{LEGACY}:496",
    "scale_rewards": "TRL 1.9.2 default 'group' (legacy did not set it)",
    "max_completion_length": f"{LEGACY}:495 (512); re-check with length_budget.py on final_train_v1",
    "stage1_cap": "owner 2026-10-02: 512 new tokens (reasoning allowed before the final <memory_call> tag)",
    "importance_sampling": f"{LEGACY}:501-505 GRPO: token level, token_truncate, cap {LEGACY}:139 (3.0)",
    "vllm_mode": f"{LEGACY}:499 server mode",
    "vllm_max_model_len": "computed: length_budget worst total + margin (>= stage1_cap), rounded up; never truncates",
    "shuffle_dataset": f"{LEGACY}:498 shuffle_dataset=False",
    "enable_thinking": f"{LEGACY}:498 chat_template_kwargs enable_thinking=False",
    "max_prompt_length": "TRL 1.9.2 has no max_prompt_length; prompts are never truncated",
    "judge": "TBD: judge_fn not wired (train_interface_v1 README); ledgers complete=false until then",
    "margin_tokens": "TBD-PhaseC (proposed 640 = stage1_cap 512 + 128 template delta): Stage-1 sampling variance (<= stage1_cap) + template delta",
}


def _validate(cfg: TrainerConfig) -> None:
    if cfg.purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if cfg.seed_label not in SEEDS or SEEDS[cfg.seed_label] != cfg.seed:
        raise ValueError(f"seed must be one of {SEEDS}")
    if cfg.steps is None:
        raise ValueError("steps is TBD (R26) and has no default; pass it explicitly")
    if not isinstance(cfg.steps, int) or isinstance(cfg.steps, bool) or cfg.steps <= 0:
        raise ValueError("steps must be a positive int")
    if cfg.purpose == "final" and (cfg.steps < MIN_FINAL_STEPS or cfg.steps % 10):
        raise ValueError("R26: final N_steps must be >= 100 and a multiple of 10")
    if cfg.purpose == "c2_smoke" and cfg.steps != C2_SMOKE_STEPS:
        raise ValueError(f"C2 smoke runs exactly {C2_SMOKE_STEPS} steps")
    if cfg.purpose == "dry_run" and cfg.steps != DRY_RUN_STEPS:
        raise ValueError(f"dry run runs exactly {DRY_RUN_STEPS} steps")
    if cfg.arm.interface_arm == "memory_call":
        if cfg.r is None:
            raise ValueError("memory-call arms require r (DF-01, TBD-PhaseC); there is no default")
        if not isinstance(cfg.r, (int, float)) or isinstance(cfg.r, bool) or cfg.r <= 0:
            raise ValueError("r must be a positive number")
    elif cfg.r is not None:
        raise ValueError("the passive arm has no decision term; r must be None")
    if cfg.reward_version != REWARD_VERSIONS[cfg.arm.interface_arm]:
        raise ValueError(f"reward_version must be {REWARD_VERSIONS[cfg.arm.interface_arm]!r} for this arm")
    if cfg.max_prompt_length is not None:
        raise ValueError("prompt truncation is forbidden")
    if cfg.shuffle_dataset or cfg.enable_thinking:
        raise ValueError("shuffle_dataset and enable_thinking must stay False")
    if (cfg.train_ranks * cfg.per_device_bs * cfg.grad_accum) % cfg.num_generations or cfg.prompts_per_step <= 0:
        raise ValueError("global batch must be a positive multiple of num_generations")
    if cfg.vllm_max_model_len < cfg.max_completion_length + cfg.stage1_cap:
        raise ValueError("vllm_max_model_len is below the completion budget")


def max_model_len_from_budget(report: dict, *, max_completion_length: int, stage1_cap: int,
                              margin_tokens: Optional[int]) -> int:
    """vLLM ``max_model_len`` from a train_interface_v1 length-budget report (never a default).

    Raises:
        ValueError: if the report is not a successful measurement, was made with other limits, or the margin
            is missing / smaller than the Stage-1 cap.
    """
    if margin_tokens is None or margin_tokens < stage1_cap:
        raise ValueError("margin_tokens is required and must be >= stage1_cap")
    if report.get("status") != "measured" or report.get("ok") is not True or not report.get("worst"):
        raise ValueError("length budget must be a successful measurement (status=measured, ok=true)")
    if report["max_completion_length"] != max_completion_length or report["stage1_cap"] != stage1_cap:
        raise ValueError("length budget was measured with different completion / Stage-1 limits")
    need = int(report["worst"]["total"]) + margin_tokens
    return int(math.ceil(need / MAX_MODEL_LEN_ROUND) * MAX_MODEL_LEN_ROUND)


def resolve_arm(name: str) -> ArmSpec:
    key = ARM_ALIASES.get(name, name)
    if key not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {sorted(ARMS) + sorted(ARM_ALIASES)}")
    return ARMS[key]


def build_config(arm: str, seed_label: str, *, purpose: str, steps: Optional[int], r: Optional[float],
                 length_budget: dict, margin_tokens: Optional[int], **overrides) -> TrainerConfig:
    """Build and validate a TrainerConfig. ``steps`` and ``r`` are explicit (None raises where required)."""
    spec = resolve_arm(arm)
    if seed_label not in SEEDS:
        raise ValueError(f"seed must be one of {sorted(SEEDS)}")
    mcl = overrides.get("max_completion_length", TrainerConfig.max_completion_length)
    cap = overrides.get("stage1_cap", TrainerConfig.stage1_cap)
    mml = max_model_len_from_budget(length_budget, max_completion_length=mcl, stage1_cap=cap,
                                    margin_tokens=margin_tokens)
    return TrainerConfig(arm=spec, seed_label=seed_label, seed=SEEDS[seed_label], purpose=purpose, steps=steps,
                         r=r, vllm_max_model_len=mml, reward_version=spec.reward_version, **overrides)


def load_length_budget(path: Path) -> dict:
    return json.loads(Path(path).read_text())
