"""Rollouts for the final arms: passive single turn (T3) and two-turn memory call (T5/T6).

Token-in/token-out: Stage 2 is generated from ``prompt_ids + stage1_ids + env_ids`` exactly as the trainer later
scores it, so the trainer forward pass and the sampler see the same ids (logp/IS parity across turns).

Loss mask (``env_mask``): 1 for every model-generated token of Stage 1 and Stage 2, 0 for every environment-injected
token (end of the Stage-1 turn if the model did not emit it, the Stage-2 user turn, the next assistant header).
Backends follow TRL ``VLLMGeneration.generate`` semantics: the input list already holds each prompt repeated
``num_generations`` times contiguously, and one completion is returned per input entry.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from .config import INTERFACE_DIR, TrainerConfig

if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))
import adapter as AD  # noqa: E402  (frozen train_interface_v1, import only)
import render as RD  # noqa: E402

ROLLOUT_VERSION = "trainer_v1.rollout_v1"


class PromptTooLongError(ValueError):
    """A sequence would exceed vLLM max_model_len. Fail closed: prompts are never truncated."""


class Backend(Protocol):
    eos_token_id: int

    def encode_prompt(self, messages: list[dict], image: Any) -> list[int]: ...

    def encode_env_turn(self, env_user_message: dict, *, stage1_ended_with_eos: bool) -> list[int]: ...

    def generate(self, prompts: list[list[int]], images: list, *, num_generations: int,
                 max_tokens: int) -> tuple[list[list[int]], list[list[float]]]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class Rollout:
    """One episode. ``completion_ids`` = Stage-1 + env + Stage-2 (call arm) or the single completion (passive)."""
    row_id: str
    arm: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    env_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    stage1_text: Optional[str]
    stage2_text: str
    decision: Optional[str]
    called: Optional[bool]
    n_stage1: int
    n_env: int


def _fits(n_prompt: int, max_tokens: int, cfg: TrainerConfig, what: str) -> None:
    if n_prompt + max_tokens > cfg.vllm_max_model_len:
        raise PromptTooLongError(f"{what}: {n_prompt} prompt + {max_tokens} new > max_model_len "
                                 f"{cfg.vllm_max_model_len}; refusing to truncate")


def assert_stage1_isolated(item: AD.TrainItem, prompt_text: str) -> None:
    """Stage 1 must carry no personal context: no rendered header, no entry line, no record text."""
    if item.arm != "memory_call":
        raise ValueError("Stage-1 isolation applies to memory-call items")
    rebuilt = AD.stage1_item(item.row).visible["prompt"]
    if item.visible["prompt"] != rebuilt:
        raise AssertionError(f"{item.row_id}: Stage-1 prompt differs from the interface builder")
    if "Retrieved personal context" in prompt_text or RD.count_rendered_entries(prompt_text):
        raise AssertionError(f"{item.row_id}: Stage-1 prompt carries a context block")
    for rec in item.row["records"]:
        if rec["text"] in prompt_text:
            raise AssertionError(f"{item.row_id}: Stage-1 prompt carries record text")


def _strip_eos(ids: Sequence[int], eos: int) -> list[int]:
    return list(ids[:-1]) if ids and ids[-1] == eos else list(ids)


def passive_rollout(items: list[AD.TrainItem], images: list, backend: Backend,
                    cfg: TrainerConfig) -> list[Rollout]:
    """T3: one turn from image + query + exact-five context; no Stage 1, every completion token weight 1."""
    if any(it.arm != "passive_rag" for it in items):
        raise ValueError("passive_rollout needs passive_rag items")
    prompts = []
    for it, img in zip(items, images, strict=True):
        if RD.count_rendered_entries(AD.visible_texts(it.visible["prompt"])[-1]) != RD.RENDER_K:
            raise AssertionError(f"{it.row_id}: passive prompt must render exactly {RD.RENDER_K} entries")
        ids = backend.encode_prompt(it.visible["prompt"], img)
        _fits(len(ids), cfg.max_completion_length, cfg, f"passive {it.row_id}")
        prompts.append(ids)
    outs, lps = backend.generate(prompts, images, num_generations=cfg.num_generations,
                                 max_tokens=cfg.max_completion_length)
    rollouts = []
    for it, p, c, lp in zip(items, prompts, outs, lps, strict=True):
        rollouts.append(Rollout(it.row_id, "passive_rag", tuple(p), tuple(c), (1,) * len(c), tuple(lp), None,
                                backend.decode(_strip_eos(c, backend.eos_token_id)), None, None, 0, 0))
    return rollouts


def two_turn_rollout(items: list[AD.TrainItem], images: list, backend: Backend,
                     cfg: TrainerConfig) -> list[Rollout]:
    """T5/T6: Stage 1 (image + query) -> parse -> environment turn -> Stage 2.

    A malformed Stage-1 output is ``decision_invalid``: the rollout continues in the not-called branch (MC prereg
    §2.3 format path) and the reward bridge scores the decision as wrong. Context is injected only after a parsed
    ``call_pdb``.
    """
    if any(it.arm != "memory_call" for it in items):
        raise ValueError("two_turn_rollout needs memory_call items")
    eos = backend.eos_token_id
    p1 = []
    for it, img in zip(items, images, strict=True):
        ids = backend.encode_prompt(it.visible["prompt"], img)
        assert_stage1_isolated(it, backend.decode(ids))
        _fits(len(ids), cfg.stage1_cap, cfg, f"stage1 {it.row_id}")
        p1.append(ids)
    s1, lp1 = backend.generate(p1, images, num_generations=cfg.num_generations, max_tokens=cfg.stage1_cap)
    if any(len(x) > cfg.stage1_cap for x in s1):
        raise AssertionError("Stage-1 output exceeds stage1_cap")

    p2, meta = [], []
    for it, prompt, out in zip(items, p1, s1, strict=True):
        text1 = backend.decode(_strip_eos(out, eos))
        messages, decision, called = AD.stage2_messages(it.row, text1)
        env = backend.encode_env_turn(messages[-1], stage1_ended_with_eos=bool(out) and out[-1] == eos)
        ids2 = list(prompt) + list(out) + list(env)
        _fits(len(ids2), cfg.max_completion_length, cfg, f"stage2 {it.row_id}")
        p2.append(ids2)
        meta.append((text1, decision, called, env))
    s2, lp2 = backend.generate(p2, images, num_generations=1, max_tokens=cfg.max_completion_length)

    rollouts = []
    for it, prompt, out1, l1, (text1, decision, called, env), out2, l2 in zip(
            items, p1, s1, lp1, meta, s2, lp2, strict=True):
        completion = tuple(out1) + tuple(env) + tuple(out2)
        mask = (1,) * len(out1) + (0,) * len(env) + (1,) * len(out2)
        logps = tuple(l1) + (0.0,) * len(env) + tuple(l2)
        rollouts.append(Rollout(it.row_id, "memory_call", tuple(prompt), completion, mask, logps, text1,
                                backend.decode(_strip_eos(out2, eos)), decision, called, len(out1), len(env)))
    return rollouts


def run_rollout(items: list[AD.TrainItem], images: list, backend: Backend, cfg: TrainerConfig) -> list[Rollout]:
    fn = two_turn_rollout if cfg.arm.interface_arm == "memory_call" else passive_rollout
    return fn(items, images, backend, cfg)
