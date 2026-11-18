"""DAPO pieces and self-teacher OPSD modes for grpo_v2.train_step (defaults leave the GRPO / frozen-teacher loop
bit-identical: nothing here runs unless ``algo == "dapo"`` or ``opsd_teacher == "self"`` / ``opsd_gate == "hint"``).

DAPO (Yu et al. 2025, arXiv 2503.14476), algo="dapo":
  * clip-higher: decoupled bounds 1 - 0.2 / 1 + 0.28 (policy_loss.DEFAULT_EPS; --clip-eps-low/high override);
  * dynamic sampling: groups whose G (shaped) rewards are all equal carry zero advantage; they are dropped and new
    prompts are drawn (``refill``; one batched rollout per round) until P informative groups are kept or
    ceil(dapo_max_prompts_factor * P) prompts were drawn in the step. Each round draws ceil(missing / observed
    informative rate) prompts (rate floored at 0.25; capped by the budget), so few rounds are needed; informative
    groups beyond P are discarded (draw order kept). At the cap the step trains on the groups it has; if none is
    informative it trains on the first round's groups (all advantages 0: only an OPSD term can move the policy) and
    flags ``dapo_no_informative``. Advantages are computed on the kept groups only;
  * token-level loss: the RL term of each episode (a token mean) is weighted n * T_k / sum_j T_j so that after the
    usual / n the step loss is sum over all model tokens / total model tokens (the OPSD KL keeps its per-episode
    1 / n weight, so lambda means the same in every arm);
  * overlong reward shaping (soft): with cap L = max_completion_length and buffer B, a Stage-2 / passive generation
    of length l (tokens incl. EOS) gets min(0, (L - B - l) / B) * overlong_penalty added to its reward for the
    advantage (raw reward_v5 totals stay in the logs; the shaping is logged separately). Stage 1 (a short decision
    turn) is not shaped;
  * no reference KL (none in this loop).

Self-teacher OPSD (opsd_teacher="self", as in OPSD / pi-Distill): the teacher is the CURRENT LoRA model (adapter on)
with the privileged (hinted) context, under no_grad (stop-gradient); the same top-k truncated KL on the student's own
rollout tokens; recomputed at every update when mu > 1 (the teacher moves with the policy). opsd_gate:
  * "table" (default for the frozen teacher): the precomputed teacher_ok table (it scores the FROZEN base, so it does
    not describe a self teacher);
  * "hint" (default for the self teacher): every episode whose item has a clean, leak-checked hint (same
    construction, stage scoping and leak check as the table builder) and at least one distillable token (Stage-2 /
    passive); ``opsd_only_failed`` restricts it to episodes failing a Stage-2 reward check.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

from . import opsd as OP

logger = logging.getLogger(__name__)
TEACHERS = ("frozen", "self")
GATES = ("table", "hint")
MIN_RATE = 0.25
Refill = Callable[[int], tuple[list, list]]


def gate_mode(cfg: Any) -> str:
    """Resolved OPSD gate: explicit ``opsd_gate``, else "table" (frozen teacher) / "hint" (self teacher)."""
    return cfg.opsd_gate or ("hint" if cfg.opsd_teacher == "self" else "table")


def validate(cfg: Any) -> None:
    if cfg.opsd_teacher not in TEACHERS or (cfg.opsd_gate is not None and cfg.opsd_gate not in GATES):
        raise ValueError(f"opsd_teacher must be in {TEACHERS} and opsd_gate in {GATES} (or None)")
    if cfg.dapo_max_prompts_factor < 1 or cfg.overlong_buffer < 0 or cfg.overlong_penalty < 0:
        raise ValueError("dapo_max_prompts_factor >= 1, overlong_buffer >= 0 and overlong_penalty >= 0 required")


def overlong_penalty(n_tokens: int, cap: int, buffer: int, factor: float) -> float:
    """DAPO soft overlong punishment: 0 up to cap - buffer, then linear down to -factor at the cap (and below)."""
    if buffer <= 0 or factor <= 0 or n_tokens <= cap - buffer:
        return 0.0
    return max(-1.0, (cap - buffer - n_tokens) / buffer) * factor


def stage2_len(ep: Any) -> int:
    """Generated Stage-2 (memory call) / passive tokens, EOS included."""
    return len(ep.completion) - ep.n_stage1 - ep.n_env


def shaped_rewards(eps: list, cfg: Any) -> list[float]:
    """Rewards used for the advantages: reward_v5 total (+ overlong shaping for DAPO)."""
    if cfg.algo != "dapo":
        return [ep.ledger["total"] for ep in eps]
    return [ep.ledger["total"] + overlong_penalty(stage2_len(ep), cfg.max_completion_length, cfg.overlong_buffer,
                                                  cfg.overlong_penalty) for ep in eps]


def informative(rewards: list[float]) -> bool:
    """A group has a non-zero advantage somewhere iff its rewards are not all equal."""
    return any(r != rewards[0] for r in rewards[1:])


def rl_weights(eps: list, cfg: Any, totals: Optional[tuple[int, int]] = None) -> Optional[list[float]]:
    """DAPO token-level aggregation as per-episode weights on the RL term (None = per-sequence mean, GRPO / GSPO).
    ``totals`` = (episodes, model tokens) of the GLOBAL step under data parallelism (ddp_v2); None = this process's
    episodes (single process, unchanged)."""
    if cfg.algo != "dapo":
        return None
    toks = [sum(ep.env_mask) for ep in eps]
    n, total = (len(eps), sum(toks)) if totals is None else totals
    return [n * t / total if total else 0.0 for t in toks]


def dynamic_rollout(items: list, images: list, backend: Any, cfg: Any, refill: Optional[Refill],
                    rollout: Callable[..., list], score: Callable[[Any, Any], dict],
                    prefetch: Optional[Callable[[list], None]] = None) -> tuple[list, list, list, dict]:
    """DAPO dynamic sampling: (kept episodes, their items, their images, info). Episodes are scored here.
    ``prefetch`` (VEF presets): called on each round's episodes before they are scored (judge replies)."""
    g, p = cfg.num_generations, len(items)
    budget = math.ceil(cfg.dapo_max_prompts_factor * p)
    kept: list[tuple[Any, Any, list]] = []
    first: list[tuple[Any, Any, list]] = []
    drawn, rounds, filtered, seen = p, 0, 0, 0
    batch_items, batch_images = items, images
    while True:
        eps = rollout(batch_items, batch_images, backend, cfg, batched=cfg.batch_rollout)
        if prefetch is not None:
            prefetch(eps)
        for ep in eps:
            ep.ledger = score(ep, cfg)
        shaped = shaped_rewards(eps, cfg)
        for j, (it, im) in enumerate(zip(batch_items, batch_images, strict=True)):
            grp = (it, im, eps[j * g:(j + 1) * g])
            first += [grp] if rounds == 0 else []
            seen += 1
            if informative(shaped[j * g:(j + 1) * g]):
                kept.append(grp)
            else:
                filtered += 1
        need = p - len(kept)
        if need <= 0 or refill is None or drawn >= budget:
            break
        rate = max(len(kept) / seen, MIN_RATE)
        k = min(budget - drawn, math.ceil(need / rate))
        batch_items, batch_images = refill(k)
        drawn, rounds = drawn + k, rounds + 1
    info = {"dapo_filtered_groups": filtered, "dapo_resample_rounds": rounds, "dapo_prompts_drawn": drawn,
            "dapo_kept_groups": min(len(kept), p), "dapo_no_informative": not kept}
    use = kept[:p] if kept else first
    logger.info("DAPO dynamic sampling: %s", info)
    return ([ep for _, _, grp in use for ep in grp], [it for it, _, _ in use], [im for _, im, _ in use], info)


def opsd_gate(opsd: Any, ep: Any, cfg: Any, distill_positions: Callable[[Any], list]) -> tuple[bool, list[str]]:
    """Gate of one scored episode: the table gate (OpsdContext.gate, unchanged) or the hint gate."""
    if not opsd or cfg.opsd_lambda <= 0:
        return False, []
    if gate_mode(cfg) == "table":
        return opsd.gate(ep.item, ep.ledger)
    if OP.build_hint(ep.item, opsd.hint_ids)[0] is None:
        return False, ["no_clean_hint"]
    if not distill_positions(ep):
        return False, ["no_distill_tokens"]
    why = OP.failure_reasons(ep.ledger, stage2_only=True)
    if cfg.opsd_only_failed and not why:
        return False, ["passes"]
    return True, ["hint"] + why
