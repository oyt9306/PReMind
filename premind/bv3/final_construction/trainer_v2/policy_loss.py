"""Policy-gradient surrogate of one episode for trainer_v2.grpo_v2: GRPO (token-level PPO ratio) or GSPO (sequence-level
ratio, Zheng et al. 2025, "Group Sequence Policy Optimization", arXiv 2507.18071).

Both take the per-token log-probs ``logp`` (with grad) of the MODEL-generated tokens only (grpo_v2.model_positions:
env_mask == 1, i.e. passive completion, or memory-call Stage-1 call turn + Stage-2 answer; env tokens excluded),
``old`` = log pi_old at the same tokens (detached; fixed for all ``updates_per_batch`` updates on one rollout batch),
``is_ratio`` = the per-token sampler-mismatch IS weight exp(old - sampler_logp) capped at ``is_cap`` (unchanged
from the GRPO loop), and the group-normalised advantage ``adv`` of the episode.

GRPO: mean_t[-min(r_t A, clip(r_t, 1-eps_lo, 1+eps_hi) A) * is_t], r_t = exp(logp_t - old_t).
GSPO: -min(s A, clip(s, 1-eps_lo, 1+eps_hi) A) * is_seq, s = exp(mean_t(logp_t - old_t)) (length-normalised sequence
ratio); is_seq = exp(mean_t(old_t - samp_t)) capped at ``is_cap`` (the sequence-level analogue of the token IS weight).
The step loss is the mean of the episode losses over the step's sequences (grpo_v2.train_step divides by N).
DAPO: the GRPO token surrogate with clip-higher bounds (1 - 0.2, 1 + 0.28); its token-level aggregation (mean over all
model tokens of the step instead of mean of per-sequence means) is applied by grpo_v2 through per-episode weights
(algo_v2.rl_weights); dynamic sampling and overlong shaping live in algo_v2.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
ALGOS = ("grpo", "gspo", "dapo")
# grpo: None -> LoopConfig.epsilon on both sides; dapo: clip-higher (Yu et al. 2025, arXiv 2503.14476) eps 0.2 / 0.28
DEFAULT_EPS = {"grpo": None, "gspo": (3e-4, 4e-4), "dapo": (0.2, 0.28)}


def clip_bounds(cfg: Any) -> tuple[float, float]:
    """(lower, upper) clip bounds of the importance ratio. GRPO defaults to 1 -/+ ``cfg.epsilon`` (bit-identical to
    the pre-GSPO loop); GSPO defaults to the paper's 3e-4 / 4e-4, DAPO to 0.2 / 0.28. ``clip_eps_low`` /
    ``clip_eps_high`` override."""
    algo = getattr(cfg, "algo", "grpo")
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}, got {algo!r}")
    lo, hi = DEFAULT_EPS[algo] or (cfg.epsilon, cfg.epsilon)
    lo = lo if getattr(cfg, "clip_eps_low", None) is None else cfg.clip_eps_low
    hi = hi if getattr(cfg, "clip_eps_high", None) is None else cfg.clip_eps_high
    return 1 - lo, 1 + hi


def grpo_term(logp: Any, old: Any, is_ratio: Any, adv: Any, cfg: Any) -> tuple[Any, Any]:
    """Token-level PPO-clip surrogate (op-for-op the pre-GSPO grpo_v2.episode_loss). Returns (loss, token ratios)."""
    import torch
    lo, hi = clip_bounds(cfg)
    coef1 = torch.exp(logp - old)
    coef2 = coef1.clamp(lo, hi)
    return (-torch.min(coef1 * adv, coef2 * adv) * is_ratio).mean(), coef1


def gspo_term(logp: Any, old: Any, samp: Any, adv: Any, cfg: Any) -> tuple[Any, Any, Any]:
    """Sequence-level clipped surrogate. ``samp`` = sampler log-probs (NaN already replaced by ``old``).
    Returns (loss, sequence ratio s (0-d, with grad), sequence IS weight (0-d, detached))."""
    import torch
    lo, hi = clip_bounds(cfg)
    s = torch.exp((logp - old).mean())
    is_seq = torch.exp((old - samp).mean()).clamp(max=cfg.is_cap).detach()
    return -torch.min(s * adv, s.clamp(lo, hi) * adv) * is_seq, s, is_seq


def sequence_diagnostics(s: Any, is_seq: Any, cfg: Any) -> dict:
    """GSPO logging (no grad): ratio, clipped flag (0/1: s outside the clip band) and IS-cap flag of one sequence;
    the step means over episodes (eval_v1.wandb_layout) are the sequence-level clip_frac / is_cap_frac."""
    lo, hi = clip_bounds(cfg)
    r = float(s.detach())
    return {"ratio_mean": r, "clip_frac": float(r < lo or r > hi), "is_ratio_mean": float(is_seq),
            "is_cap_frac": float(float(is_seq) >= cfg.is_cap)}
