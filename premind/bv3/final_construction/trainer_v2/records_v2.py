"""Logging-only pieces of grpo_v2 (moved here unchanged to keep grpo_v2 under 400 lines; grpo_v2 re-exports them):
per-token diagnostics, truncation flag, reward component means and the per-episode training record."""
from __future__ import annotations

from typing import Any, Optional

from . import opsd as OP
from . import policy_loss as PL
from . import reward_v5 as R5


def token_diagnostics(logp_all: Any, coef1: Any, is_ratio: Any, cfg: Any, chunk: int = 256) -> dict:
    """Logging only (wandb), no grad, no effect on the loss: mean token entropy (nats) of the policy at the model
    positions, fraction of tokens whose PPO ratio is outside the clip band (policy_loss.clip_bounds; GRPO default
    [1 - eps, 1 + eps]), fraction whose IS ratio hit the cap. GSPO overrides the last two at the sequence level."""
    import torch
    lo, hi = PL.clip_bounds(cfg)
    if logp_all.shape[0] == 0:
        return {"entropy": None, "clip_frac": None, "is_cap_frac": None}
    with torch.no_grad():
        ent = sum(float(-torch.where(c.isfinite(), c.exp() * c, torch.zeros_like(c)).sum())
                  for c in logp_all.detach().split(chunk))
        r = coef1.detach()
        return {"entropy": ent / logp_all.shape[0],
                "clip_frac": float(((r < lo) | (r > hi)).float().mean()),
                "is_cap_frac": float((is_ratio.detach() >= cfg.is_cap).float().mean())}


def truncated(ep: Any, cfg: Any, eos: int) -> bool:
    """Logging only: a generation turn hit its token cap without EOS (Stage 1 at ``stage1_cap``, Stage 2 / passive at
    ``max_completion_length``)."""
    c1, c2 = ep.completion[:ep.n_stage1], ep.completion[ep.n_stage1 + ep.n_env:]
    return (len(c2) >= cfg.max_completion_length and c2[-1] != eos) or (
        len(c1) >= cfg.stage1_cap and c1[-1] != eos)


def component_means(ledgers: list[dict]) -> dict[str, Optional[float]]:
    """Mean value of each reward component over the episodes where it applied (None if it never applied)."""
    out = {}
    vef = ("vef",) if any("vef" in led["components"] for led in ledgers) else ()      # VEF presets only
    for name in R5.COMPONENTS + vef:
        vals = [led["components"][name]["value"] for led in ledgers
                if led["components"][name]["applicable"] and led["components"][name]["value"] is not None]
        out[name] = round(sum(vals) / len(vals), 6) if vals else None
    return out


def episode_record(ep: Any, a: float, gated: bool, why: list, st: dict, cfg: Any, eos: int) -> dict:
    """Per-episode training record (wandb / step log) from the loss stats ``st``."""
    return {"row_id": ep.item.row_id, "arm": ep.item.arm, "reward": ep.ledger["total"], "advantage": a,
            "opsd_gate": gated, "gate_reasons": why, "n_completion": len(ep.completion),
            "n_prompt": len(ep.prompt.ids), "decision": ep.decision, **st, "truncated": truncated(ep, cfg, eos),
            "success": not OP.failure_reasons(ep.ledger),
            "action_pred": (ep.ledger.get("parse") or {}).get("action_pred")}
