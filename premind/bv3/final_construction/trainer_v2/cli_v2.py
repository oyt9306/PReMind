"""CLI / logging helpers of trainer_v2.run_v2 (also used by eval_v1.main_run): wandb helpers (moved from run_v2
unchanged; run_v2 re-exports them) and the DAPO / OPSD-teacher argument groups (defaults = the GRPO, frozen-teacher
loop; LoopConfig fields documented in grpo_v2 / algo_v2)."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def flat_scalars(rec: Any, prefix: str = "") -> dict:
    """Numeric leaves of a step record as ``a/b/c`` keys (lists and strings dropped) for wandb."""
    if isinstance(rec, bool):
        return {prefix: int(rec)}
    if isinstance(rec, (int, float)):
        return {prefix: rec}
    if isinstance(rec, dict):
        out: dict = {}
        for k, v in rec.items():
            out.update(flat_scalars(v, f"{prefix}/{k}" if prefix else str(k)))
        return out
    return {}


def wandb_init(args: argparse.Namespace, config: dict, out: Path) -> Any:  # pragma: no cover - network
    """wandb run for a GPU job, or None (``--no-wandb`` or init failure: training never dies for logging)."""
    if args.no_wandb:
        return None
    try:
        import wandb
        os.environ.setdefault("WANDB_DIR", str(out))
        return wandb.init(project=args.wandb_project, name=args.wandb_name or f"{args.cmd}_{out.name}",
                          group=args.wandb_group, config={**config, "args": vars(args)}, dir=str(out))
    except Exception as exc:  # noqa: BLE001 - logging is optional
        logger.warning("wandb disabled: %s: %s", type(exc).__name__, exc)
        return None


def dapo_kwargs(args: argparse.Namespace) -> dict:
    """LoopConfig DAPO fields from CLI args; absent args = defaults (only used when --algo dapo)."""
    return {"dapo_dynamic_sampling": not getattr(args, "no_dynamic_sampling", False),
            "dapo_max_prompts_factor": getattr(args, "dapo_max_prompts_factor", 3.0),
            "overlong_buffer": getattr(args, "overlong_buffer", 128),
            "overlong_penalty": getattr(args, "overlong_penalty", 1.0)}


def add_dapo_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--no-dynamic-sampling", action="store_true", help="dapo: keep zero-variance groups")
    ap.add_argument("--dapo-max-prompts-factor", type=float, default=3.0,
                    help="dapo: at most ceil(factor * P) prompts drawn per step (dynamic sampling cap)")
    ap.add_argument("--overlong-buffer", type=int, default=128, help="dapo: soft overlong buffer (tokens)")
    ap.add_argument("--overlong-penalty", type=float, default=1.0, help="dapo: overlong shaping at the cap (0 = off)")


def opsd_mode_kwargs(args: argparse.Namespace) -> dict:
    """LoopConfig OPSD teacher / gate fields from CLI args; absent args = frozen teacher, table gate."""
    return {"opsd_teacher": getattr(args, "opsd_teacher", "frozen"), "opsd_gate": getattr(args, "opsd_gate", None),
            "opsd_only_failed": bool(getattr(args, "opsd_only_failed", False))}


def add_opsd_mode_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--opsd-teacher", default="frozen", choices=["frozen", "self"],
                    help="frozen: base model (LoRA off); self: current LoRA model with the hinted context, no grad")
    ap.add_argument("--opsd-gate", default=None, choices=["table", "hint"],
                    help="default: table (frozen teacher) / hint (self teacher)")
    ap.add_argument("--opsd-only-failed", action="store_true", help="hint gate: only episodes failing a check")


def resolved_gate(args: argparse.Namespace) -> str:
    return getattr(args, "opsd_gate", None) or ("hint" if getattr(args, "opsd_teacher", "frozen") == "self"
                                                else "table")
