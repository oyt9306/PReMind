"""wandb run plumbing for eval_v1.main_run: hyperparameter config, group/tags, aligned ``train/step`` axes, logging
that never raises, and resume (same wandb run id; accumulated reward / best-val state replayed from the run dir).
Metric definitions live in eval_v1.wandb_layout."""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from eval_v1 import wandb_layout as WL
from trainer_v2 import paths_v2 as P

logger = logging.getLogger(__name__)
RUN_ID_FILE = "wandb_run_id.txt"
NAMESPACES = ("train/*", "val/*", "sys/*", "stop/*")


def arm_id(args: argparse.Namespace) -> str:
    """Sweep arm id (rlvr_main.sh ARM_ID, passed as --wandb-group), else a descriptive fallback."""
    return os.environ.get("ARM_ID") or args.wandb_group or f"{args.arm}_{args.reward_preset}"


def pin_info(prereg: Optional[str]) -> dict:
    if not prereg:
        return {"prereg": None, "prereg_sha256": None, "data_sha256": None}
    try:
        pre = json.loads(Path(prereg).read_text())
        return {"prereg": str(prereg), "prereg_sha256": P.sha256_file(Path(prereg)),
                "data_sha256": (pre.get("data") or {}).get("sha256"), "prereg_status": pre.get("status")}
    except (OSError, ValueError) as exc:
        logger.warning("wandb config: cannot read prereg %s: %s", prereg, exc)
        return {"prereg": str(prereg), "prereg_sha256": None, "data_sha256": None}


def run_config(args: argparse.Namespace, cfg: Any, manifest: Any, start: int) -> dict:
    return {"arm_id": arm_id(args), "arm": args.arm, "preset": args.reward_preset, "r": args.r, "opsd": args.opsd,
            "opsd_lambda": cfg.opsd_lambda, "hint_set": args.hint_set if args.opsd else None,
            "teacher_table": args.teacher_table, "lr": cfg.learning_rate, "P": args.prompts_per_step,
            "G": args.num_generations, "steps": args.steps, "seed": args.seed, "model": args.model,
            "model_pin": list(P.MODELS[args.model]), "release": args.release, "mode": args.mode,
            **pin_info(args.prereg), "dev_every": args.dev_every, "dev_limit": args.dev_limit,
            "kret_dev": args.kret_dev, "start_step": start, "loop": dict(cfg.__dict__), "data": manifest,
            "ema_alpha": WL.EMA_ALPHA, "success_def": WL.SUCCESS_DEF, "cap_def": WL.CAP_DEF,
            "args": {k: v for k, v in vars(args).items()}}


def tags(args: argparse.Namespace) -> list[str]:
    out = [arm_id(args), f"arm:{args.arm}", f"preset:{args.reward_preset}", f"model:{args.model}",
           f"mode:{args.mode}", f"seed:{args.seed}", f"r:{args.r}", f"P{args.prompts_per_step}G{args.num_generations}"]
    if args.opsd:
        out += ["opsd", f"lambda:{args.opsd_lambda}", f"hints:{args.hint_set}"]
        if getattr(args, "opsd_teacher", "frozen") != "frozen" or getattr(args, "opsd_gate", None):
            out += [f"opsd_teacher:{getattr(args, 'opsd_teacher', 'frozen')}",
                    f"opsd_gate:{getattr(args, 'opsd_gate', None) or 'default'}"]
    if getattr(args, "algo", "grpo") == "dapo":
        out += ["algo:dapo"]
    out += [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t]
    return [t[:64] for t in dict.fromkeys(out)]          # wandb caps tag length at 64


def run_id(out: Path, resume: bool) -> str:
    """Stable wandb id per output dir: reused on --resume (one continuous run), fresh otherwise."""
    f = out / RUN_ID_FILE
    if resume and f.is_file() and f.read_text().strip():
        return f.read_text().strip()
    rid = uuid.uuid4().hex[:16]
    f.write_text(rid + "\n")
    return rid


def define_axes(wb: Any) -> None:
    wb.define_metric("train/step")
    for ns in NAMESPACES:
        wb.define_metric(ns, step_metric="train/step")


def init(args: argparse.Namespace, config: dict, out: Path) -> Any:  # pragma: no cover - network
    """wandb run or None (--no-wandb or any init failure: training never dies for logging)."""
    if args.no_wandb:
        return None
    try:
        import wandb
        os.environ.setdefault("WANDB_DIR", str(out))
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name or f"{arm_id(args)}_{out.name}",
                        group=arm_id(args), tags=tags(args), config=config, dir=str(out),
                        id=run_id(out, args.resume), resume="allow")
        define_axes(wb)
        return wb
    except Exception as exc:  # noqa: BLE001 - logging is optional
        logger.warning("wandb disabled: %s: %s", type(exc).__name__, exc)
        return None


def safe_log(wb: Any, payload: dict) -> bool:
    if wb is None:
        return False
    try:
        wb.log(payload)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb log failed: %s: %s", type(exc).__name__, exc)
        return False


def safe_summary(wb: Any, values: dict) -> None:
    if wb is None:
        return
    try:
        wb.summary.update(values)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb summary failed: %s: %s", type(exc).__name__, exc)


def safe_payload(fn: Any, *a: Any, **kw: Any) -> dict:
    """Run a payload builder; on any error log a warning and return {} (never kills training)."""
    try:
        return fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb payload %s failed: %s: %s", getattr(fn, "__name__", fn), type(exc).__name__, exc)
        return {}


def dev_val_payload(recs: list[dict], step: int, kret_recs: Optional[list[dict]] = None) -> dict:
    from eval_v1 import analysis as A
    kret = None
    if kret_recs:
        from eval_v1 import kret_stats as KS
        kret = KS.summary(kret_recs)
    return WL.val_payload(A.summary(recs), recs, step=step, kret=kret)


def read_records(path: Path) -> list[dict]:
    P.refuse_holdout_path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def replay(out: Path, start: int, acc: WL.Accumulator, best: WL.BestTracker) -> None:
    """Resume: rebuild the accumulated reward (steps.jsonl, optimizer steps < ``start``, last record per step) and the
    best-val tracker (dev_eval/step_* with step <= ``start``) so summaries and EMA continue across restarts."""
    by_step: dict[int, dict] = {}
    for rec in read_records(out / "steps.jsonl"):
        if isinstance(rec.get("step"), int) and rec["step"] < start:
            by_step[rec["step"]] = rec
    for s in sorted(by_step):
        if by_step[s].get("reward_mean") is not None:
            acc.update(float(by_step[s]["reward_mean"]))
    for d in sorted((out / "dev_eval").glob("step_*")):
        step = int(d.name.removeprefix("step_"))
        recs = read_records(d / "records.jsonl")
        if step <= start and recs:
            best.update(dev_val_payload(recs, step))
