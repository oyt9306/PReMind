"""Over-abstain hard-stop decision for eval_v1.main_run (owner rule 2026-10-03).

``legacy``    : stop as soon as trainer_v2.metrics_v9.OverAbstainMonitor trips (window precision < threshold).
``confirmed`` : the monitor only logs; stop ONLY IF step >= warm-up AND the train window precision is below the
                threshold AND dev ``val/false_withhold`` rose strictly at each of the last two consecutive dev evals
                (three dev points, two consecutive rises).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)
RULES = ("legacy", "confirmed")
REASON = "over_abstain"
DEV_KEY = "val/false_withhold"


class StopRule:
    """Tracks dev false_withhold and decides the hard stop; ``enabled`` False = log ``stop/armed`` but never stop."""

    def __init__(self, rule: str, enabled: bool, warmup: int, window: int) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown stop rule: {rule}")
        self.rule, self.enabled, self.warmup, self.window = rule, enabled, warmup, window
        self.dev: list[tuple[int, Optional[float]]] = []    # (step, val/false_withhold); None = dev eval failed
        self.stopped = False
        self.step: Optional[int] = None

    def add_dev(self, step: int, payload: dict) -> None:
        val = (payload or {}).get(DEV_KEY)
        self.dev = [d for d in self.dev if d[0] != step] + [(step, None if val is None else float(val))]
        self.dev.sort(key=lambda d: d[0])

    def dev_rising(self) -> bool:
        """Two consecutive strict rises across the last three dev points (a failed point breaks the chain)."""
        last = [v for _, v in self.dev[-3:]]
        return len(last) == 3 and None not in last and last[0] < last[1] < last[2]

    def armed(self, step: int, monitor_rec: Optional[dict]) -> bool:
        breach = bool((monitor_rec or {}).get("below_threshold"))
        if self.rule == "legacy":
            return breach
        return step >= self.warmup and breach and self.dev_rising()

    def update(self, step: int, monitor_rec: Optional[dict], monitor_stopped: bool) -> dict:
        """Decide after the step (and its dev eval, if any); return the wandb ``stop/*`` + window payload."""
        armed = self.armed(step, monitor_rec)
        if self.enabled and not self.stopped:
            hit = monitor_stopped if self.rule == "legacy" else armed
            if hit:
                self.stopped, self.step = True, step
                logger.warning("over-abstain stop (%s) at step %d: monitor=%s dev=%s", self.rule, step,
                               monitor_rec, self.dev[-3:])
        out = {"train/step": step, "stop/armed": int(armed)}
        prec = (monitor_rec or {}).get("window_withhold_precision")
        if prec is not None:
            out[f"train/behavior/withhold_precision_window{self.window}"] = prec
        return out

    def summary(self) -> dict:
        return {"stop/reason": REASON if self.stopped else None, "stop/step": self.step, "stop/rule": self.rule}


def dev_history(out: Path, start: int, read: Callable[[Path], list], payload: Callable[[list, int], dict]) -> list:
    """Resume: (step, payload) of every written dev eval with step <= ``start`` (same source as wandb replay)."""
    hist = []
    for d in sorted((out / "dev_eval").glob("step_*"), key=lambda p: int(p.name.removeprefix("step_"))):
        step = int(d.name.removeprefix("step_"))
        if step <= start and (d / "eval_report.json").exists():
            recs = read(d / "records.jsonl")
            hist.append((step, payload(recs, step) if recs else {}))
    return hist
