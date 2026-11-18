"""Warning-credit hook for the training reward (owner decision 2026-10-14; NEW file, wired into NO entry point).

Rule (same as eval_v1/metrics_v6): the EFFECTIVE risk prediction of an episode is
    "unsafe"  if <risk> == "unsafe" OR the <answer> warns specifically (warning detector)
    <risk>    otherwise
so on a gold-unsafe row a specific warning earns the risk credit despite <risk>safe</risk>, and on a gold-safe row an
unnecessary specific warning counts as a false alarm. Scoring eff == gold reproduces metrics_v6.risk_outcome.
Backends: "cue" (default; frozen wd_cue_v2, CPU, deterministic) or "judge" (gpt-6-luna@medium, wd_judge_v2, via
v4_build/warning_judge/wd_judge.WarningJudge: USD cap, ledger, reply cache; per-step call cap; every unjudged episode
falls back to the cue). Only episodes whose outcome the warning can change are sent to the judge
(gold unsafe with <risk> != unsafe; gold safe with <risk> == safe); masking stays the caller's (reward_v5) business.
When the decisive memory was not available (gt_absent row, or a memory_call episode whose Stage 1 did not call) the
reference is stripped to the visible hazard (HazardRef.without_memory): no credit for guessing a memory-only risk.
Integration for pilot F's wrapper: v4_build/warning_judge/INTEGRATION.md.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

_WJ = Path(__file__).resolve().parents[1] / "v4_build" / "warning_judge"
if str(_WJ) not in sys.path:
    sys.path.insert(0, str(_WJ))
import warning_detector as WD  # noqa: E402
import wd_judge as J  # noqa: E402

logger = logging.getLogger(__name__)
_RISK = re.compile(r"<risk>\s*(.*?)\s*</risk>", re.S | re.I)
MAX_JUDGE_CALLS_PER_STEP = 32          # = G 8 x 4 prompts: every episode of a step at most
JUDGE_USD_CAP = 2.0
STEP_BUDGET_S = 60.0


def risk_tag(stage2: str) -> Optional[str]:
    m = _RISK.search(stage2 or "")
    return m.group(1).strip().lower() if m else None


def effective_risk(gold_risk: str, risk_pred: Optional[str], warns_specific: bool) -> Optional[str]:
    """The rule: "unsafe" if tagged unsafe or the answer warns specifically, else the tag."""
    return "unsafe" if (risk_pred == "unsafe" or warns_specific) else risk_pred


def needs_judge(gold_risk: str, risk_pred: Optional[str]) -> bool:
    """Can a warning change this episode's risk outcome? (unsafe: tag not unsafe; safe: tag safe)"""
    return (gold_risk == "unsafe" and risk_pred != "unsafe") or (gold_risk == "safe" and risk_pred == "safe")


_CALL = re.compile(r"<memory_call>\s*(\w+)\s*</memory_call>", re.I)


def memory_called(stage1: Optional[str]) -> Optional[bool]:
    """memory_call arm: True iff the Stage-1 decision is call_pdb; None for a passive episode (no Stage 1)."""
    if stage1 is None:
        return None
    m = _CALL.findall(stage1)
    return bool(m) and m[-1].lower() == "call_pdb"


def rewrite_risk_tag(stage2: str, eff: Optional[str]) -> str:
    """Copy of the Stage-2 text with its <risk> tag set to ``eff`` (unchanged if no tag / eff None): lets a wrapper
    rescore with reward_v5 untouched."""
    if eff is None or not _RISK.search(stage2 or ""):
        return stage2
    return _RISK.sub(f"<risk>{eff}</risk>", stage2, count=1)


class WarningRewardHook:
    """Per-run hook. ``refs``: sample_id -> HazardRef (load_refs). backend "cue" | "judge"."""

    def __init__(self, refs: dict[str, "WD.HazardRef"], backend: str = "cue", judge: Any = None,
                 max_judge_calls_per_step: int = MAX_JUDGE_CALLS_PER_STEP, step_budget_s: float = STEP_BUDGET_S):
        if backend not in WD.BACKENDS:
            raise ValueError(f"backend must be one of {WD.BACKENDS}")
        if backend == "judge" and judge is None:
            raise ValueError("backend 'judge' needs a wd_judge.WarningJudge (make_judge)")
        self.refs, self.backend, self.judge = refs, backend, judge
        self.max_calls, self.budget_s = int(max_judge_calls_per_step), float(step_budget_s)
        self.stats = {"episodes": 0, "judge_sent": 0, "judge_failed": 0, "judge_skipped_cap": 0,
                      "warn_credit": 0, "warn_false_alarm": 0}

    def ref(self, sample_id: str) -> "WD.HazardRef":
        return self.refs[sample_id]

    def ref_for(self, sample_id: str, stage1: Optional[str] = None) -> "WD.HazardRef":
        """The row's reference, stripped of memory content when the decisive memory was not available
        (gt_absent row, or a memory_call episode without a call)."""
        return self.refs[sample_id].for_episode(memory_called(stage1))

    def judge_step(self, episodes: list[tuple]) -> list[Optional[dict]]:
        """episodes = [(sample_id, stage2_text[, stage1_text])] of ONE step -> parsed judgments (None = use the
        cue). Only outcome-relevant episodes are sent, at most ``max_judge_calls_per_step`` unique prompts."""
        out: list[Optional[dict]] = [None] * len(episodes)
        if self.backend != "judge":
            return out
        idx, prompts = [], []
        for i, (sid, s2, *s1) in enumerate(episodes):
            ref = self.ref_for(sid, s1[0] if s1 else None)
            if needs_judge(ref.gold_risk, risk_tag(s2)):
                idx.append(i)
                prompts.append(J.judge_prompt(ref, J.answer_text(s2)))
        uniq = list(dict.fromkeys(prompts))
        send = set(uniq[: self.max_calls])
        self.stats["judge_skipped_cap"] += len(uniq) - len(send)
        replies, _ = self.judge.run(list(send), self.budget_s) if send else ({}, {})
        self.stats["judge_sent"] += len(send)
        for i, p in zip(idx, prompts):
            out[i] = J.parse_reply(replies.get(p)) if p in send else None
            self.stats["judge_failed"] += p in send and out[i] is None
        return out

    def score(self, sample_id: str, stage2: str, judgment: Optional[dict] = None,
              stage1: Optional[str] = None) -> dict:
        """{risk_pred, warns_specific, effective_risk, changed, backend, reason, memory_ref} for one episode."""
        ref = self.ref_for(sample_id, stage1)
        tag = risk_tag(stage2)
        d = WD.detect(ref, J.answer_text(stage2), self.backend, judgment)
        eff = effective_risk(ref.gold_risk, tag, d["warns_specific"])
        changed = eff != tag
        self.stats["episodes"] += 1
        if changed:
            self.stats["warn_credit" if ref.gold_risk == "unsafe" else "warn_false_alarm"] += 1
        return {"risk_pred": tag, "warns_specific": d["warns_specific"], "effective_risk": eff, "changed": changed,
                "backend": "cue_fallback" if d.get("fallback") else d["backend"], "reason": d["reason"],
                "memory_ref": ref.memory_present}

    def score_step(self, episodes: list[tuple]) -> list[dict]:
        """episodes = [(sample_id, stage2_text[, stage1_text])]; pass stage1 on the memory_call arm."""
        js = self.judge_step(episodes)
        return [self.score(e[0], e[1], j, e[2] if len(e) > 2 else None) for e, j in zip(episodes, js)]


def load_refs(release_dir: Path, splits: Iterable[str] = ("train", "dev")) -> dict[str, "WD.HazardRef"]:
    """sample_id -> HazardRef for a release directory (e.g. RELEASE_TRAIN_DEV_v11); axes joined for dev only."""
    import json
    out: dict[str, WD.HazardRef] = {}
    for sp in splits:
        rows = [json.loads(x) for x in (Path(release_dir) / f"{sp}.jsonl").read_text().splitlines() if x.strip()]
        out.update(WD.refs_for_rows(rows, "dev" if sp == "dev" else sp))
    return out


def make_judge(usage_log: Path, usd_cap: float = JUDGE_USD_CAP, cache_path: Optional[Path] = None,
               model: str = "gpt-6-luna@medium") -> "J.WarningJudge":
    """PAID judge for backend "judge" (the caller logs the PRELAUNCH line first)."""
    return J.WarningJudge(Path(usage_log), model=model, usd_cap=usd_cap, cache_path=cache_path)


__all__ = ["WarningRewardHook", "memory_called", "effective_risk", "needs_judge", "risk_tag", "rewrite_risk_tag", "load_refs",
           "make_judge", "MAX_JUDGE_CALLS_PER_STEP", "JUDGE_USD_CAP"]
