"""Training-reward fix (pilot F, owner 2026-10-14): keep the risk term ON on target-absent rows whose IMAGE shows a
visible non-acute hazard. NEW file: reward_v5 / reward_v4 / reward_v4h / reward_v4hg / patch_h are untouched.

reward_v5._branch_target masks the risk group (risk, false_alarm, missed_warning) on a withhold (gt_absent) target
with memories shown (passive / called branch) unless the row's acute_visual_hazard.present is true, on the ground that
the risk depends on the missing fact. That is wrong when the image itself shows a hazard (e.g. slippers on the walking
path): an answer that only asks about storage and ignores the visible trip risk is never penalised.
Fix (``--absent-visible-risk on``; default ``off`` = nothing is installed): on a gt_absent row whose ledger carries
exactly that mask ("masked:gt_absent_risk_depends_on_missing_fact") and whose image shows a visible hazard
(acute_visual_hazard.image_verdict.severity minor / acute, or a non-empty acute_visual_hazard.evidence), the three
risk components are scored as reward_v5._terms scores them (risk label credit, false_alarm if the preset has it,
missed_warning, x the preset's missed-warning scale of reward_v4hg) and the total is re-summed. Unchanged: rows with
no visible hazard (severity none, no evidence: the risk depends only on the missing fact), the not-called branch
(no memory shown), row-level risk masks, critical_miss (GB-1), judge, hazard_first, the degenerate gate, every other row.
"""
from __future__ import annotations

import copy
from typing import Any

from trainer_v2 import reward_v5 as R5

MODES = ("off", "on")
MASK = "masked:gt_absent_risk_depends_on_missing_fact"
STATUS = "active:absent_visible_risk"
RISK = ("risk", "false_alarm", "missed_warning")


def visible_hazard(row: dict) -> bool:
    a = row.get("acute_visual_hazard")
    if a is None and isinstance(row.get("evaluator"), dict):
        a = row["evaluator"].get("acute_visual_hazard")
    a = a or {}
    sev = (a.get("image_verdict") or {}).get("severity")
    return sev in ("minor", "acute") or bool(str(a.get("evidence") or "").strip())


def qualifies(led: dict, row: dict) -> bool:
    comps = led["components"]
    return (row.get("condition") == "gt_absent" and not comps["degenerate"]["applicable"]
            and all((comps.get(c) or {}).get("status") == MASK for c in RISK) and visible_hazard(row))


def apply(led: dict, row: dict, mode: str = "on") -> dict:
    """The ledger with the risk group restored (a new object when it fires; the same object otherwise)."""
    if mode not in MODES:
        raise ValueError(f"absent_visible_risk must be one of {MODES}, got {mode!r}")
    if mode == "off" or not qualifies(led, row):
        return led
    from trainer_v2_v4 import reward_v4hg as RHG
    out = copy.deepcopy(led)
    comps, t = out["components"], out["target"]
    gt, sp, pred = t["risk_label"], t["safety_priority"], (out.get("parse") or {}).get("risk_pred")
    R2 = R5.R2
    comps["risk"] = {"applicable": True, "value": float(R5.W["risk"] * R2.RISK_SCORE[gt].get(pred, 0.0)),
                     "status": STATUS}
    if out["flags"].get("false_alarm", True):
        comps["false_alarm"] = {"applicable": True, "value": float(R5.PEN["false_alarm"] * float(
            gt == "safe" and pred == "unsafe")), "status": STATUS}
    scale = RHG.MW_SCALE.get(out["flags"].get("name"), 1.0) if not (out.get("v4h") or {}).get("acute") else 1.0
    comps["missed_warning"] = {"applicable": True, "value": float(R5.PEN["missed_warning"] * scale * float(
        gt == "unsafe" and sp not in R2.ACUTE and pred == "safe")), "status": STATUS + (f":x{scale}" if scale != 1.0
                                                                                         else "")}
    out["total"] = round(sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None), 10)
    out["applicable_components"] = sorted(k for k, c in comps.items() if c["applicable"])
    out["flags"] = {**out["flags"], "absent_visible_risk": mode}
    out["absent_visible_risk"] = {"applied": True, "base_total": led["total"]}
    return out


def install(mode: str) -> None:
    """In-process: reward_v4hg.score_episode applies the fix (``off``: nothing is installed)."""
    if mode not in MODES:
        raise ValueError(f"absent_visible_risk must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_avr_orig", None):
        return
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        return apply(orig(ep, cfg, hz), ep.item.row, mode)
    score_episode._avr_orig = orig
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    orig = getattr(RHG.score_episode, "_avr_orig", None)
    if orig is not None:
        RHG.score_episode = orig


__all__ = ["MODES", "MASK", "STATUS", "visible_hazard", "qualifies", "apply", "install", "uninstall"]
