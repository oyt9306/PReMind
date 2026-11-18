"""Training-reward fix for acute target-absent rows (pilot F, owner 2026-10-14; test-validity audit
v4_build/test_validity_v1/REPORT.md). NEW file: reward_v5 / reward_v4 / reward_v4h / reward_v4hg / patch_h are untouched.

Problem: on an acute (image verdict) row whose decisive memory is absent (context_condition gt_absent, withhold
target) gold_action_set = {safety_escalation}, yet every gold answer addresses the hazard AND asks for the missing
fact. A rollout that does exactly that but is labelled ``clarify`` loses the action term (W 0.08) although its answer
is the gold behaviour; in the v1 pilots 50-62 % of acute-absent GRPO groups had all 8 actions wrong.

Fix (``--acute-absent-clarify accept``; default ``off`` = nothing is installed): on such a row a ``clarify`` label is
action-correct iff the hazard is addressed by the criterion of the v4h hazard term, i.e. the (guard-applied) judge or
cue state is "specific" AND nothing is contraindicated
(ledger ``v4h.state`` / ``v4h.contraindicated``; accepting clarify is never a safety escape). Then the action
component gets W["action"] and the total is re-summed
exactly as the guard sums it. Everything else is unchanged: clarify without hazard handling stays wrong, abstain,
the withhold credit (withhold_v9 (a)-(d), gold_pattern based), hazard terms, degenerate gate, every other row.
Rows: item.row["condition"] == "gt_absent", ledger target withhold, hazard term applicable (image-level acute verdict,
reward_v4h.hazard_info), gold_action_set without clarify. In-process hook: reward_v4hg.score_episode (patch_h._score
calls it for every non-aux episode under a v4wh preset).
"""
from __future__ import annotations

import copy
from typing import Any

from trainer_v2 import reward_v5 as R5

MODES = ("off", "accept")
STATUS = "active:acute_absent_clarify_accept(hazard_specific_not_contraindicated)"


def qualifies(led: dict, row: dict) -> bool:
    """The rollout is a clarify on an acute target-absent withhold row with the hazard addressed (state specific)
    and the action term currently 0."""
    comp = led["components"].get("action") or {}
    hz = led.get("v4h") or {}
    return (row.get("condition") == "gt_absent" and bool(led["target"].get("withhold"))
            and "clarify" not in led["target"].get("gold_action_set", ())
            and (led.get("parse") or {}).get("action_pred") == "clarify"
            and bool(hz.get("acute")) and hz.get("state") == "specific" and hz.get("contraindicated") is False
            and comp.get("applicable") is True and comp.get("value") == 0.0)


def apply(led: dict, row: dict, mode: str = "accept") -> dict:
    """The ledger with the fix (a new object when it fires; the same object otherwise)."""
    if mode not in MODES:
        raise ValueError(f"acute_absent_clarify must be one of {MODES}, got {mode!r}")
    if mode == "off" or not qualifies(led, row):
        return led
    out = copy.deepcopy(led)
    comps = out["components"]
    comps["action"] = {**comps["action"], "value": float(R5.W["action"]), "status": STATUS}
    out["total"] = round(sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None), 10)
    out["flags"] = {**out["flags"], "acute_absent_clarify": mode}
    out["acute_absent_clarify"] = {"applied": True, "base_total": led["total"]}
    return out


def install(mode: str) -> None:
    """In-process: reward_v4hg.score_episode applies the fix (``off``: nothing is installed)."""
    if mode not in MODES:
        raise ValueError(f"acute_absent_clarify must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_aa_orig", None):
        return
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        return apply(orig(ep, cfg, hz), ep.item.row, mode)
    score_episode._aa_orig = orig
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    orig = getattr(RHG.score_episode, "_aa_orig", None)
    if orig is not None:
        RHG.score_episode = orig


__all__ = ["MODES", "STATUS", "qualifies", "apply", "install", "uninstall"]
