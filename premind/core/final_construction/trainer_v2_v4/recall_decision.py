"""Recall-phase (Stage B, memory_call arm) reward terms for CUR23R (prepared 2026-10-16; RECALL_PHASE.md). NEW file:
reward_v5 / reward_v4 / reward_v4h / reward_v4hg and every CUR23 hook are untouched (pure ledger post-processing).

1. Call decision ``--recall-decision ws52`` (default ``off`` = the v4wh_g ledger unchanged). The v4wh_g ledger scores
   the Stage-1 decision with reward_v5's "main" flags (correct +r, missed needed call -r, unneeded call 0, invalid -r).
   ws52 re-values it with the decision weights of the frozen warm-start Stage-B preset ``v52ws_ag`` (owner 2026-10-10,
   adopted PReMind-4B; read from reward_v5.PRESETS and asserted, never re-typed):
       correct decision                                   +r
       missed needed call (direct answer / invalid Stage 1 on a memory-required row)   missed_call   x r = -2r
       unneeded call (call_pdb on a row that does not need memory)                     unneeded_call x r = -0.5r
       invalid Stage 1 on a row that does not need memory                              -r (unchanged)
       generic panic / calm G0 query (reward_v5 NEUTRAL_DECISION_STYLES)               0  (unchanged, "neutral")
   The Match-gated withhold credit of v52ws_ag is NOT carried over: it needs the vef2 <visual_evidence> schema, which
   the CUR23 Stage A never trained (reward_v4.register refuses a withhold_match base).
2. Crowd-out guard ``--crowdout-guard credit_gate`` (default ``off``; design audit v4_build/design_audit_v1 section 1:
   after a memory call 52.6 % of acute answers never address the hazard vs 32.6 % passive): on an acute row (image
   verdict, ledger ``v4h.acute``) whose Stage 1 correctly called memory (branch ``called``, decision value > 0), the
   +r call credit is paid only if the answer addresses the hazard, i.e. the (guard-applied) hazard state is
   ``specific`` or ``generic_only`` and nothing is contraindicated; otherwise the decision component is 0.
   The call target is unchanged: for every Stage-2 outcome, calling (0 or +r) still beats not calling (-2r) on a
   memory-required row, so the guard only widens the after-call gap between "hazard addressed" and "hazard crowded
   out" by r. Rows that do not need memory, not-called branches, non-acute rows and degenerate episodes are untouched.
Both are applied innermost (installed before HJ4 / bonus v1.1 / fa25 / AA / AVR / warning credit / severity cap), so
every outer hook re-sums the adjusted decision component and the severity cap stays the outermost transform.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from trainer_v2 import reward_v5 as R5

DECISION_MODES = ("off", "ws52")
GUARD_MODES = ("off", "credit_gate")
SOURCE_PRESET = "v52ws_ag"
MISSED_CALL = R5.PRESETS[SOURCE_PRESET].missed_call        # -2.0 (frozen)
UNNEEDED_CALL = R5.PRESETS[SOURCE_PRESET].unneeded_call    # -0.5 (frozen)
assert (MISSED_CALL, UNNEEDED_CALL) == (-2.0, -0.5), "v52ws_ag decision weights changed: re-pin the recall phase"
ADDRESSED = ("specific", "generic_only")
VERSION = "cur23r_recall_v1"
GUARD_RULE = ("acute (v4h.acute) & branch called & decision credit > 0 (correct call on a memory-required row) & "
              "(hazard state not in {specific, generic_only} or contraindicated) -> decision = 0")


def _main_value(decision: Optional[str], memory_required: bool, r: float) -> float:
    """reward_v5 decision value under the "main" flags (what the v4wh_g ledger carries)."""
    if decision == ("call_pdb" if memory_required else "direct_answer"):
        return r
    if decision == "call_pdb" and not memory_required:
        return R5.PRESETS["main"].unneeded_call * r
    return (R5.PRESETS["main"].missed_call * r) if memory_required else -r


def ws52_value(decision: Optional[str], memory_required: bool, r: float) -> float:
    """reward_v5._score's decision value under the v52ws_ag decision weights."""
    if decision == ("call_pdb" if memory_required else "direct_answer"):
        return r
    if decision == "call_pdb" and not memory_required:
        return UNNEEDED_CALL * r
    return MISSED_CALL * r if memory_required else -r


def crowded_out(led: dict) -> Optional[str]:
    """The hazard state that gates the call credit (None = addressed / not applicable)."""
    v = led.get("v4h") or {}
    if not v.get("acute"):
        return None
    if v.get("contraindicated"):
        return "contraindicated"
    st = v.get("state")
    return None if st in ADDRESSED else str(st)


def apply(led: dict, decision_mode: str = "ws52", guard: str = "credit_gate") -> dict:
    """The ledger with the recall-phase decision terms (a new object when anything changes; the same otherwise)."""
    if decision_mode not in DECISION_MODES or guard not in GUARD_MODES:
        raise ValueError(f"recall_decision {decision_mode!r} / crowdout_guard {guard!r} not in {DECISION_MODES} / "
                         f"{GUARD_MODES}")
    if (decision_mode == "off" and guard == "off") or "recall" in led:
        return led
    comps = led["components"]
    dec = comps.get("decision") or {}
    if comps["degenerate"]["applicable"] or not dec.get("applicable") or led.get("arm") != "call":
        return led
    r, d, mr = float(led["r"]), led.get("decision"), bool(led["memory_required"])
    base = float(dec["value"])
    active = str(dec.get("status")).startswith("active")
    if active and abs(base - _main_value(d, mr, r)) > 1e-9:
        raise RuntimeError(f"recall_decision: base decision {base} != main-flag value for {d!r}/{mr} (preset drift)")
    value, status = base, str(dec.get("status"))
    if active and decision_mode == "ws52":
        value = ws52_value(d, mr, r)
        status = f"{status}:ws52"
    gated = None
    if guard == "credit_gate" and led.get("branch") == "called" and value > 0:
        gated = crowded_out(led)
        if gated is not None:
            value, status = 0.0, f"{status}:crowdout_gate({gated})"
    out = copy.deepcopy(led)
    out["components"]["decision"] = {**dec, "value": float(value), "status": status}
    out["total"] = round(sum(c["value"] for c in out["components"].values()
                             if c["applicable"] and c["value"] is not None), 10)
    out["flags"] = {**out["flags"], "recall_decision": {"mode": decision_mode, "guard": guard,
                                                        "missed_call": MISSED_CALL, "unneeded_call": UNNEEDED_CALL}}
    out["recall"] = {"version": VERSION, "base_decision": base, "decision": float(value),
                     "crowdout_gated": gated, "base_total": led["total"]}
    return out


def install(decision_mode: str, guard: str) -> None:
    """In-process: reward_v4hg.score_episode -> apply (both ``off``: nothing). Install BEFORE main_v4hcur23.main so
    this is the innermost wrapper (every outer hook re-sums the adjusted decision)."""
    if decision_mode not in DECISION_MODES or guard not in GUARD_MODES:
        raise ValueError(f"recall_decision {decision_mode!r} / crowdout_guard {guard!r} invalid")
    if decision_mode == "off" and guard == "off":
        return
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_rd_orig", None):
        return
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        return apply(orig(ep, cfg, hz), decision_mode, guard)
    score_episode._rd_orig = orig
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    orig = getattr(RHG.score_episode, "_rd_orig", None)
    if orig is not None:
        RHG.score_episode = orig


__all__ = ["DECISION_MODES", "GUARD_MODES", "SOURCE_PRESET", "MISSED_CALL", "UNNEEDED_CALL", "ADDRESSED", "VERSION",
           "GUARD_RULE", "ws52_value", "crowded_out", "apply", "install", "uninstall"]
