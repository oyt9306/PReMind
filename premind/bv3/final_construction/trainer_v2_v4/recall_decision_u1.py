"""Recall-phase call decision ``ws52u1`` for CUR22R3 (prepared 2026-10-16; CUR22R3.md). NEW file: wraps
recall_decision.py (imported read-only, never edited) as a pure ledger post-processing of its ``ws52`` output.

ws52u1 = ws52 with ONLY the unneeded-call weight doubled (-0.5r -> -1r, = -0.10 at r = 0.1):
    correct decision                                                         +r   (ws52)
    missed needed call (direct answer / invalid Stage 1 on a memory-required row)  -2r  (ws52 = v52ws_ag.missed_call)
    unneeded call (call_pdb on a row that does not need memory)              -1r  (ws52: -0.5r)
    invalid Stage 1 on a row that does not need memory                       -r   (ws52)
    generic panic / calm G0 query (non-active "neutral" status)              0    (ws52)
No existing frozen preset has this table: reward_v5 "dec-sym" (the "w/ symmetric recall reward" ablation) is main with
unneeded_call -1 but missed_call -1 (not -2); "v52ws_ag" has missed -2 / unneeded -0.5. Hence a new mode.
The credit_gate crowd-out guard is unchanged (it only zeroes a positive call credit; an unneeded call is negative).
"""
from __future__ import annotations

import copy
from typing import Any

from trainer_v2 import reward_v5 as R5
from trainer_v2_v4 import recall_decision as RD

MODE = "ws52u1"
DECISION_MODES = RD.DECISION_MODES + (MODE,)
GUARD_MODES = RD.GUARD_MODES
MISSED_CALL = RD.MISSED_CALL            # -2.0 (frozen v52ws_ag)
UNNEEDED_CALL = -1.0                    # ws52: -0.5 (v52ws_ag)
WEIGHTS = {"correct": 1.0, "missed_call": MISSED_CALL, "unneeded_call": UNNEEDED_CALL, "invalid_not_required": -1.0,
           "neutral": 0.0}
assert (RD.MISSED_CALL, RD.UNNEEDED_CALL) == (-2.0, -0.5) == (R5.PRESETS["v52ws_ag"].missed_call,
                                                             R5.PRESETS["v52ws_ag"].unneeded_call), "ws52 drifted"
assert WEIGHTS == {"correct": 1.0, "missed_call": -2.0, "unneeded_call": -1.0, "invalid_not_required": -1.0,
                   "neutral": 0.0}, "ws52u1 weight table changed: re-pin CUR22R3"
VERSION = "cur22r3_recall_ws52u1_v1"


def ws52u1_value(decision: Any, memory_required: bool, r: float) -> float:
    """ws52 value with the unneeded-call weight replaced."""
    if decision == "call_pdb" and not memory_required:
        return UNNEEDED_CALL * r
    return RD.ws52_value(decision, memory_required, r)


def apply(led: dict, decision_mode: str = MODE, guard: str = "credit_gate") -> dict:
    """``ws52u1``: recall_decision.apply(led, "ws52", guard) with the unneeded-call value -1r; any other mode is
    recall_decision.apply unchanged."""
    if decision_mode != MODE:
        return RD.apply(led, decision_mode, guard)
    out = RD.apply(led, "ws52", guard)
    if out is led:                                      # untouched (degenerate / passive / not applicable / done)
        return out
    out = copy.deepcopy(out)
    dec = out["components"]["decision"]
    status = str(dec.get("status"))
    if ":ws52" in status:                               # active decision re-valued by ws52
        r, d, mr = float(out["r"]), out.get("decision"), bool(out["memory_required"])
        value = float(dec["value"])
        if d == "call_pdb" and not mr:
            value = UNNEEDED_CALL * r
        dec.update(value=float(value), status=status.replace(":ws52", f":{MODE}", 1))
        out["recall"]["decision"] = float(value)
    out["total"] = round(sum(c["value"] for c in out["components"].values()
                             if c["applicable"] and c["value"] is not None), 10)
    out["flags"] = {**out["flags"], "recall_decision": {**out["flags"]["recall_decision"], "mode": MODE,
                                                        "unneeded_call": UNNEEDED_CALL}}
    out["recall"]["version_u1"] = VERSION
    return out


def install(decision_mode: str, guard: str) -> None:
    """In-process: reward_v4hg.score_episode -> apply (innermost, as recall_decision.install)."""
    if decision_mode != MODE:
        return RD.install(decision_mode, guard)
    if guard not in GUARD_MODES:
        raise ValueError(f"crowdout_guard {guard!r} invalid")
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_rd_orig", None):
        return None
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        return apply(orig(ep, cfg, hz), MODE, guard)
    score_episode._rd_orig = orig
    RHG.score_episode = score_episode
    return None


uninstall = RD.uninstall

__all__ = ["MODE", "DECISION_MODES", "GUARD_MODES", "MISSED_CALL", "UNNEEDED_CALL", "WEIGHTS", "VERSION",
           "ws52u1_value", "apply", "install", "uninstall"]
