"""Acute memory-informed response bonus v1.1 (Codex review K, 2026-10-15, blocker 3). NEW file: acute_memory_bonus.py
(v1) is imported read-only, never edited. Design: ACUTE_MEMORY_BONUS_V1_1.md.

Same criteria and weight as v1 (+0.06, component ``acute_memory``) except the "hazard action outside the memory fact"
gate, which is now semantic at the key-action level:
  memory_labels = the specific key-action labels (hazard_cue.classify, the row's RowActions) matched by
                  (i)  every <answer> sentence that hits a support pattern, UNMASKED (if no single sentence hits, the
                       shortest run(s) of consecutive sentences that hit; at worst the whole answer = fail closed), and
                  (ii) the support pattern text itself (the remembered fact, regex syntax stripped)
  credited      = the specific labels v1 credits (masked answer) MINUS memory_labels
  fires only if credited is non-empty.
So a paraphrase of the remembered action ("Move the backpack beside the desk. Clear the bag from his feet.":
clear_trip_hazard is matched by the memory sentence) earns nothing, while a distinct hazard action (ankle check / stop
walking) still does.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Optional

from trainer_v2 import personal_match as PM
from trainer_v2_v4 import acute_memory_bonus as V1

MODES, BONUS, COMPONENT = V1.MODES, V1.BONUS, V1.COMPONENT
VERSION = "v1.1"
STATUS = "active:acute_memory_bonus_v1_1(hazard_specific_not_contraindicated&personal_content_full&distinct_key_action)"
CRITERION = ("not degenerate & v4h.acute & condition gt_present & personal_content applicable & v4h.state == specific "
             "& not contraindicated & personal_content == W[personal_content] & >= 1 specific key-action label credited "
             "in the masked answer that is NOT matched by the memory-support sentence(s) (unmasked) or the support "
             "fact text -> + BONUS (component acute_memory)")
_FLAGS = re.compile(r"\(\?[a-zA-Z]+\)|\(\?[:=!]|\(\?<[=!]")


def fact_text(pattern: str) -> str:
    """The support pattern as plain text (inline flags, regex syntax and alternation bars removed)."""
    return re.sub(r"\s+", " ", re.sub(r"[()|\\]", " ", PM._clean(_FLAGS.sub(" ( ", pattern)))).strip()


def memory_spans(answer: str, pattern: str, mode: str = "soft") -> list[str]:
    """Answer sentences that carry the memory fact (unmasked): sentences hitting the pattern, else the shortest
    consecutive runs that hit, else [] (the pattern is not in the answer)."""
    sents = [x for x in V1._SENT.split(answer or "") if x and x.strip()]
    for size in range(1, len(sents) + 1):
        runs = [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]
        hits = [x for x in runs if PM.hit(pattern, x, mode)]
        if hits:
            return hits
    return []


def memory_labels(ra: Any, answer: str, patterns: list[str], hazard_pattern: Optional[str] = None,
                  mode: str = "soft") -> list[str]:
    from trainer_v2_v4 import hazard_cue as HC
    texts = [s for p in patterns for s in memory_spans(answer, p, mode)] + [fact_text(p) for p in patterns]
    return sorted({lab for t in texts if t for lab in HC.classify(ra, t, hazard_pattern)["specific_hits"]})


def credit(ra: Any, answer: str, patterns: list[str], hazard_pattern: Optional[str] = None,
           mode: str = "soft") -> dict:
    """{"credited": labels outside the memory, "masked_hits": v1 labels, "memory_labels": ...}."""
    masked = V1.credited_actions(ra, answer, patterns, hazard_pattern, mode)
    mem = memory_labels(ra, answer, patterns, hazard_pattern, mode)
    return {"credited": [x for x in masked if x not in mem], "masked_hits": list(masked), "memory_labels": mem}


def credit_of(ep: Any, hz: Optional[dict], led: dict) -> Optional[dict]:
    """credit() for one episode (None when no RowActions: fail closed)."""
    from trainer_v2_v4 import hazard_rubric as HR
    if not hz or hz.get("ra") is None:
        return None
    t = ep.item.target
    return credit(hz["ra"], HR.answer_text(ep.stage2_text), [p for _, p in sorted(t["support"].items())],
                  t.get("hazard_pattern"), (led.get("flags") or {}).get("personal_match", "soft"))


def reason(led: dict, row: dict, cred: Optional[dict] = None, *, check_credit: bool = True) -> Optional[str]:
    """None when the bonus fires; otherwise the first failing criterion. ``cred`` = credit() (None = not evaluated)."""
    why = V1.reason(led, row, check_credit=False)
    if why is not None or not check_credit:
        return why
    if cred is None:
        return "credit_not_evaluated"
    return None if cred["credited"] else "no_key_action_outside_memory"


def apply(led: dict, row: dict, mode: str = "on", cred: Optional[dict] = None) -> dict:
    """The ledger with the bonus (a new object when it fires; the same object otherwise)."""
    if mode not in MODES:
        raise ValueError(f"acute_memory_bonus must be one of {MODES}, got {mode!r}")
    if mode == "off" or reason(led, row, cred) is not None:
        return led
    out = copy.deepcopy(led)
    comps = out["components"]
    comps[COMPONENT] = {"applicable": True, "value": float(BONUS), "status": STATUS}
    out["total"] = round(sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None), 10)
    out["applicable_components"] = sorted(k for k, c in comps.items() if c["applicable"])
    out["flags"] = {**out["flags"], "acute_memory_bonus": mode}
    out["acute_memory_bonus"] = {"applied": True, "version": VERSION, "base_total": led["total"],
                                 "credited_actions": list(cred["credited"]),
                                 "memory_labels": list(cred["memory_labels"])}
    return out


def install(mode: str) -> None:
    """In-process: reward_v4hg.score_episode applies the v1.1 bonus (``off``: nothing). Refuses next to v1."""
    if mode not in MODES:
        raise ValueError(f"acute_memory_bonus must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_amb11_orig", None):
        return
    if getattr(RHG.score_episode, "_amb_orig", None):
        raise RuntimeError("acute memory bonus v1 is installed; v1.1 refuses to stack on it")
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        led = orig(ep, cfg, hz)
        if reason(led, ep.item.row, check_credit=False) is not None:     # cheap criteria first
            return led
        return apply(led, ep.item.row, mode, credit_of(ep, hz, led))
    score_episode._amb11_orig = orig
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    orig = getattr(RHG.score_episode, "_amb11_orig", None)
    if orig is not None:
        RHG.score_episode = orig


__all__ = ["MODES", "BONUS", "COMPONENT", "VERSION", "STATUS", "CRITERION", "fact_text", "memory_spans",
           "memory_labels", "credit", "credit_of", "reason", "apply", "install", "uninstall"]
