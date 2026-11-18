"""Acute memory-informed response bonus (owner request 2026-10-15; NEW file: reward_v5 / reward_v4 / reward_v4h /
reward_v4hg / patch_h and every pilot-F hook are untouched). Design + replay numbers: ACUTE_MEMORY_BONUS.md.

Motivation: in pilot CUR (pilot_v4hA_cur_v12_4b_CUR) the policy handles the acute hazard but drops the memory fact on
acute G1G2 gt_present rows (personal_content hit ~.09). The owner wants memory-informed emergency answers ("call
emergency services and tell them she takes a blood thinner"). Today the two terms are paid independently
(hazard_appropriate +ALPHA, personal_content W 0.12), so the conjunction earns nothing extra.

Bonus (``--acute-memory-bonus on``; default ``off`` = nothing is installed): +BONUS (0.06 = reward_v4h.ALPHA, the
specific-hazard bonus) iff ALL of
  * not degenerate (the degenerate gate keeps total -1)
  * acute row (image-level acute verdict: ledger ``v4h.acute``, reward_v4h.hazard_info)
  * target memory shown (item.row["condition"] == "gt_present") and personal_content applicable (support pattern(s)
    shown, target not withhold, memory shown on this branch)
  * hazard state == "specific" (guard-applied judge / cue state, ledger ``v4h.state``) AND not contraindicated
  * personal_content == W["personal_content"] (every target entry: support pattern hit AND entry cited)
  * a credited hazard action that is NOT the memory fact (orchestrator 2026-10-15, row aug_text_v1/sample_041 where
    "move the backpack beside the desk" is both the support fact and a trip-hazard action): at least one specific
    key-action cue (hazard_cue.classify, the row's RowActions) still matches after the memory fact is masked out of
    the <answer>. Masking (``mask_memory``): in every sentence that hits a support pattern (personal_match, the
    preset's mode), each answer word sharing a 5-letter stem with the pattern's content words is blanked; if no single
    sentence hits, the same words are blanked in the whole answer. The judge reply has no per-action field
    ({appropriate, specific, contraindicated} only), so the cue labels are the only per-action evidence; the judge /
    cue state "specific" is still required on top.
Then component ``acute_memory`` = {applicable True, value BONUS, status STATUS} is added and the total is re-summed
exactly as the guard / acute_absent_fix sum it; ledger key ``acute_memory_bonus`` = {applied, base_total} and flag
``acute_memory_bonus`` = mode. Every other ledger is returned unchanged (same object). ``reason`` gives the first
failing criterion (offline audit / replay only; never written into a ledger).
In-process hook: reward_v4hg.score_episode (patch_h._score calls it for every non-aux episode under a v4wh preset).
Composable with any entry that ends in main_v4haa.main / main_v4hcur.main: call ``install(mode)`` before it (one line).
"""
from __future__ import annotations

import copy
import re
from typing import Any, Optional

from trainer_v2 import personal_match as PM
from trainer_v2 import reward_v5 as R5

MODES = ("off", "on")
BONUS = 0.06                     # = reward_v4h.ALPHA (specific-hazard bonus); documented in ACUTE_MEMORY_BONUS.md
COMPONENT = "acute_memory"
STATUS = "active:acute_memory_bonus(hazard_specific_not_contraindicated&personal_content_full)"
CRITERION = ("not degenerate & v4h.acute & condition gt_present & personal_content applicable & v4h.state == specific "
             "& not contraindicated & personal_content == W[personal_content] & >= 1 specific key-action cue outside "
             "the masked memory fact -> + BONUS (component acute_memory)")
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
_TOK = re.compile(r"[A-Za-z][A-Za-z']*")
MASK = "___"


def _stems(pattern: str) -> set[str]:
    words, groups = PM._units(pattern)
    allw = list(words) + [w for g in groups for alt in g for w in alt]
    return {w.lower()[:PM.STEM_LEN] for w in allw}


def mask_memory(answer: str, patterns: list[str], mode: str = "soft") -> str:
    """The <answer> with the memory fact(s) blanked (see the module docstring)."""
    out = answer or ""
    for pat in patterns:
        stems = _stems(pat)
        blank = lambda m: MASK if m.group(0).lower()[:PM.STEM_LEN] in stems else m.group(0)  # noqa: E731
        sents = [x for x in _SENT.split(out) if x]
        hits = [x for x in sents if PM.hit(pat, x, mode)]
        if hits:
            for x in hits:
                out = out.replace(x, _TOK.sub(blank, x), 1)
        else:
            out = _TOK.sub(blank, out)
    return out


def credited_actions(ra: Any, answer: str, patterns: list[str], hazard_pattern: Optional[str] = None,
                     mode: str = "soft") -> list[str]:
    """Specific key-action labels the cue credits in the <answer> once the memory fact is masked."""
    from trainer_v2_v4 import hazard_cue as HC
    return HC.classify(ra, mask_memory(answer, patterns, mode), hazard_pattern)["specific_hits"]


def credit_of(ep: Any, hz: Optional[dict], led: dict) -> Optional[list[str]]:
    """credited_actions for one episode (None when no RowActions: fail closed)."""
    from trainer_v2_v4 import hazard_rubric as HR
    if not hz or hz.get("ra") is None:
        return None
    t = ep.item.target
    return credited_actions(hz["ra"], HR.answer_text(ep.stage2_text), [p for _, p in sorted(t["support"].items())],
                            t.get("hazard_pattern"), (led.get("flags") or {}).get("personal_match", "soft"))


def reason(led: dict, row: dict, credited: Optional[list[str]] = None, *, check_credit: bool = True) -> Optional[str]:
    """None when the bonus fires; otherwise the first failing criterion (audit / replay). ``credited`` =
    credited_actions of the episode (None = not evaluated -> fails closed unless ``check_credit`` is False)."""
    comps = led["components"]
    hz = led.get("v4h") or {}
    pc = comps.get("personal_content") or {}
    if comps["degenerate"]["applicable"]:
        return "degenerate"
    if not hz.get("acute"):
        return "not_acute"
    if row.get("condition") != "gt_present":
        return "not_gt_present"
    if not pc.get("applicable") or pc.get("value") is None:
        return "personal_content_not_applicable"
    if hz.get("state") != "specific":
        return f"hazard_{hz.get('state')}"
    if hz.get("contraindicated") is not False:
        return "contraindicated"
    if abs(pc["value"] - R5.W["personal_content"]) > 1e-9:
        return "personal_content_partial" if pc["value"] > 0 else "personal_content_zero"
    if COMPONENT in comps:
        return "already_applied"
    if check_credit and not credited:
        return "no_credited_action_outside_memory" if credited is not None else "credit_not_evaluated"
    return None


def qualifies(led: dict, row: dict, credited: Optional[list[str]] = None) -> bool:
    return reason(led, row, credited) is None


def apply(led: dict, row: dict, mode: str = "on", credited: Optional[list[str]] = None) -> dict:
    """The ledger with the bonus (a new object when it fires; the same object otherwise)."""
    if mode not in MODES:
        raise ValueError(f"acute_memory_bonus must be one of {MODES}, got {mode!r}")
    if mode == "off" or not qualifies(led, row, credited):
        return led
    out = copy.deepcopy(led)
    comps = out["components"]
    comps[COMPONENT] = {"applicable": True, "value": float(BONUS), "status": STATUS}
    out["total"] = round(sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None), 10)
    out["applicable_components"] = sorted(k for k, c in comps.items() if c["applicable"])
    out["flags"] = {**out["flags"], "acute_memory_bonus": mode}
    out["acute_memory_bonus"] = {"applied": True, "base_total": led["total"], "credited_actions": list(credited)}
    return out


def install(mode: str) -> None:
    """In-process: reward_v4hg.score_episode applies the bonus (``off``: nothing is installed)."""
    if mode not in MODES:
        raise ValueError(f"acute_memory_bonus must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return
    from trainer_v2_v4 import reward_v4hg as RHG
    if getattr(RHG.score_episode, "_amb_orig", None):
        return
    orig = RHG.score_episode

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        led = orig(ep, cfg, hz)
        if reason(led, ep.item.row, check_credit=False) is not None:     # cheap criteria first
            return led
        return apply(led, ep.item.row, mode, credit_of(ep, hz, led))
    score_episode._amb_orig = orig
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    orig = getattr(RHG.score_episode, "_amb_orig", None)
    if orig is not None:
        RHG.score_episode = orig


__all__ = ["MODES", "BONUS", "COMPONENT", "STATUS", "CRITERION", "MASK", "mask_memory", "credited_actions", "credit_of",
           "reason", "qualifies", "apply", "install",
           "uninstall"]
