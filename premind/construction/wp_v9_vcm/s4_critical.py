"""R1 decision-critical facts (split out of s4_checks; re-exported there): a GT-absent partner needs a fact whose
absence makes the generic answer wrong. Pilot fix (2026-10-03): the author self-check absent_fact_decision_critical
plus a generic answer that does not already apply the fact suffice; the constraint-marker lexicon (extended with
promise / request / agreement / hand-off / damage / health words) only backs a FALSE self-check. Soft preferences
("prefers", "likes", "settles best when") never qualify.
"""
from __future__ import annotations

import re

# R1: a GT-absent partner needs a decision-critical fact (constraint, prohibition, permission, ownership, health or
# safety condition, hand-off instruction); soft preferences only feed GT-present rows.
_CRITICAL = re.compile(
    r"\b(?:never|must|only|not allowed|allowed|permission|permitted|ask(?:ed|s)?|belongs?|owns?|owned|borrowed|lent|"
    r"allerg\w*|medical|medication|asthma|diabet\w*|epilep\w*|injur\w*|fragile|broken|cracked|loose|unsafe|safety|"
    r"danger\w*|toxic|hand(?:ed|s)? (?:it |them )?(?:over|back|to)|return\w*|give (?:it |them )?back|do not|don't|"
    r"cannot|can't|no longer|forbidden|restrict\w*|sensitive|reacts?|sick|surgery|stitches|pregnan\w*|rules?|"
    r"promised?|asked|agreed|requested|needs?|keepsake|loan|owner|leak\w*|sharp|burr|seam|swell\w*|vet|doctor|"
    r"prescribed|cough\w*|trigger\w*|rash|concussion|damaged|keep (?:it|them|her|him) away)\b", re.I)
_SOFT = re.compile(r"\b(?:prefers?|preferred|likes?|liked|loves?|enjoys?|favou?rite|settles? best|best when|happiest|"
                   r"calmer|fond of)\b", re.I)


def decision_critical(fam: dict, facts: dict, item: dict, slot: str) -> dict:
    """R1 / A4: the used fact of every target is decision-critical and the generic answer without memory does not
    already apply it (so the clarify gold of a GT-absent partner is actually required)."""
    gen = (item.get("generic_answer_without_memory") or "").strip()
    bad = [] if (item.get("self_check") or {}).get("absent_fact_decision_critical") else ["self_check_false"]
    if not gen:
        bad.append("generic_answer_missing")
    for t in fam["targets"]:
        f = facts.get(t, {}).get(slot)
        if not f:
            bad.append(f"{t}:no_fact")
            continue
        if _SOFT.search(f["text"]):
            bad.append(f"{t}:soft_preference")
        if not _CRITICAL.search(f["text"]) and "self_check_false" in bad:
            bad.append(f"{t}:no_constraint_marker")      # pilot fix: the author self-check + a differing generic answer
                                                         # suffice; the marker lexicon only backs a false self-check
        if gen and (_ck().matches(f.get("application_pattern", ""), gen) or _ck().overlap_applies(f, gen)):
            bad.append(f"{t}:generic_answer_already_applies_fact")
    return _ck().ok(not bad, bad)


def _ck():  # noqa: ANN202 - lazy: s4_checks imports this module
    from premind.construction.wp_v9_vcm import s4_checks
    return s4_checks
