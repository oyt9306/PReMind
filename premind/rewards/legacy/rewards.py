"""Decomposed RLVR rewards for the RL sweep (owner spec, 2026-09-25). Every component is its own TRL reward function so
TRL logs its mean/std separately. Row columns used: risk_level, safety_priority, exp_action, scenario, gt_ranks
(1-based ranks of the item's reference memory inside the shown top-5), is_twin, storage_discard (food_storage rows only).
Weights sum to 1 over the positive components; penalties are negative.
"""
from __future__ import annotations

import os
import re

from premind.rewards.legacy.baseline_eval.evaluate import ACUTE, RISK_SCORE, parse  # noqa: E402

# REWARD_VERSION (env): v1 = POC runs (judge 0.20, ungated; false alarm = unsafe-on-safe only); v2 (owner, 2026-09-26,
# after the POC judge audit: the RL judge correlates only 0.29 with deterministic correctness and the eval judge scores an
# alarmed answer on a safe twin higher than a correct one) = judge 0.10 and only on rollouts whose risk call is right,
# plus a half penalty for `uncertain` on the safe member of a counterfactual twin.
VERSION = os.environ.get("REWARD_VERSION", "v2")
# v2.1 retains v2 weights and components; it only fixes the uncertainty-structure overlap (2026-09-27).
# Deterministic signals dominate; the judge is a bounded 0.05 tie-breaker on correct-risk rollouts only.
if VERSION == "v1":
    W = {"format": 0.08, "risk": 0.32, "action": 0.12, "attribution": 0.10, "twin": 0.06, "uncertain_structure": 0.04,
         "storage_content": 0.08, "judge": 0.20}
else:
    W = {"format": 0.08, "risk": 0.32, "action": 0.12, "attribution": 0.10, "twin": 0.12, "uncertain_structure": 0.06,
         "storage_content": 0.08, "judge": 0.05}
PEN = {"distractor": -0.10, "false_alarm": -0.15, "ownership": -0.15, "critical_miss": -0.5, "length": -0.1}
MAX_ANSWER_CHARS = 1200
_CHECK = re.compile(r"\b(check|confirm|verify|look (?:at|for)|make sure|ask)\b", re.I)
_UNSURE = re.compile(r"\b(can(?:no|')t (?:tell|confirm|see)|not (?:sure|clear|certain)|unclear|may|might|possibl[ey]|could)\b", re.I)
_CANNOT_CONFIRM = re.compile(
    r"\b(can(?:no|')t (?:tell|confirm|see|verify|know|determine)|impossible to (?:tell|confirm|verify|know|determine)|"
    r"not (?:sure|clear|certain|visible|shown|known)|unclear|unknown)\b", re.I)
_POSSIBLE_RISK = re.compile(r"\b(possib\w*|might|may|could|potential\w*)\b[^.]{0,80}\b(risk|hazard|danger\w*|unsafe|allerg\w*|"
                            r"burn|fire|scald|fall|trip|spill|cut|sharp|toxic|poison\w*|shock|electri\w*|choke|sick)", re.I)
_DISCARD = re.compile(r"\b(throw (?:it |them )?(?:away|out)|discard|toss|bin it|don't eat|do not eat|not safe to eat)\b", re.I)


def text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


def cited(p: dict) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d)\]", p["context_use"] or "")}


def r_format(completions, **_):
    return [W["format"] * float(parse(text(c))["format_strict"]) for c in completions]


def r_risk(completions, risk_level, safety_priority, **_):
    out = []
    for c, gt, sp in zip(completions, risk_level, safety_priority):
        pred = parse(text(c))["risk"]
        out.append(W["risk"] * RISK_SCORE[gt].get(pred, 0.0) + (PEN["critical_miss"] if sp in ACUTE and pred == "safe" else 0.0))
    return out


def r_action(completions, exp_action, **_):
    return [W["action"] * float(parse(text(c))["action"] == e) for c, e in zip(completions, exp_action)]


def r_attribution(completions, gt_ranks, **_):
    """Rows whose reference memory was retrieved: cite it. Rows without one: cite nothing retrieved."""
    out = []
    for c, g in zip(completions, gt_ranks):
        cs = cited(parse(text(c)))
        out.append(W["attribution"] * float(bool(cs & set(g)) if g else not cs))
    return out


def r_distractor(completions, gt_ranks, **_):
    return [PEN["distractor"] * float(bool(cited(parse(text(c))) - set(g))) for c, g in zip(completions, gt_ranks)]


_PAIR_CACHE: dict[str, float] = {}  # sample_id -> correct-rate of its latest rollout group (pair-aware twin reward)


def _gather(obj):
    try:
        import torch.distributed as dist
    except ImportError:  # CPU-only test env
        return list(obj)
    if dist.is_available() and dist.is_initialized():
        out = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return [x for part in out for x in part]
    return list(obj)


def r_twin(completions, is_twin, risk_level, sample_id=None, partner_id=None, **_):
    """v1: per-member correctness bonus. v2 (pair-aware): a twin rollout earns the bonus only when it is correct AND its
    partner's rollouts are correct -- the partner's correct-rate comes from the same step (v2 ordering puts both members
    in one step on different processes; exchanged with all_gather) or, failing that, its latest earlier step."""
    ok = [parse(text(c))["risk"] == gt for c, gt in zip(completions, risk_level)]
    if VERSION == "v1" or sample_id is None:
        return [W["twin"] * float(t and o) for t, o in zip(is_twin, ok)]
    rate = {}
    for sid, o in zip(sample_id, ok):
        rate.setdefault(sid, []).append(o)
    for sid, r in _gather([(k, sum(v) / len(v)) for k, v in rate.items()]):
        _PAIR_CACHE[sid] = r
    return [W["twin"] * float(t and o and bool(p)) * _PAIR_CACHE.get(p, 0.0)
            for t, o, p in zip(is_twin, ok, partner_id or [None] * len(ok))]


def r_uncertain_structure(completions, risk_level, **_):
    """Uncertain rows: the answer hedges AND names a concrete check. Applied only on uncertain rows (so it cannot be
    farmed on safe/unsafe rows, where the risk table already penalizes an uncertain call)."""
    out = []
    for c, gt in zip(completions, risk_level):
        a = parse(text(c))["answer"]
        parts = [(_UNSURE if VERSION == "v1" else _CANNOT_CONFIRM).search(a), _CHECK.search(a)]
        if VERSION != "v1":
            parts.append(_POSSIBLE_RISK.search(a))
        out.append(W["uncertain_structure"] * float(gt == "uncertain" and all(parts)))  # v2: possible risk + cannot confirm + check
    return out


def r_false_alarm(completions, risk_level, is_twin=None, **_):
    """Safe scene called unsafe. v2: twin members are scored by r_ownership instead (logged separately)."""
    out = []
    for i, (c, gt) in enumerate(zip(completions, risk_level)):
        twin = VERSION != "v1" and is_twin is not None and is_twin[i]
        out.append(PEN["false_alarm"] * float(gt == "safe" and not twin and parse(text(c))["risk"] == "unsafe"))
    return out


def r_ownership(completions, risk_level, is_twin=None, **_):
    """v2: the safe member of a counterfactual twin -- the memory's constraint belongs to someone else (or was removed).
    Treating it as the user's own (unsafe) is penalized fully, hedging on it (uncertain) by half."""
    if VERSION == "v1" or is_twin is None:
        return [0.0] * len(completions)
    out = []
    for c, gt, t in zip(completions, risk_level, is_twin):
        pred = parse(text(c))["risk"]
        out.append(PEN["ownership"] * (1.0 if pred == "unsafe" else 0.5 if pred == "uncertain" else 0.0) * float(t and gt == "safe"))
    return out


def r_storage_content(completions, scenario, storage_discard, **_):
    """Crossed food_storage rows: the answer must say to discard exactly when the memory makes the food unsafe."""
    out = []
    for c, sc, d in zip(completions, scenario, storage_discard):
        if sc != "food_storage_care" or d is None:
            out.append(0.0)
            continue
        says = bool(_DISCARD.search(parse(text(c))["answer"]))
        out.append(W["storage_content"] * float(says == d))
    return out


def r_length(completions, **_):
    return [PEN["length"] * float(len(parse(text(c))["answer"]) > MAX_ANSWER_CHARS) for c in completions]


RLVR_FUNCS = [r_format, r_risk, r_action, r_attribution, r_distractor, r_twin, r_uncertain_structure, r_false_alarm,
              r_ownership, r_storage_content, r_length]
