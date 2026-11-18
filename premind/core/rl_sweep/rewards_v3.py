"""Reward v3 (docs/REWARD_V3_ABSTENTION_PREREG_2026-09-30.md). rewards.py (v1/v2/v2.1) stays unchanged for reproducibility.

Unchanged v2 components are reused and rescaled to the v3 weights. New or changed behaviour:
  * withhold rows (decisive evidence absent from the shown context): risk `uncertain`, action abstain|clarify, cite nothing,
    plus a withhold reward for a cannot-decide statement that names the missing fact type;
  * over_abstain penalty for withholding when the target is not withhold;
  * missed_warning penalty (non-acute unsafe called safe), symmetric with false_alarm;
  * personal_content: the decisive fact's keyphrase appears in the answer with the right polarity and its record is cited.
Extra row columns: withhold (bool), decisive_fact (dict | None: fact_type, keyphrases, polarity "apply"|"avoid"), decisive_ranks.
"""
from __future__ import annotations

import os
import re

_prior_reward_version = os.environ.get("REWARD_VERSION")
os.environ["REWARD_VERSION"] = "v2"
import rewards as R2  # noqa: E402
if _prior_reward_version is None:
    del os.environ["REWARD_VERSION"]
else:
    os.environ["REWARD_VERSION"] = _prior_reward_version

assert R2.VERSION == "v2", "v3 rescales the v2 components; import with REWARD_VERSION=v2"
VERSION = "v3"
text, cited = R2.text, R2.cited
W = {"format": 0.06, "risk": 0.26, "action": 0.08, "attribution": 0.10, "twin": 0.12, "uncertain_structure": 0.04,
     "storage_content": 0.06, "personal_content": 0.12, "withhold": 0.12, "judge": 0.04}
PEN = {**R2.PEN, "missed_warning": -0.15, "over_abstain": -0.10}
WITHHOLD_ACTIONS = {"abstain", "clarify"}
CANNOT_DECIDE = re.compile(
    r"\b(can(?:no|')t (?:decide|determine|tell|know|say|answer|confirm|be sure)|unable to (?:decide|determine|tell|answer)|"
    r"(?:don't|do not) (?:know|have (?:enough|any) (?:information|record|memory|details))|not enough (?:information|context|details)|"
    r"no (?:record|memory|information) (?:of|about|on)|insufficient (?:information|evidence)|not sure (?:whether|if))", re.I)
FACT_TYPE_WORDS = {
    "allergy": r"allerg\w*|intoleran\w*",
    "diet": r"diet\w*|vegetarian|vegan|halal|kosher|avoid(?:s|ing)? \w+",
    "medical": r"medic\w*|condition|diagnos\w*|prescri\w*",
    "preference": r"prefer\w*|like[sd]?|favou?rite",
    "ownership": r"whose|belong\w*|owner|yours|who (?:it|this) is for",
    "rule": r"rule|guideline|safety (?:advice|guidance)",
    "temporal_status": r"current|valid|expired|supersed\w*|date|status",
}
_NEG = re.compile(r"\b(not|never|avoid|don't|do not|without|no)\b", re.I)


def _p(c) -> dict:
    return R2.parse(R2.text(c))


def withholds(p: dict) -> bool:
    return p["action"] in WITHHOLD_ACTIONS and bool(CANNOT_DECIDE.search(p["answer"] or ""))


def _scaled(fn, key):
    def f(completions, **kw):
        return [v * W[key] / R2.W[key] for v in fn(completions, **kw)]
    f.__name__ = f"r3_{key}"
    return f


r_format = _scaled(R2.r_format, "format")
r_twin = _scaled(R2.r_twin, "twin")
r_uncertain_structure = _scaled(R2.r_uncertain_structure, "uncertain_structure")
r_storage_content = _scaled(R2.r_storage_content, "storage_content")
r_distractor, r_false_alarm, r_ownership, r_length = R2.r_distractor, R2.r_false_alarm, R2.r_ownership, R2.r_length


def r_risk(completions, risk_level, safety_priority, withhold, **_):
    out = []
    for c, gt, sp, wh in zip(completions, risk_level, safety_priority, withhold):
        pred = _p(c)["risk"]
        if wh:
            out.append(W["risk"] * float(pred == "uncertain"))
        else:
            out.append(W["risk"] * R2.RISK_SCORE[gt].get(pred, 0.0) + (R2.PEN["critical_miss"] if sp in R2.ACUTE and pred == "safe" else 0.0))
    return out


def r_action(completions, exp_action, withhold, **_):
    return [W["action"] * float(_p(c)["action"] in WITHHOLD_ACTIONS if wh else _p(c)["action"] == e)
            for c, e, wh in zip(completions, exp_action, withhold)]


def r_attribution(completions, gt_ranks, withhold, **_):
    out = []
    for c, g, wh in zip(completions, gt_ranks, withhold):
        cs = R2.cited(_p(c))
        out.append(W["attribution"] * float((not cs) if (wh or not g) else bool(cs & set(g))))
    return out


def r_withhold(completions, withhold, decisive_fact, **_):
    """Withhold rows: cannot-decide statement + structured withhold action + names the missing fact type."""
    out = []
    for c, wh, fact in zip(completions, withhold, decisive_fact):
        p = _p(c)
        names = bool(fact) and bool(re.search(FACT_TYPE_WORDS.get(fact["fact_type"], r"$^"), p["answer"] or "", re.I))
        out.append(W["withhold"] * float(bool(wh) and withholds(p) and names))
    return out


def r_over_abstain(completions, withhold, **_):
    return [PEN["over_abstain"] * float(not wh and withholds(_p(c))) for c, wh in zip(completions, withhold)]


def r_missed_warning(completions, risk_level, safety_priority, withhold, **_):
    return [PEN["missed_warning"] * float(not wh and gt == "unsafe" and sp not in R2.ACUTE and _p(c)["risk"] == "safe")
            for c, gt, sp, wh in zip(completions, risk_level, safety_priority, withhold)]


def _applied(answer: str, phrase: str, polarity: str) -> bool:
    m = re.search(re.escape(phrase), answer, re.I)
    if not m:
        return False
    negated = bool(_NEG.search(answer[max(0, m.start() - 40):m.start()]))
    return negated if polarity == "avoid" else not negated


def r_personal_content(completions, withhold, decisive_fact, decisive_ranks, **_):
    """Decisive fact (new schema) mentioned with the right polarity AND its record cited; 0 on rows without one."""
    out = []
    for c, wh, fact, ranks in zip(completions, withhold, decisive_fact, decisive_ranks):
        if wh or not fact:
            out.append(0.0)
            continue
        p = _p(c)
        hit = any(_applied(p["answer"] or "", k, fact["polarity"]) for k in fact["keyphrases"])
        out.append(W["personal_content"] * float(hit and bool(R2.cited(p) & set(ranks or []))))
    return out


RLVR_FUNCS = [r_format, r_risk, r_action, r_attribution, r_distractor, r_twin, r_uncertain_structure, r_false_alarm,
              r_ownership, r_storage_content, r_length, r_withhold, r_over_abstain, r_missed_warning, r_personal_content]


def gtdrop(item: dict, drop: bool, ranks: set[int] | None = None) -> dict:
    """Drop GT docs by default; optionally drop only specified original ranks. Renumber the rest 1..k."""
    out = {**item, "results": [dict(r) for r in item["results"]]}
    if drop:
        out["results"] = [r for r in out["results"] if (r["rank"] not in ranks if ranks is not None else not r["is_gt"])]
        for i, r in enumerate(out["results"], 1):
            r["rank"] = i
    return out


def withhold_target(decisive: bool, gt_shown: bool) -> bool:
    """Withhold iff the item is memory-decisive and its decisive evidence is not in the shown context."""
    return decisive and not gt_shown


CUED_MISSES = frozenset({"other_person", "referential", "query_reference"})


def target(decisive: bool, decisive_shown: bool, missing_cue: str | None) -> str:
    """Owner §10 three-way target; `withhold_target` remains unchanged for r1 reproducibility."""
    if not decisive or decisive_shown:
        return "answer"
    if missing_cue in CUED_MISSES:
        return "withhold"
    if missing_cue is None or missing_cue == "none":
        return "exclude"
    raise ValueError(f"invalid missing_cue: {missing_cue!r}")
