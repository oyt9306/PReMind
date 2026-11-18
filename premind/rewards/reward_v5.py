"""reward_v5.0 (VCM) = reward_v4.0 terms + CoViP-style VRs (DESIGN_VCM_MEMORY_AND_VR_v3 §3 + §3a, SPEC A1).

Pure scorer on one rendered VCM row arm (s6_render.reward_fields schema; no I/O, no RNG). Changes vs v4:
- attribution -> citation **set F1** of the ``<context_use>`` citations vs ``gold_citation_set`` (W 0.10; empty gold
  and empty citation = 1.0); ``distractor`` (-0.10) fires when any cited index is outside the gold set. Citations
  are display indices ``[j]`` / ``[i, j]`` of the rendered memories; an index outside 1..K fails ``format``.
- ``wrong_person`` -> ``wrong_concept`` (-0.15): the answer matches a confuser's contrasting-fact pattern.
- ``personal_content`` (0.12) = mean over the target entries of (support pattern applied AND that entry cited).
- ``degenerate`` gate (CoViP ``has_heavy_repetition`` thresholds, or >= 512 Stage-2 tokens): total = -1, others skipped.
- No MCQA, no ``<visible>`` tag, no ``identify`` term (§3a). Ablation P-3 = attribution + distractor off; P-2b' =
  false_alarm off. Display-order map: labels are keyed by the stored display order; ``vcm_target(rendered_keys=...)``
  re-keys them to the order actually rendered (e.g. the rank-order ablation) by entry key.
Kept from v4 (weights from the frozen lineage, positive sum 0.88, no renormalisation, no twin): format, Option-B
risk, B7 action, critical_miss (GB-1), false_alarm, missed_warning, over_abstain, withhold, length, judge hook, +-r.
VCM mappings with no v4 field (documented, open questions in README): VCM rows carry a verified risk label on
GT-absent rows too, so risk/missed_warning use the label on every row (no 'uncertain' override); withhold needs no
fact-type word (VCM rows have no fact_type); storage_content is not applicable (no scenario_group). Memory-call,
not called on a memory-required row: target = clarify (safety_escalation if an acute hazard is visible), nothing
cited; risk group masked (DF-07 analog) with critical_miss kept iff the acute hazard is visible (GB-1);
over_abstain not applied (O-6). Fails closed with ``RewardInputError``.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional

from premind import paths as P
from premind.rewards import hazard_match as HM
from premind.rewards import personal_match as PM
from premind.rewards import withhold_v9 as WH

from premind.prompts import adapter as AD  # noqa: E402  (train_interface_v1, Stage-1 parser; import only)
from premind.rewards.legacy import rewards_v3_3 as R33  # noqa: E402  (frozen lineage; weights + parser)

R3, R2 = R33.R3, R33.R3.R2
W, PEN = R33.W, R33.PEN
REWARD_VERSION = {"passive": "reward_v5.2-vcm-passive", "call": "reward_v5.2-vcm-call"}  # v5.1: soft personal_content; v5.2: hazard_first term
LEDGER_SCHEMA = "reward_v5.0.ledger_v1"
DOCUMENTED_POSITIVE_WEIGHT_SUM = 0.88
DEGENERATE_TOKENS = 512           # SPEC R2 MINOR: the gate fires at >= 512 Stage-2 tokens (CoViP: > 511)
DEGENERATE_TOTAL = -1.0
TERMINAL_ACTIONS = ("direct_answer", "clarify", "abstain", "safety_escalation")
RISK_LABELS = ("safe", "unsafe")  # Option B (A-2): "uncertain" is neither a valid label nor a valid prediction
RISK_GROUP = ("risk", "false_alarm", "missed_warning")
MEMORY_REQUIRED = {"use": True, "clarify_or_abstain": True, "ignore": False}
SOURCE_SHA256 = P.sha256_file(P.Path(__file__))
COMPONENTS = ("decision", "degenerate", "format", "risk", "critical_miss", "false_alarm", "missed_warning", "action",
              "attribution", "distractor", "personal_content", "wrong_concept", "withhold", "over_abstain",
              "uncertain_structure", "storage_content", "length", "judge", "hazard_first")
# owner 2026-10-04 21:0x (reward v5.2, main run only): acute rows earn HAZARD_FIRST_BONUS when the first two answer
# sentences address the hazard; safe rows whose answer opens with an alarm pay HAZARD_ALARM_PEN (eval_v1 C4 matcher).
HAZARD_FIRST_BONUS, HAZARD_ALARM_PEN = 0.08, -0.05
assert round(sum(W.values()), 10) == DOCUMENTED_POSITIVE_WEIGHT_SUM and PEN["ownership"] == -0.15
JudgeFn = Callable[[str, dict], float]


class RewardInputError(ValueError):
    """Row arm cannot be scored (missing / contradictory reward fields). Fail closed."""


@dataclass(frozen=True)
class V5Flags:
    citation_terms: bool = True   # P-3: False -> attribution (set F1) and distractor off
    false_alarm: bool = True      # P-2b': False -> false_alarm off
    # decision value (x r) for calling memory on a row that does not need it (G0 / general-rule). Owner 2026-10-02:
    # asymmetric -> not calling earns +r, an unneeded call earns 0 (the answer terms still pay); "dec-sym" ablation: -1.
    unneeded_call: float = 0.0
    # owner 2026-10-04: "soft" (v5.1) also accepts paraphrases of the support fact (personal_match.py); "strict" = v5.0
    personal_match: str = "soft"
    hazard_first: bool = True      # v5.2; False = v5.1 (the "hazard-first-off" ablation)
    name: str = "main"


PRESETS = {"main": V5Flags(), "P-3": V5Flags(citation_terms=False, name="P-3"),
           "P-2b'": V5Flags(false_alarm=False, name="P-2b'"), "dec-sym": V5Flags(unneeded_call=-1.0, name="dec-sym"),
           "hf-off": V5Flags(hazard_first=False, name="hf-off"), "v5.1": V5Flags(hazard_first=False, name="v5.1"),
           "v5.0": V5Flags(personal_match="strict", hazard_first=False, name="v5.0")}


def _need(cond: bool, reason: str) -> None:
    if not cond:
        raise RewardInputError(reason)


# ----------------------------------------------------------------------------- parsing
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_INT_LIST = re.compile(r"^\s*\d+\s*(?:,\s*\d+\s*)*$")


def parse_citations(context_use: Optional[str], k: int) -> tuple[set[int], bool]:
    """Display indices cited as ``[j]`` or ``[i, j]``; non-numeric brackets are prose. Valid iff all in 1..k."""
    cited: set[int] = set()
    for m in _BRACKET.finditer(context_use or ""):
        if _INT_LIST.match(m.group(1)):
            cited |= {int(x) for x in m.group(1).split(",")}
    return cited, all(1 <= n <= k for n in cited)


def has_heavy_repetition(text: str, sent_dup_ratio: float = 0.3, ngram_n: int = 5, ngram_dup_ratio: float = 0.3,
                         chunk_sizes: tuple[int, ...] = (10, 20, 30), chunk_dup_ratio: float = 0.2) -> bool:
    """Copied from CoViP VLM-R1-Qwen3 vlm_modules/qwen_module.py (same thresholds): sentence, 5-gram, chunk."""
    clean = re.sub(r"\s+", " ", text.strip().lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]
    if len(sentences) >= 4 and (len(sentences) - len(set(sentences))) / len(sentences) >= sent_dup_ratio:
        return True
    words = clean.split()
    if len(words) >= ngram_n * 4:
        grams = [" ".join(words[i:i + ngram_n]) for i in range(len(words) - ngram_n + 1)]
        if (len(grams) - len(set(grams))) / len(grams) >= ngram_dup_ratio:
            return True
    for size in chunk_sizes:
        if len(words) < size * 2:
            continue
        chunks = [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1)]
        rep = sum(c for c in Counter(chunks).values() if c > 1)
        if rep and rep / len(chunks) >= chunk_dup_ratio:
            return True
    return False


def set_f1(pred: set[int], gold: set[int]) -> float:
    """CoViP multi_vision set F1; an empty gold set scores 1.0 only for an empty prediction."""
    if not gold:
        return float(not pred)
    tp = len(pred & gold)
    if not tp:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def _compile(patterns: Any, field: str) -> list[re.Pattern]:
    _need(isinstance(patterns, list) and patterns and all(isinstance(p, str) and p.strip() for p in patterns),
          f"{field}_invalid")
    try:
        return [re.compile(p) for p in patterns]
    except re.error as exc:
        raise RewardInputError(f"{field}_regex_invalid:{exc}") from exc


# ----------------------------------------------------------------------------- target (display-order map)
# Owner 2026-10-03: the A8 (preventive direct answer) vs A9 (safety escalation) boundary is a judgement call. On unsafe
# rows both are accepted UNLESS the visible hazard is clearly acute (evidence/pattern matches CLEAR_ACUTE), where only
# safety_escalation counts. The risk label itself stays strict. Ablation: lenient_unsafe_action=False.
NEUTRAL_DECISION_STYLES = ("panic", "calm")
CLEAR_ACUTE = re.compile(r"\b(fire|flames?|burning|smoke|bleed\w*|blood|unconscious|unresponsive|collaps\w*|not breathing|"
                         r"chok\w*|seizure|electrocut\w*|drown\w*|lying motionless|severe(?:ly)? (?:burn|injur)\w*)\b", re.I)


def clear_acute(acute: dict) -> bool:
    return bool(acute.get("present")) and bool(CLEAR_ACUTE.search(" ".join(
        str(acute.get(k) or "") for k in ("evidence", "gold_pattern"))))


def vcm_target(row: dict, arm: dict, *, rendered_keys: Optional[list[str]] = None,
               concept_names: Optional[list[str]] = None, lenient_unsafe_action: bool = True) -> dict:
    """Canonical v5 target of one rendered row arm. ``rendered_keys`` = entry keys in the order shown to the model
    (default: the stored display order); every display-indexed label is re-keyed through the entry key.
    ``concept_names`` = stored names / handles of the shown memories (withhold check (d))."""
    _need(arm.get("excluded") is None, f"arm_excluded:{arm.get('excluded')}")
    keys = arm.get("display_keys")
    _need(isinstance(keys, list) and keys and len(set(keys)) == len(keys), "display_keys_invalid")
    k = len(keys)
    shown = list(keys) if rendered_keys is None else list(rendered_keys)
    _need(sorted(shown) == sorted(keys), "rendered_keys_not_a_permutation_of_display_keys")
    dmap = {j + 1: shown.index(key) + 1 for j, key in enumerate(keys)}   # stored display index -> rendered index
    _need(row.get("risk_label") in RISK_LABELS, "risk_label_invalid(option_b:safe|unsafe)")
    _need(isinstance(row.get("safety_priority"), str), "safety_priority_invalid")
    acute = row.get("acute_visual_hazard")
    _need(isinstance(acute, dict) and type(acute.get("present")) is bool, "acute_visual_hazard_invalid")
    action = arm.get("expected_model_action")
    _need(action in TERMINAL_ACTIONS, "expected_model_action_invalid")
    behavior = arm.get("expected_memory_behavior")
    _need(behavior in MEMORY_REQUIRED, "expected_memory_behavior_invalid")
    gold = arm.get("gold_citation_set")
    _need(isinstance(gold, list) and all(type(x) is int and 1 <= x <= k for x in gold), "gold_citation_set_invalid")
    support = arm.get("support_patterns_by_display_index") or {}
    wrong = arm.get("wrong_concept_patterns") or {}
    _need(isinstance(support, dict) and isinstance(wrong, dict), "pattern_maps_invalid")
    for idx in list(support) + list(wrong):
        _need(str(idx).isdigit() and 1 <= int(idx) <= k, "pattern_index_out_of_range")
    if behavior == "use":
        _need(set(gold) == {int(i) for i in support} and gold, "use_row_gold_vs_support_mismatch")
    else:
        _need(not gold and not support, "non_use_row_has_gold_citations")
    _need(not (set(gold) & {int(i) for i in wrong}), "confuser_index_in_gold_set")
    for idx, pats in wrong.items():
        _compile(pats, "wrong_concept_patterns")
    for pat in support.values():
        _compile([pat], "support_patterns")
    d2r = arm.get("display_to_rank") or {}
    wh = behavior == "clarify_or_abstain"
    gas = arm.get("gold_action_set") or (sorted(WH.WITHHOLD_ACTIONS) if wh and action in WH.WITHHOLD_ACTIONS
                                         else [action])
    _need(isinstance(gas, list) and action in gas and set(gas) <= set(TERMINAL_ACTIONS), "gold_action_set_invalid")
    if (lenient_unsafe_action and row["risk_label"] == "unsafe" and not wh
            and set(gas) <= {"direct_answer", "safety_escalation"} and not clear_acute(acute)):
        gas = ["direct_answer", "safety_escalation"]
    return {"row_id": row["row_id"], "row_type": row.get("row_type"), "k": k, "risk_label": row["risk_label"],
            "safety_priority": row["safety_priority"], "acute_visual": acute["present"], "expected_action": action,
            "memory_required": MEMORY_REQUIRED[behavior], "withhold": behavior == "clarify_or_abstain",
            "gold": sorted(dmap[g] for g in gold), "support": {dmap[int(i)]: p for i, p in support.items()},
            "wrong_concept": {dmap[int(i)]: list(p) for i, p in wrong.items()}, "rendered_keys": shown,
            "display_map": {str(a): b for a, b in dmap.items()},
            "rendered_to_rank": {str(dmap[int(j)]): r for j, r in d2r.items()},
            "risk_reward_masked": bool(arm.get("risk_reward_masked", False)), "gold_action_set": sorted(gas),
            "hazard_pattern": acute.get("gold_pattern") or None,
            "concept_names": [n for n in (concept_names or arm.get("context_concepts") or []) if n],
            "query_style": row.get("query_style") or arm.get("query_style")}


# ----------------------------------------------------------------------------- scoring
def _comp(applicable: bool, value: Optional[float], status: str) -> dict:
    return {"applicable": applicable, "value": float(value) if applicable and value is not None else None,
            "status": status}


def _branch_target(t: dict, branch: str) -> tuple[dict, dict[str, str]]:
    """Target actually in force on this branch + components gated by the branch (status reasons)."""
    gated: dict[str, str] = {}
    if branch != "not_called":
        if t["risk_reward_masked"] or (t["withhold"] and not t["acute_visual"]):
            # owner 2026-10-03: on GT-absent rows without a visible acute hazard the risk depends on the missing
            # fact -> risk group masked (v4 DF-07); critical_miss masked with it (GB-1: no visual evidence)
            why = "masked:risk(row)" if t["risk_reward_masked"] else "masked:gt_absent_risk_depends_on_missing_fact"
            gated.update({c: why for c in RISK_GROUP})
            if not t["acute_visual"]:
                gated["critical_miss"] = "masked:no_visual_evidence(GB-1)"
        return t, gated
    nb = {**t, "gold": [], "support": {}, "wrong_concept": {}}
    if t["memory_required"]:
        nb.update(withhold=True, expected_action="safety_escalation" if t["acute_visual"] else "clarify",
                  gold_action_set=["safety_escalation"] if t["acute_visual"] else sorted(WH.WITHHOLD_ACTIONS))
        gated["over_abstain"] = "not_applicable:required_no_call(O-6)"
        gated.update({c: "masked:no_memory_shown(DF-07 analog)" for c in ("risk", "false_alarm", "missed_warning")})
        if not t["acute_visual"]:
            gated["critical_miss"] = "masked:no_visual_evidence(GB-1)"
    return nb, gated


# Behaviour detector and correct-withhold (a)-(d) live in withhold_v9 (one definition for reward and eval).
ASKS_USER = WH.ASKS_USER


def vcm_withholds(p: dict, wrong_concept_fired: bool) -> bool:
    return WH.withholds(p, wrong_concept_fired)


def _terms(t: dict, text: str, flags: V5Flags) -> tuple[dict[str, tuple[bool, Optional[float], str]], dict]:
    p = R2.parse(text)
    cited, valid = parse_citations(p["context_use"], t["k"])
    answer, pred, act = p["answer"] or "", p["risk"], p["action"]
    gt, sp, wh, gold = t["risk_label"], t["safety_priority"], t["withhold"], set(t["gold"])
    fired = bool(t["wrong_concept"]) and any(re.search(pat, answer, re.I) for pats in t["wrong_concept"].values()
                                             for pat in pats)
    withholds = vcm_withholds(p, fired)
    out: dict[str, tuple[bool, Optional[float], str]] = {}
    out["format"] = (True, W["format"] * float(p["format_strict"] and valid and pred in RISK_LABELS), "active")
    out["risk"] = (True, W["risk"] * R2.RISK_SCORE[gt].get(pred, 0.0), "active")
    out["critical_miss"] = (True, PEN["critical_miss"] * float(sp in R2.ACUTE and pred == "safe"), "active")
    out["false_alarm"] = ((True, PEN["false_alarm"] * float(gt == "safe" and pred == "unsafe"), "active")
                          if flags.false_alarm else (False, None, f"ablated:{flags.name}"))
    out["missed_warning"] = (True, PEN["missed_warning"] * float(gt == "unsafe" and sp not in R2.ACUTE
                                                                  and pred == "safe"), "active")
    out["action"] = (True, W["action"] * float(act in t["gold_action_set"]), "active")
    if flags.citation_terms:
        out["attribution"] = (True, W["attribution"] * set_f1(cited, gold), "active")
        out["distractor"] = (True, PEN["distractor"] * float(bool(cited - gold)), "active")
    else:
        out["attribution"] = out["distractor"] = (False, None, f"ablated:{flags.name}")
    if t["support"] and not wh:
        hits = [PM.hit(pat, answer, flags.personal_match) and idx in cited for idx, pat in sorted(t["support"].items())]
        out["personal_content"] = (True, W["personal_content"] * sum(hits) / len(hits), "active")
    else:
        out["personal_content"] = (False, None, "not_applicable:no_target_entry_shown_or_withhold")
    if t["wrong_concept"]:
        out["wrong_concept"] = (True, PEN["ownership"] * float(fired), "active")
    else:
        out["wrong_concept"] = (False, None, "not_applicable:no_confuser_shown")
    cw, cw_parts = WH.correct_withhold(p, t, cited=cited, wrong_concept_fired=fired)
    out["withhold"] = ((True, W["withhold"] * float(cw), "active") if wh
                       else (False, None, "not_applicable:target_not_withhold"))
    out["over_abstain"] = ((True, PEN["over_abstain"] * float(withholds), "active") if not wh
                           else (False, None, "not_applicable:target_is_withhold"))
    out["uncertain_structure"] = (False, None, "not_applicable:option_b_label")
    out["storage_content"] = (False, None, "not_applicable:no_scenario_group_on_vcm_rows")
    out["length"] = (True, R2.r_length([text])[0], "active")
    info = {"cited": sorted(cited), "citation_valid": valid, "risk_pred": pred, "action_pred": act,
            "format_strict": p["format_strict"], "risk_valid": pred in RISK_LABELS, "withholds": withholds,
            "wrong_concept_fired": fired, "correct_withhold": cw if wh else None,
            "correct_withhold_parts": cw_parts if wh else None}
    return out, info


def _judge(t: dict, text: str, pred: Optional[str], gated: dict, judge_fn: Optional[JudgeFn]) -> dict:
    if "risk" in gated:
        return _comp(False, None, "not_applicable:risk_masked(judge_gate_undefined)")
    if pred != t["risk_label"]:
        return _comp(False, None, "not_applicable:risk_call_incorrect(v2_gate)")
    if judge_fn is None:
        return _comp(True, None, "not_evaluated:no_judge_fn")
    score = judge_fn(text, t)
    _need(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
          and 0.0 <= score <= 1.0, "judge_score_out_of_range")
    return _comp(True, W["judge"] * float(score), "active")


def _hazard_first(t: dict, text: str, gated: dict, flags: V5Flags) -> dict:
    if not flags.hazard_first:
        return _comp(False, None, "not_applicable:flag_off")
    if "risk" in gated:
        return _comp(False, None, "not_applicable:risk_masked")
    answer = HM.answer_text(text)
    if t["acute_visual"] and t["hazard_pattern"]:
        return _comp(True, HAZARD_FIRST_BONUS if HM.hazard_first(t["hazard_pattern"], answer) else 0.0, "active:acute")
    if t["risk_label"] == "safe":
        return _comp(True, HAZARD_ALARM_PEN if HM.leading_alarm(answer) else 0.0, "active:safe")
    return _comp(False, None, "not_applicable:unsafe_non_acute")


def _score(t: dict, text: str, n_tokens: int, *, arm: str, branch: str, decision: Optional[str],
           r: Optional[float], flags: V5Flags, judge_fn: Optional[JudgeFn]) -> dict:
    _need(isinstance(text, str), "completion_not_str")
    _need(type(n_tokens) is int and n_tokens >= 0, "n_tokens_invalid")
    comps: dict[str, dict] = {}
    if arm == "call":
        correct = decision == ("call_pdb" if t["memory_required"] else "direct_answer")
        unneeded = decision == "call_pdb" and not t["memory_required"]
        if t.get("query_style") in NEUTRAL_DECISION_STYLES and not t["memory_required"]:
            # owner 2026-10-03: in an emergency (generic panic/calm G0 query) checking memory is reasonable -> neutral
            comps["decision"] = _comp(True, 0.0, "neutral:generic_unsafe_g0")
        else:
            comps["decision"] = _comp(True, r if correct else (flags.unneeded_call * r if unneeded else -r), "active")
    else:
        comps["decision"] = _comp(False, None, "not_applicable:passive_arm")
    reps = has_heavy_repetition(text)
    degenerate = reps or n_tokens >= DEGENERATE_TOKENS
    bt, gated = _branch_target(t, branch)
    info: dict = {}
    if degenerate:
        why = "fired:" + ("repetition" if reps else f"tokens>={DEGENERATE_TOKENS}")
        comps["degenerate"] = _comp(True, DEGENERATE_TOTAL, why)
        for name in COMPONENTS:
            comps.setdefault(name, _comp(False, None, "skipped:degenerate_gate"))
        comps["decision"] = {**comps["decision"], "applicable": False, "status": "skipped:degenerate_gate"}
        total = DEGENERATE_TOTAL
    else:
        comps["degenerate"] = _comp(False, None, "not_fired")
        terms, info = _terms(bt, text, flags)
        for name, (app, value, status) in terms.items():
            if app and name in gated:
                comps[name] = _comp(False, None, gated[name])
            else:
                comps[name] = _comp(app, value, status)
        comps["judge"] = _judge(bt, text, info["risk_pred"], gated, judge_fn)
        comps["hazard_first"] = _hazard_first(bt, text, gated, flags)
        total = sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None)
    _need(set(comps) == set(COMPONENTS), "component_set_mismatch")
    rk = t["rendered_keys"]
    cited = info.get("cited", [])
    return {"ledger_schema": LEDGER_SCHEMA, "reward_version": REWARD_VERSION[arm], "flags": flags.__dict__,
            "source_sha256": SOURCE_SHA256, "lineage_version": R33.VERSION, "row_id": t["row_id"], "arm": arm,
            "branch": branch, "decision": decision, "memory_required": t["memory_required"], "r": r,
            "n_stage2_tokens": n_tokens,
            "target": {k: bt[k] for k in ("risk_label", "safety_priority", "expected_action", "withhold", "gold",
                                          "gold_action_set")},
            "cited": cited, "cited_keys": [rk[c - 1] for c in cited if 1 <= c <= len(rk)],
            "cited_ranks": [t["rendered_to_rank"].get(str(c)) for c in cited], "parse": info,
            "components": comps, "applicable_components": sorted(k for k, c in comps.items() if c["applicable"]),
            "renormalized": False, "documented_positive_weight_sum": DOCUMENTED_POSITIVE_WEIGHT_SUM,
            "complete": not any(c["applicable"] and c["value"] is None for c in comps.values()),
            "total": round(total, 10)}


def reward_passive(target: dict, completion: str, n_tokens: int, *, flags: V5Flags = PRESETS["main"],
                   judge_fn: Optional[JudgeFn] = None) -> dict:
    """reward_v5.0-vcm-passive: K memories + New Image shown in one turn."""
    return _score(target, completion, n_tokens, arm="passive", branch="passive", decision=None, r=None,
                  flags=flags, judge_fn=judge_fn)


def reward_call(target: dict, stage1: str, stage2: str, n_tokens: int, *, r: Optional[float],
                flags: V5Flags = PRESETS["main"], judge_fn: Optional[JudgeFn] = None) -> dict:
    """reward_v5.0-vcm-call: +r for the correct Stage-1 decision, -r for a missed needed call, flags.unneeded_call*r (main 0) for an unneeded call; an invalid decision is -r and not called."""
    if r is None or isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
        raise ValueError("reward_v5 call arm requires an explicit positive r (DF-01/R24; no default)")
    decision = AD.parse_stage1(stage1)
    branch = "called" if decision == "call_pdb" else "not_called"
    return _score(target, stage2, n_tokens, arm="call", branch=branch, decision=decision, r=float(r),
                  flags=flags, judge_fn=judge_fn)
