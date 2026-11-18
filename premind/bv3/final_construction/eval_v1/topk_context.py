"""Test-time context-size (top-K) ablation on immutable release rows (CPU only; owner 2026-10-05).

The retrieved context of a release row holds the retriever's top-8 entries in a seeded display order
(``evaluator.display_to_rank``: display index -> retriever rank, rank 1 = best). ``truncate`` keeps the K entries
with the best rank, in their original display order, renumbered 1..K, and re-keys every display-indexed label
(``gold_citations``, ``support_patterns_by_display_index``, ``wrong_concept_patterns``, ``display_to_rank``) on the
RELEASE ROW, so ``release_data.convert`` renders and ``reward_v5.vcm_target`` builds the target exactly as for a
K-entry release row (``display_map`` / ``rendered_to_rank`` follow). ``vcm_target(rendered_keys=...)`` cannot be
reused directly: it re-keys permutations of the stored keys only (``rendered_keys_not_a_permutation``).

Label rule when truncation drops targets (the release rule, wp_v9_vcm/s4_rows.arm_labels):
  * GT-present ``use`` row, every gold target kept -> unchanged labels (``target_retained`` True);
  * no gold target kept (retrieval miss at inference) -> GT-absent iff the family has the constructed GT-absent
    sibling (same family, ``<row_type>_absent``, same query text; the sibling exists only when A4 / decision-critical
    held at build time): action / action set / B7 subtype / risk / reference gold answer / memory behaviour come
    from the sibling; the context stays the truncated one; wrong-concept patterns of the kept confusers stay
    (s4 rule: a GT-absent row keeps the confusers of its own context). No sibling -> excluded;
  * two-target row with one target kept -> excluded (``partial_retrieval_two_concept``, as at release build);
  * GT-absent and G0 / no-memory rows keep their labels; only the context is truncated.
K >= the row's context size returns the row object unchanged (K=8 is byte-identical to the default path).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)
CONTEXT_KS = (1, 2, 4, 8)
FULL_K = 8
SIBLING = {"G1_visual": "G1_visual_absent", "G1G2_visual": "G1G2_visual_absent"}
# release-row (top-level) label fields copied from the GT-absent sibling on relabel
ROW_LABELS = ("gold_action", "gold_action_set", "b7_subtype", "gold_answer", "reference_only", "risk")
EV_LABELS = ("expected_memory_behavior", "decisive_keyphrases", "safety_priority")
SibKey = tuple[str, str, str]


def evaluator(row: dict) -> dict:
    ev = row["evaluator"]
    return json.loads(ev) if isinstance(ev, str) else ev


def check_k(k: int) -> int:
    if k not in CONTEXT_KS:
        raise ValueError(f"context_k must be one of {CONTEXT_KS}, got {k!r}")
    return k


def sibling_index(rows: list[dict]) -> dict[SibKey, dict]:
    """(family_id, row_type, query_text) -> release row; the key is unique in the v7f / v3f releases."""
    out: dict[SibKey, dict] = {}
    for r in rows:
        key = (r["family_id"], r["row_type"], r["query_text"])
        if key in out:
            raise ValueError(f"duplicate sibling key {key}")
        out[key] = r
    return out


def kept_display(ev: dict, n: int, k: int) -> list[int]:
    """Original display indices (1-based) of the K best-ranked entries, in display order."""
    d2r = {int(d): int(r) for d, r in ev["display_to_rank"].items()}
    if sorted(d2r) != list(range(1, n + 1)) or sorted(d2r.values()) != list(range(1, n + 1)):
        raise ValueError("display_to_rank is not a bijection on 1..n")
    return sorted(sorted(d2r, key=d2r.__getitem__)[:k])


def _rekey(mapping: dict, new: dict[int, int]) -> dict:
    return {str(new[int(i)]): v for i, v in mapping.items() if int(i) in new}


def _info(k: int, retained: Optional[bool], relabelled: bool = False, excluded: Optional[str] = None,
          source: Optional[str] = None) -> dict:
    return {"context_k": k, "target_retained": retained, "relabelled": relabelled,
            "topk_excluded": excluded is not None, "topk_exclusion": excluded, "relabel_source": source}


def _relabel(row: dict, ev: dict, sib: dict) -> tuple[dict, dict]:
    sev = evaluator(sib)
    if sev.get("expected_memory_behavior") != "clarify_or_abstain" or not sev.get("accepted"):
        raise ValueError("sibling_not_an_accepted_gt_absent_row")
    if sev["acute_visual_hazard"]["present"] != ev["acute_visual_hazard"]["present"]:
        raise ValueError("sibling_acute_mismatch")
    row = {**row, **{f: sib.get(f) for f in ROW_LABELS}, "gold_citations": [], "context_condition": "gt_absent"}
    ev = {**ev, **{f: sev.get(f) for f in EV_LABELS}, "support_patterns_by_display_index": {},
          "absent_reason": "retrieval_miss_topk"}
    return row, ev


def truncate(row: dict, k: int, siblings: dict[SibKey, dict]) -> tuple[Optional[dict], dict]:
    """(row restricted to its top-K context with re-keyed labels | None if excluded, per-row info)."""
    check_k(k)
    ev = evaluator(row)
    use = ev.get("expected_memory_behavior") == "use"
    n = len(row["contexts"])
    if k >= n:
        return row, _info(k, True if use else None)
    keep = kept_display(ev, n, k)
    new = {old: j + 1 for j, old in enumerate(keep)}
    gold = [int(g) for g in row["gold_citations"]]
    kept_gold = [g for g in gold if g in new]
    retained: Optional[bool] = None
    relabelled, source = False, None
    if use:
        retained = len(kept_gold) == len(gold)
        if kept_gold and not retained:
            return None, _info(k, False, excluded="partial_retrieval_two_concept")
    ev = {**ev, "context_concepts": [ev["context_concepts"][i - 1] for i in keep],
          "display_to_rank": {str(new[int(d)]): r for d, r in ev["display_to_rank"].items() if int(d) in new},
          "support_patterns_by_display_index": _rekey(ev.get("support_patterns_by_display_index") or {}, new),
          "wrong_concept_patterns": _rekey(ev.get("wrong_concept_patterns") or {}, new)}
    out = {**row, "contexts": [row["contexts"][i - 1] for i in keep],
           "context_texts": [row["context_texts"][i - 1] for i in keep],
           "gold_citations": sorted(new[g] for g in kept_gold), "k": k}
    if use and not retained:
        sib_type = SIBLING.get(row["row_type"])
        sib = siblings.get((row["family_id"], sib_type, row["query_text"])) if sib_type else None
        if sib is None:
            return None, _info(k, False, excluded="no_absent_sibling")
        try:
            out, ev = _relabel(out, ev, sib)
        except ValueError as exc:
            return None, _info(k, False, excluded=str(exc))
        relabelled, source = True, sib["sample_id"]
    out["evaluator"] = ev
    return out, _info(k, retained, relabelled, source=source)


def record_fields(item_row: dict, target: dict, k: int) -> dict:
    """Per-record top-K fields; items of the unchanged K=8 path carry none, so derive them from the target."""
    if "context_k" in item_row:
        out = {f: item_row[f] for f in ("context_k", "target_retained", "relabelled", "topk_excluded")}
        for f in ("online_k", "absent_case", "online_equals_stored_topk"):    # online path only (v4)
            if f in item_row:
                out[f] = item_row[f]
        return out
    use = bool(target.get("memory_required")) and not target.get("withhold")
    return {"context_k": k, "target_retained": True if use else None, "relabelled": False, "topk_excluded": False}


def label_counts(rows: list[dict], k: int) -> dict:
    """CPU-only relabel / exclusion counts of one release split file at K (accepted rows with a target only)."""
    check_k(k)
    acc = [r for r in rows if evaluator(r).get("accepted") and evaluator(r).get("expected_memory_behavior")]
    sibs = sibling_index(rows)
    c = {"k": k, "n_rows": len(acc), "gt_present_use": 0, "retained": 0, "relabelled": 0, "excluded": {},
         "risk_changed_by_relabel": 0, "unchanged_labels_non_use": 0}
    for r in acc:
        out, info = truncate(r, k, sibs)
        if evaluator(r)["expected_memory_behavior"] != "use":
            c["unchanged_labels_non_use"] += 1
            continue
        c["gt_present_use"] += 1
        if info["topk_excluded"]:
            c["excluded"][info["topk_exclusion"]] = c["excluded"].get(info["topk_exclusion"], 0) + 1
        elif info["relabelled"]:
            c["relabelled"] += 1
            c["risk_changed_by_relabel"] += int(out["risk"] != r["risk"])
        else:
            c["retained"] += 1
    return c
