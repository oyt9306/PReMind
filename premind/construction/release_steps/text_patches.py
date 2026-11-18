"""Generic text patches (text_qa_v1/patches.jsonl) applied by assemble.py AFTER the G0 rewrites. CPU only.

Patch row: {"id", "field": "query_text" | "gold_answer" | "context_text", "old": str, "new": str,
            "scope": "all" | [sample_ids]}.
Match = whole-value equality: row[field] == old (context_text: an element of row["context_texts"] == old). Copies
updated with it: gold_answer -> evaluator.legacy_arms.<retriever>.gold_answer (when equal to old); the memory text
(context_texts) has no other copy in the release rows (checked on v7f/v3f; the context images are photos).
A patch is all-or-nothing over every mode file of both releases; it is REFUSED when
  * scope is a list and a listed row (in any file where it exists) does not hold `old`, or a listed id is unknown;
  * nothing matches;
  * protection: a pattern of the row's evaluator (decisive_keyphrases [case-insensitive substring],
    support_patterns_by_display_index, wrong_concept_patterns, acute_visual_hazard.gold_pattern [re.search, re.I])
    that matched the old field value no longer matches the new one;
  * context_text: the stored concept name ("<Name>: ...", trainer_v2.vcm_data.concept_name) would change;
  * query_text: another row of the same file, family and query text is not patched with it (keeps the
    G1_visual / G1_visual_absent sibling link of eval_v1.topk_context, which is keyed by the query text).
Patched rows carry "text_qa_patches": [patch ids].
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FIELDS = ("query_text", "gold_answer", "context_text")


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [p.get("id") for p in out]
    if len(set(ids)) != len(ids) or None in ids:
        raise ValueError(f"{path}: missing or duplicate patch ids")
    for p in out:
        ok = (p.get("field") in FIELDS and isinstance(p.get("old"), str) and isinstance(p.get("new"), str)
              and p["old"] and p["new"].strip() and p["old"] != p["new"]
              and (p.get("scope") == "all" or (isinstance(p.get("scope"), list) and p["scope"]
                                               and all(isinstance(s, str) for s in p["scope"]))))
        if not ok:
            raise ValueError(f"{path}: malformed patch {p.get('id')!r}")
    return out


def _ev(row: dict) -> dict:
    ev = row["evaluator"]
    return json.loads(ev) if isinstance(ev, str) else ev


def patterns(row: dict) -> list[tuple[str, str]]:
    """(kind, pattern) of every label pattern stored in the row's evaluator."""
    ev = _ev(row)
    out = [("phrase", p) for p in ev.get("decisive_keyphrases") or []]
    out += [("regex", p) for p in (ev.get("support_patterns_by_display_index") or {}).values()]
    out += [("regex", p) for ps in (ev.get("wrong_concept_patterns") or {}).values() for p in ps]
    gp = (ev.get("acute_visual_hazard") or {}).get("gold_pattern")
    return out + ([("regex", gp)] if gp else [])


def _hit(kind: str, pat: str, text: str) -> bool:
    return pat.lower() in text.lower() if kind == "phrase" else bool(re.search(pat, text, re.I))


def concept_name(text: str) -> str | None:   # mirrors trainer_v2.vcm_data.concept_name
    head = text.split(":", 1)[0].strip()
    return head if ":" in text and 0 < len(head) <= 40 and len(head.split()) <= 4 else None


def _holds(row: dict, p: dict) -> bool:
    return p["old"] in row["context_texts"] if p["field"] == "context_text" else row[p["field"]] == p["old"]


def patch_row(row: dict, p: dict) -> dict:
    f, old, new = p["field"], p["old"], p["new"]
    out = {**row}
    if f == "context_text":
        out["context_texts"] = [new if t == old else t for t in row["context_texts"]]
    else:
        out[f] = new
    if f == "gold_answer" and "legacy_arms" in _ev(row):
        ev = _ev(row)
        ev = {**ev, "legacy_arms": {k: ({**a, "gold_answer": new} if a.get("gold_answer") == old else a)
                                    for k, a in ev["legacy_arms"].items()}}
        out["evaluator"] = json.dumps(ev, ensure_ascii=False) if isinstance(row["evaluator"], str) else ev
    out["text_qa_patches"] = list(row.get("text_qa_patches") or []) + [p["id"]]
    return out


def _errors(row: dict, p: dict) -> list[str]:
    errs = [f"pattern_lost:{pat[:60]}" for kind, pat in patterns(row)
            if _hit(kind, pat, p["old"]) and not _hit(kind, pat, p["new"])]
    if p["field"] == "context_text" and concept_name(p["old"]) != concept_name(p["new"]):
        errs.append(f"concept_name_changed:{concept_name(p['old'])}->{concept_name(p['new'])}")
    return errs


def apply(files: list[tuple[str, list[dict]]], patches: list[dict]) -> list[dict]:
    """Apply in order, in place on the row lists (rows are replaced, never mutated). Returns one record per patch."""
    report = []
    for p in patches:
        scope = None if p["scope"] == "all" else set(p["scope"])
        targets, errors, seen = [], [], set()
        for fi, (label, rows) in enumerate(files):
            for ri, r in enumerate(rows):
                if scope is not None and r["sample_id"] not in scope:
                    continue
                seen.add(r["sample_id"])
                if not _holds(r, p):
                    if scope is not None:
                        errors.append(f"{label}:{r['sample_id']}:old_not_found")
                    continue
                targets.append((fi, ri))
                errors += [f"{label}:{r['sample_id']}:{e}" for e in _errors(r, p)]
            if p["field"] == "query_text":
                hit = {ri for f2, ri in targets if f2 == fi}
                fams = {rows[ri]["family_id"] for ri in hit}
                errors += [f"{label}:{r['sample_id']}:same_family_query_not_in_scope" for ri, r in enumerate(rows)
                           if ri not in hit and r["family_id"] in fams and r["query_text"] == p["old"]]
        errors += [f"unknown_sample_id:{s}" for s in sorted((scope or set()) - seen)]
        if not targets:
            errors.append("old_not_found")
        if errors:
            report.append({"id": p["id"], "field": p["field"], "status": "refused", "n_errors": len(errors),
                           "errors": errors[:20]})
            continue
        for fi, ri in targets:
            files[fi][1][ri] = patch_row(files[fi][1][ri], p)
        report.append({"id": p["id"], "field": p["field"], "status": "applied", "n_rows": len(targets),
                       "rows": sorted({f"{files[fi][0]}:{files[fi][1][ri]['sample_id']}" for fi, ri in targets})})
    return report
