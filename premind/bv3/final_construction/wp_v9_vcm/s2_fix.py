"""Targeted repair of the FINAL units (owner 2026-10-03: acute family consistency, unsafe-G0 redesign, general +
personal safety response; audit 10b / 5.3b): pass 1 = round 3, pass 2 = round 4 (`--round 3|4 --retry`, cm.FIX_ROUNDS).

Pass 1 (round 3): affected rows = rows of s5_final_r2 failing S1-S6 (s4_acute). Pass 2 (round 4, wave-1 fix
2026-10-03): affected rows = the round-3 affected rows that were NOT accepted (s5_final_r3 dropped_affected_rows);
round-3 accepted rows are never re-paid. Per family:
  * prev = the family's latest output (pass 1: its base round; pass 2: author_rev3 when valid, else the base), with
    the LABELS NORMALIZED in code (normalize: acute family -> every item acute, unsafe, safety_escalation A9;
    non-acute unsafe -> direct_answer A8; safe -> direct_answer A1/A2; GT-absent hazard pattern rebuilt name-free
    from the visible hazard). GPT only ever writes text.
  * failures = the item-level hard precheck failures of the normalized prev for the affected items (+ the S5 order /
    personal-action failures of the affected rows). None -> the normalized prev is copied with NO call ("copy").
  * else ONE text-only delta on those items (facts are never re-sent: memory texts stay byte-identical, memtext reuses
    the base texts, s2_memtext.reuse_plan uses base_round), labels normalized again, precheck; ONE extra text-only
    attempt when every remaining hard failure is item-level (e.g. S4 names, S5 openers).
Only affected rows are reviewed (s5_review._rows, text-only check + Qwen with images); s5_fix selects per row
(pass 2 > pass 1 > dropped). The unit's plan (s2/author_rev<N>/FIX_PLAN[_vK].json, write-once, versioned) is its
start marker: before it exists the pass is a NEW run (open stop / start projection > cm.PROD_BUDGET_STOP refuse it);
a stored plan that the current code no longer computes is superseded by a new version only if no revision exists yet.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import gpt_client as gc
from final_construction.wp_v9_vcm import llm
from final_construction.wp_v9_vcm.s2_fix_text import (TEXT_DELTA, TEXT_PROBLEMS, run_text_review,  # noqa: F401
                                                      text_family_estimate, text_only)

logger = logging.getLogger(__name__)
PLAN = "FIX_PLAN"
START_STOP = "PROD_STOP_fix_start.json"
ITEM_TYPES = ("G1_visual", "G1G2_visual", "G0_closed")
BASE_ITEM = {"G1_visual": "G1_visual", "G1_visual_absent": "G1_visual", "G1G2_visual": "G1G2_visual",
             "G1G2_visual_absent": "G1G2_visual", "G0_closed": "G0_closed"}
# image-based comparison model (ledger fits 2026-10-03; the rounds run text-only, s2_fix_text)
DELTA_TOKENS, REVIEW_FIXED, REVIEW_PER_ROW = 5_004, 5_077, 854
ADJ_FIXED, ADJ_PER_ROW, ADJ_RATE = 5_278, 1_147, 0.35


def _version(p) -> int:  # noqa: ANN001
    m = re.search(r"_v(\d+)\.json$", p.name)
    return int(m.group(1)) if m else 1


def plan_files(cfg: cm.Cfg) -> list:
    d = cfg.p("s2", f"author_rev{cfg.round}")
    return sorted(d.glob(f"{PLAN}*.json"), key=_version) if d.exists() else []


def plan_path(cfg: cm.Cfg):  # noqa: ANN201
    """The active (latest) plan of this pass, or the default first path."""
    files = plan_files(cfg)
    return files[-1] if files else cfg.p("s2", f"author_rev{cfg.round}", f"{PLAN}.json")


def _plan(cfg: cm.Cfg) -> dict | None:
    p = plan_path(cfg)
    return cm.read(p) if cm.is_fix(cfg) and p.exists() else None


def base_round(cfg: cm.Cfg, fid: str) -> int | None:
    doc = _plan(cfg)
    return (doc["families"].get(fid) or {}).get("base_round") if doc else None


def affected_row_ids(cfg: cm.Cfg) -> set[str] | None:
    doc = _plan(cfg)
    return {r for d in doc["families"].values() for r in d["affected_rows"]} if doc else None


def normalize(fam: dict, out: dict | None) -> dict | None:
    """Labels from the rules (contract_v1 B7 cells), never from GPT; text untouched except the absent hazard pattern,
    rebuilt name-free from the visible hazard when the family is acute."""
    from final_construction.wp_v9_vcm import s4_acute as ac
    if not out:
        return out
    items = [dict(i) for i in out.get("items", [])]
    fa = ac.family_acute([ac.view(i) for i in items])
    banned = {w.lower() for n in ac.labels(fam) for w in ac.words(n)}
    def hazard_pattern(text: str) -> str | None:
        """Name-free pattern of the hazard terms in the text's first sentence (None if it names no hazard)."""
        first = ac.first_sentence(text)
        strong = {m.group(0).lower() for rx in (ac.LEXICON, ac.RESPONSE, ac.SAFETY_ACTION) for m in rx.finditer(first)}
        weak = set().union(*(ac.specific_terms(e) for e in fa["evidence"])) & set(ac.words(first))
        terms = sorted((strong or weak) - banned, key=lambda t: (-len(t), t))   # hazard/safety terms first
        return "(?i)\\b(?:" + "|".join(re.escape(t) for t in terms) + ")\\b" if terms else None

    def matches(p: str, text: str) -> bool:
        try:
            return bool(p) and re.search(p, text or "") is not None
        except re.error:
            return False
    for it in items:
        if it["row_type"] not in ITEM_TYPES:
            continue
        if fa["present"]:
            it.update(acute_visual_hazard_present=True, risk="unsafe", expected_model_action="safety_escalation",
                      b7_subtype="A9")
            it["acute_visual_hazard_evidence"] = it.get("acute_visual_hazard_evidence") or (fa["evidence"] or [""])[0]
            if not matches(it.get("acute_gold_pattern", ""), ac.first_sentence(it.get("gold_answer") or "")):
                it["acute_gold_pattern"] = hazard_pattern(it.get("gold_answer") or "") or it.get("acute_gold_pattern", "")
            absent = it.get("gt_absent_response") or ""
            if it["row_type"] != "G0_closed" and absent and hazard_pattern(absent):
                it["acute_absent_gold_pattern"] = hazard_pattern(absent)
        elif it.get("risk") == "unsafe":
            it.update(expected_model_action="direct_answer", b7_subtype="A8")
        else:
            it.update(expected_model_action="direct_answer", b7_subtype="A1" if it["row_type"] == "G0_closed" else "A2")
    return {**out, "items": items}


def _prev(cfg: cm.Cfg, fam: dict, d: dict) -> dict | None:
    from final_construction.wp_v9_vcm import s2_author
    fid = fam["family_id"]
    p = s2_author.author_path(cfg, fid, d["prev_round"]) if d["prev_round"] > 2 else None
    out = cm.read(p)["output"] if p is not None and p.exists() and cm.read(p)["valid"] else \
        s2_author.load_author(cfg, fid, max_round=d["base_round"])["output"]
    return normalize(fam, out)


def _failures(fam: dict, prev: dict | None, items: set[str], rows: list[dict]) -> list[dict]:
    """Item-level hard failures of the normalized prev for the affected items + S5 row failures (order / personal)."""
    from final_construction.wp_v9_vcm import s2_revise, s4_acute as ac
    fails = [f for f in s2_revise.hard(s2_revise.precheck_output(fam, prev)) if f["item"] in items]
    have = {(f["item"], f["check"]) for f in fails}
    for r in rows:
        res = ac.row_results(fam, [r], (prev or {}).get("items", []))[r["row_id"]][ac.S5]
        key = (BASE_ITEM[r["row_type"]], ac.S5)
        if not res["pass"] and key not in have:
            fails.append({"item": key[0], "check": ac.S5, "detail": {"problems": res["detail"]}})
            have.add(key)
    return fails


def compute_plan(cfg: cm.Cfg) -> dict:
    """CPU, no write: affected rows, base/prev rounds and failures of one pass (round 3 or 4) of a unit."""
    from final_construction.wp_v9_vcm import s2_author, s4_acute
    fams = {f["family_id"]: f for f in cm.plan_raw(cfg)["families"]}
    rows_by: dict[str, list[dict]] = defaultdict(list)
    if cfg.round == cm.FIX_ROUNDS[0]:
        rows = [r for r in cm.read(cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final_r2.json"))["rows"]
                if r["row_type"] not in cm.DROPPED_ROW_TYPES]
        src = cm.read(cfg.p("s5_final_r2", "gate.json"))["family_source_round"]
        for r in rows:
            rows_by[r["family_id"]].append(r)
        cand = {}
        for fid, rs in rows_by.items():
            a = s2_author.load_author(cfg, fid, max_round=src.get(fid, 1))
            if not a["valid"]:
                continue
            res = s4_acute.row_results(fams[fid], rs, a["output"]["items"])
            bad = sorted(rid for rid, d in res.items() if any(not v["pass"] for v in d.values()))
            if bad:
                cand[fid] = {"base_round": src.get(fid, 1), "prev_round": src.get(fid, 1), "affected_rows": bad}
    else:
        prev_cfg = cm.Cfg(**{**cfg.__dict__, "round": cfg.round - 1})
        dropped = cm.read(cfg.p(f"s5_final_r{cfg.round - 1}", "gate.json"))["dropped_affected_rows"]
        pplan = cm.read(plan_path(prev_cfg))["families"]
        pr = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_r{cfg.round - 1}.json")
        for r in cm.read(pr)["rows"] if pr.exists() else []:
            rows_by[r["family_id"]].append(r)
        cand = {}
        for rid in dropped:
            fid = rid.split(":")[1] + ":" + rid.split(":")[2]
            d = cand.setdefault(fid, {"base_round": pplan[fid]["base_round"], "prev_round": cfg.round - 1,
                                      "affected_rows": []})
            d["affected_rows"].append(rid)
    out, counts = {}, defaultdict(int)
    for fid, d in sorted(cand.items()):
        fam = fams[fid]
        prev = _prev(cfg, fam, d)
        items = sorted({BASE_ITEM[r.rsplit(":", 1)[1]] for r in d["affected_rows"]})
        rows = [r for r in rows_by.get(fid, []) if r["row_id"] in set(d["affected_rows"])]
        fails = _failures(fam, prev, set(items), rows)
        out[fid] = {**d, "affected_rows": sorted(d["affected_rows"]), "items": items, "failures": fails,
                    "copy": not fails}
        counts["rows"] += len(d["affected_rows"])
        counts["copy_families"] += not fails
        for f in fails:
            counts[f["check"]] += 1
    counts["families"] = len(out)
    return {"unit": cfg.out.name, "round": cfg.round, "families": out, "counts": dict(counts)}


def unit_spend(cfg: cm.Cfg, uc: cm.Cfg) -> int:
    fams = {f["family_id"] for f in cm.plan_raw(uc)["families"]}
    tag, regen, n = f"@r{uc.round}", bool(uc.regen), 0
    if cfg.usage_log.exists():
        for line in cfg.usage_log.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            parts = r["tag"].split("::") if r else []
            if len(parts) >= 3 and parts[0] == cfg.run_id and tag in parts[2] and (
                    f"@{cm.GEN_REGEN}" in parts[2]) == regen and parts[2].split("@")[0].split("#")[0] in fams:
                n += int(r["total_tokens"] or 0)
    return n


def family_estimate(n_rows: int) -> float:
    """Image-based version of one repaired family (comparison only)."""
    return DELTA_TOKENS + REVIEW_FIXED + REVIEW_PER_ROW * n_rows + ADJ_RATE * (ADJ_FIXED + ADJ_PER_ROW)


def fix_units(cfg: cm.Cfg, rnd: int = cm.FIX_ROUND) -> dict[str, cm.Cfg]:
    from final_construction.wp_v9_vcm import s7_retry
    return {u: cm.Cfg(**{**uc.__dict__, "round": rnd}) for u, uc in s7_retry.units(cfg).items()}


def pending(uc3: cm.Cfg) -> cm.Cfg | None:
    """The unit's pending repair pass (round 3, or round 4 after a finalized round 3 with dropped rows), or None."""
    if not uc3.p("s5_final_r2", "gate.json").exists():
        return None
    if not uc3.p("s5_final_r3", "gate.json").exists():
        return uc3
    uc4 = cm.Cfg(**{**uc3.__dict__, "round": cm.FIX_ROUNDS[1]})
    if uc4.p("s5_final_r4", "gate.json").exists():
        return None
    return uc4 if cm.read(uc3.p("s5_final_r3", "gate.json")).get("dropped_affected_rows") else None


def _estimate(doc: dict) -> float:
    return sum(text_family_estimate(len(d["affected_rows"]), delta=not d.get("copy")) for d in doc["families"].values())


def projection(cfg: cm.Cfg, starting: dict | None = None) -> dict:
    """ledger + every unit's pending pass (in flight: max(estimate - its @r<N> spend, 20% of it))."""
    from final_construction.wp_v9_vcm import s7_retry
    spent, _ = gc.ledger(cfg.usage_log)
    remaining, detail = 0.0, {}
    for u, uc3 in fix_units(cfg).items():
        uc = pending(uc3)
        if uc is None:
            continue
        flight = plan_path(uc).exists()
        if starting and (starting["unit"], starting["round"]) == (u, uc.round):
            doc, flight = starting, False
        else:
            doc = cm.read(plan_path(uc)) if flight else compute_plan(uc)
        est = _estimate(doc)
        rem = max(est - unit_spend(cfg, uc), s7_retry.INFLIGHT_MARGIN * est) if flight else est
        remaining += rem
        detail[u] = {"round": uc.round, "families": len(doc["families"]), "rows": doc["counts"].get("rows", 0),
                     "copy_families": doc["counts"].get("copy_families", 0), "estimate": round(est),
                     "remaining": round(rem), "state": "in_flight" if flight else "not_started"}
    projected = spent + remaining
    return {"spent_total": spent, "units": detail, "projected_total_tokens": round(projected),
            "projected_usd": round(projected / 1e6 * cm.PRICE_USD_PER_M, 2), "budget_stop": cm.PROD_BUDGET_STOP,
            "cap": cm.PROD_CAP, "model": "text_repair (s2_fix_text)"}


def start_gate(cfg: cm.Cfg, plan: dict) -> dict:
    stops = gc.open_stops(cfg)
    if stops:
        raise RuntimeError(f"stop rule fired (s1/{stops[0].name}): no new repair unit may start "
                           "(owner override: run_v9.py --resume-after-stop REASON --owner-instruction TEXT)")
    rep = projection(cfg, plan)
    flag = cfg.p("s1", START_STOP)
    if rep["projected_total_tokens"] > cm.PROD_BUDGET_STOP and not flag.exists():
        cm.write_once(flag, {**rep, "unit": cfg.out.name, "stop": True})
        raise RuntimeError(f"repair start projection {rep['projected_total_tokens']} > budget stop "
                           f"{cm.PROD_BUDGET_STOP}: s1/{START_STOP} written (owner override: --resume-after-stop)")
    return rep


def _resolve_plan(cfg: cm.Cfg) -> dict:
    """The pass's plan: the stored one when the current code computes the same; a new version (write-once) when it
    is stale but unused (no revision yet); refused when stale and already used."""
    cur = compute_plan(cfg)
    files = plan_files(cfg)
    if files:
        stored = cm.read(files[-1])
        if {k: v for k, v in stored.items() if k != "start_projection"} == cur:
            return stored
        used = [p for p in files[-1].parent.glob("*.json") if not p.name.startswith(PLAN)]
        if used:
            raise RuntimeError(f"{files[-1]} differs from the plan the current code computes and revisions exist; "
                               "the owner decides how to proceed")
        path = files[-1].with_name(f"{PLAN}_v{_version(files[-1]) + 1}.json")
    else:
        path = plan_path(cfg)
    cur["start_projection"] = start_gate(cfg, cur)["projected_total_tokens"]
    cur["supersedes"] = files[-1].name if files else None
    cm.write_once(path, cur)
    return cur


def item_level(fails: list[dict]) -> bool:
    """Every remaining hard failure is an item-level text failure (a second text-only attempt can fix it)."""
    from final_construction.wp_v9_vcm import s2_revise
    return all(f["item"] in ITEM_TYPES for f in s2_revise.hard(fails))


def run_fix(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    """`revise_review --round 3|4 --retry`: plan (start marker), copies (no call) and text-only deltas."""
    from final_construction.wp_v9_vcm import s2_author, s2_revise
    plan = _resolve_plan(cfg)
    fams = {f["family_id"]: f for f in cm.plan_raw(cfg)["families"]}
    todo, copied = [], 0
    for fid, d in sorted(plan["families"].items()):
        if s2_author.author_path(cfg, fid, cfg.round).exists():
            continue
        prev = _prev(cfg, fams[fid], d)
        if d["copy"]:
            cm.write_once(s2_author.author_path(cfg, fid, cfg.round),
                          {"family_id": fid, "valid": prev is not None, "output": prev, "revision": cfg.round,
                           "chosen_attempt": None, "attempts": [], "failures_addressed": [], "review_problems": [],
                           "remaining_failures": s2_revise.precheck_output(fams[fid], prev),
                           "copied_from_revision": d["prev_round"], "backend": "copy (labels normalized, no call)",
                           "prompt_sha256": None})
            copied += 1
            continue
        todo.append((fams[fid], prev, d["failures"], []))
    rep = s2_revise.revise_with_feedback(cfg, "revise_review", cfg.round, todo, gpt_fn, generate, max_attempts=2,
                                         retry_if=item_level)
    return {**rep, "copied": copied, "plan_counts": plan["counts"]}
