"""Repair passes (round 3, round 4): `finalize --round N --retry` writes s5_final_rN/ and `prod_check --round N
--retry` the repair check. Pass 2 (round 4) starts from s5_final_r3 and only revisits its dropped affected rows, so the
per-row priority is pass 2 > pass 1 > dropped; rows accepted in pass 1 are never re-paid or re-reviewed.

Per-ROW selection (owner's simpler alternative, 2026-10-03): every row of the current final selection (s5_final_r2)
that is NOT affected stays as it is; an AFFECTED row (s2_fix FIX_PLAN.json) is replaced by its round-3 row when that
row passed the deterministic checks (incl. S1-S4) and the review, and is DROPPED otherwise - a defective row is never
kept. Facts and memory texts of a repaired family are byte-identical to its base round (items-only deltas), so its
rows stay consistent although they now come from two rounds ("round" 3 on the repaired rows).
General-rule rows (cm.DROPPED_ROW_TYPES, owner 2026-10-03) are excluded from this selection (stored rows untouched).
Release selection r3 > r2 > r1 (common.final_tag: s6_r3/, release_r3/).
"""
from __future__ import annotations

import logging
from collections import Counter

from final_construction.wp_v9_vcm import common as cm

logger = logging.getLogger(__name__)
RULE = ("per row: affected rows -> repaired row of this pass if accepted, else dropped; every other row from the "
        "previous final selection (pass 1: s5_final_r2, pass 2: s5_final_r3)")


def _rows(cfg: cm.Cfg, rnd: int) -> dict[str, dict]:
    p = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}{'' if rnd == 1 else f'_r{rnd}'}.json")
    return {r["row_id"]: r for r in cm.read(p)["rows"]} if p.exists() else {}


def run_finalize_fix(cfg: cm.Cfg) -> dict:
    from final_construction.wp_v9_vcm import s2_fix, s5_final
    n = cfg.round
    out_rows, out_gate = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final_r{n}.json"), cfg.p(f"s5_final_r{n}", "gate.json")
    if out_gate.exists():
        return _summary(cm.read(out_gate))
    if not cm.is_fix(cfg):
        raise RuntimeError("finalize of a repair pass runs with --round 3|4 --retry")
    affected = s2_fix.affected_row_ids(cfg) or set()
    fin2 = cm.read(cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final_r{n - 1}.json"))
    g2 = cm.read(cfg.p(f"s5_final_r{n - 1}", "gate.json"))
    by_round: dict[int, dict[str, dict]] = {}
    keep, cand, excluded = [], [], Counter()
    for o in g2["outcomes"]:                                          # every round-2-final candidate row
        rnd = o.get("round", 1)
        if rnd not in by_round:
            by_round[rnd] = _rows(cfg, rnd)
        row = by_round[rnd][o["row_id"]]
        if row["row_type"] in cm.DROPPED_ROW_TYPES:
            excluded[row["row_type"]] += o["accepted"]
            continue
        if o["row_id"] in affected:
            continue
        cand.append(({**row, "round": rnd, **({"source_round": rnd, "review_mode": "text_repair"}
                                              if rnd in cm.FIX_ROUNDS else {})}, o))
        keep.append(o["row_id"])
    rows3 = _rows(cfg, n)
    g3 = {o["row_id"]: o for o in cm.read(cfg.p(f"s5_r{n}", "gate.json"))["outcomes"]} \
        if cfg.p(f"s5_r{n}", "gate.json").exists() else {}
    repaired, dropped = list(g2.get("repaired_rows", [])) if n > cm.FIX_ROUNDS[0] else [], []
    for rid in sorted(affected):
        o, row = g3.get(rid), rows3.get(rid)
        if row and o and o["accepted"]:
            cand.append(({**row, "round": n, "source_round": n, "review_mode": "text_repair"},
                         {**o, "round": n, "review_mode": "text_repair"}))
            repaired.append(rid)
        else:
            dropped.append(rid)
    rows, outcomes = [r for r, _ in cand], [o for _, o in cand]
    acc = {o["row_id"] for o in outcomes if o["accepted"]}
    g = s5_final.gate_over(cfg, rows, acc)
    n2 = sum(o["accepted"] for o in g2["outcomes"])
    rep = {"gate": f"v9 repair final ({RULE})", "selection_rule": RULE, **g, "outcomes": outcomes,
           "affected_rows": len(affected), "repaired_rows": repaired, "dropped_affected_rows": dropped,
           "excluded_row_types": {t: {"accepted_rows_excluded": k, "reason": cm.DROP_REASON} for t, k in excluded.items()},
           "family_source_round": g2.get("family_source_round", {}),
           "yield": {"candidate_rows": len(rows), "accepted": len(acc), "accepted_round2_final": n2,
                     "accepted_previous_final": n2, "repaired_this_pass": len(repaired) - (
                         len(g2.get("repaired_rows", [])) if n > cm.FIX_ROUNDS[0] else 0),
                     "repaired": len(repaired), "dropped_affected": len(dropped),
                     "general_rule_excluded": sum(excluded.values()), "change": len(acc) - n2},
           "accepted_by_round": dict(Counter(o["round"] for o in outcomes if o["accepted"])),
           "split_accepted": dict(Counter(r["split"] for r in rows if r["row_id"] in acc)), "retry": True,
           "no_review_round": False}
    cm.write_once(out_rows, {"rows": [r for r in rows if r["row_id"] in acc],
                             "dropped_rows": [r["row_id"] for r in rows if r["row_id"] not in acc] + dropped,
                             "report": {**fin2["report"], "finalized": True, "repair_round": n,
                                        "selection_rule": RULE}})
    cm.write_once(out_gate, rep)
    logger.info("repair finalize %s: %d repaired, %d dropped", cfg.out.name, len(repaired), len(dropped))
    return _summary(rep)


def _summary(rep: dict) -> dict:
    return {k: rep[k] for k in ("rates", "counts", "gate_pass", "yield", "accepted_by_round", "safety_coverage",
                                "split_accepted", "selection_rule")}


def fix_check(cfg: cm.Cfg) -> dict:
    """`prod_check --round 3|4 --retry` after the unit's repair finalize: projection + coverage, stop file if needed."""
    from final_construction.wp_v9_vcm import s2_fix, s5_final, s7_retry
    tag = f"fix_{cfg.out.name}" if cfg.round == cm.FIX_ROUNDS[0] else f"fix{cfg.round - 2}_{cfg.out.name}"
    out = cfg.p("s1", f"PROD_CHECK_{tag}.json")
    if out.exists():
        return cm.read(out)
    if not cfg.p(f"s5_final_r{cfg.round}", "gate.json").exists():
        raise RuntimeError(f"{tag} is not finalized ({cfg.p(f's5_final_r{cfg.round}', 'gate.json')})")
    rep = s2_fix.projection(cfg)
    cov = s5_final.safety_coverage(s7_retry.selected_accepted(cfg))["unsafe_share"]
    low = {t: v for t, v in cov.items() if (v or 0.0) < cm.PROD_COVERAGE_STOP}
    plan = cm.read(s2_fix.plan_path(cfg))
    spend = s2_fix.unit_spend(cfg, cfg)
    rep.update(check=tag, unit=cfg.out.name, unit_families=len(plan["families"]), unit_fix_spend=spend,
               unit_yield=cm.read(cfg.p(f"s5_final_r{cfg.round}", "gate.json"))["yield"], unsafe_share_accepted=cov,
               coverage_min=cm.PROD_COVERAGE_STOP, coverage_below=low,
               prereg_sha256=cm.sha_file(cfg.prereg) if cfg.prereg.exists() else None,
               stop=rep["projected_total_tokens"] > cm.PROD_BUDGET_STOP or bool(low))
    cm.write_once(out, rep)
    if rep["stop"]:
        cm.write_once(cfg.p("s1", f"PROD_STOP_{tag}.json"), rep)
    return rep
