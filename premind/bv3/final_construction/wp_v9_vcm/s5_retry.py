"""Production retry round: `finalize --retry` (CPU) writes the NEW s5_final_r2/ (s5_final/ is write-once and kept).

Selection rule (owner 2026-10-03), per family, never mixing rounds inside a family:
  * not revised in round 2 (no s2/author_rev2/<family>.json)  -> its round-1 final rows (source_round 1);
  * revised -> its round-2 accepted rows when their count >= its round-1 accepted count (source_round 2), else its
    round-1 final rows (source_round 1; a revision that failed precheck has 0 round-2 rows, so a family with round-1
    rows keeps them).
Round 1 = the unit's s5_final/gate.json outcomes (its candidates are read from the per-round rows files named by each
outcome's "round"); round 2 = s5_r2/gate.json over s4/rows_<main>_r2.json. The gate (v8a minima, I1-I7 on accepted,
prod coverage) is recomputed over the selected candidates with s5_final.gate_over. Outputs (write-once):
s4/rows_<main>_final_r2.json (accepted rows, each with "round" = "source_round") and s5_final_r2/gate.json
(family_source_round, per-family accepted counts, revised / regressed families). Render and export pick these up via
common.final_tag ("_r2" -> s6_r2/, release_r2/).
"""
from __future__ import annotations

import logging
from collections import Counter

from final_construction.wp_v9_vcm import common as cm

logger = logging.getLogger(__name__)
RULE = "per family: round 2 if revised and accepted_r2 >= accepted_r1, else round 1 (never mixed)"


def select_sources(n1: dict[str, int], n2: dict[str, int], revised: set[str], families: list[str]) -> dict[str, int]:
    """family -> source round (the owner's selection rule)."""
    return {f: 2 if f in revised and n2.get(f, 0) >= n1.get(f, 0) else 1 for f in families}


def _rows(cfg: cm.Cfg, rnd: int) -> dict[str, dict]:
    p = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}{'' if rnd == 1 else f'_r{rnd}'}.json")
    return {r["row_id"]: r for r in cm.read(p)["rows"]}


def revised_families(cfg: cm.Cfg) -> set[str]:
    d = cfg.p("s2", "author_rev2")
    return {cm.read(p)["family_id"] for p in d.glob("*.json") if p.name != "CARRY_FORWARD.json"} if d.exists() else set()


def run_finalize_retry(cfg: cm.Cfg) -> dict:
    """Merge the unit's round-1 final result with its round-2 retry under the selection rule."""
    if cm.is_fix(cfg):                                  # acute/G0 repair round 3: per-row selection (s5_fix)
        from final_construction.wp_v9_vcm import s5_fix
        return s5_fix.run_finalize_fix(cfg)
    out_rows, out_gate = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final_r2.json"), cfg.p("s5_final_r2", "gate.json")
    if out_gate.exists():
        return _summary(cm.read(out_gate))
    if not (cfg.retry and cfg.round == 2):
        raise RuntimeError("finalize --retry runs with --round 2 --retry")
    g1, g2 = cm.read(cfg.p("s5_final", "gate.json")), cm.read(cfg.p("s5_r2", "gate.json"))
    by_round: dict[int, dict[str, dict]] = {}
    cand1 = []
    for o in g1["outcomes"]:
        rnd = o.get("round", 1)
        if rnd not in by_round:
            by_round[rnd] = _rows(cfg, rnd)
        cand1.append((by_round[rnd][o["row_id"]], {**o, "round": rnd}))
    rows2 = _rows(cfg, 2)
    cand2 = [(rows2[o["row_id"]], {**o, "round": 2}) for o in g2["outcomes"]]
    n1 = Counter(r["family_id"] for r, o in cand1 if o["accepted"])
    n2 = Counter(r["family_id"] for r, o in cand2 if o["accepted"])
    fin1 = cm.read(cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final.json"))
    rep2 = cm.read(cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_r2.json"))["report"]
    revised = revised_families(cfg)
    families = sorted(set(fin1["report"]["families"]) | set(rep2["families"]) | {r["family_id"] for r, _ in cand1})
    source = select_sources(dict(n1), dict(n2), revised, families)
    cand = [({**r, "round": o["round"], "source_round": 1}, o) for r, o in cand1 if source[r["family_id"]] == 1]
    cand += [({**r, "round": 2, "source_round": 2}, o) for r, o in cand2 if source[r["family_id"]] == 2]
    if any(len({o["round"] == 2 for r, o in cand if r["family_id"] == f}) > 1 for f in families):
        raise RuntimeError("selection mixed rounds inside a family")              # by construction impossible
    rows, outcomes = [r for r, _ in cand], [o for _, o in cand]
    acc = {o["row_id"] for o in outcomes if o["accepted"]}
    from final_construction.wp_v9_vcm import s5_final
    g = s5_final.gate_over(cfg, rows, acc)
    status = {f: (rep2["families"].get(f, "not_in_round2") if source[f] == 2 else
                  fin1["report"]["families"].get(f, "assembled")) for f in families}
    rep = {"gate": f"v9 retry final ({RULE})", "selection_rule": RULE, **g, "outcomes": outcomes,
           "round_rates": {"round1_final": {k: g1.get(k) for k in ("rates", "counts")},
                           "round2": {k: g2.get(k) for k in ("rates", "counts")}},
           "family_source_round": source,
           "family_accepted": {f: {"r1": n1.get(f, 0), "r2": n2.get(f, 0) if f in revised else None,
                                   "source_round": source[f]} for f in families},
           "revised_families": sorted(revised & set(families)),
           "kept_round1_after_revision": sorted(f for f in revised & set(families) if source[f] == 1),
           "yield": {"candidate_rows": len(rows), "deterministic_pass": sum(r["deterministic_pass"] for r in rows),
                     "accepted": len(acc), "accepted_round1_final": sum(n1.values()),
                     "gain": len(acc) - sum(n1.values()), "dropped": len(rows) - len(acc)},
           "accepted_by_round": dict(Counter(o["round"] for o in outcomes if o["accepted"])),
           "zero_row_families": {"round1": sum(n1.get(f, 0) == 0 for f in families),
                                 "final": sum(1 for f in families if not any(
                                     r["family_id"] == f and r["row_id"] in acc for r in rows))},
           "no_review_round": False, "retry": True}
    cm.write_once(out_rows, {"rows": [r for r in rows if r["row_id"] in acc],
                             "dropped_rows": [r["row_id"] for r in rows if r["row_id"] not in acc],
                             "report": {**fin1["report"], "families": status, "finalized": True, "retry": True,
                                        "selection_rule": RULE, "rounds": sorted(set(source.values()))}})
    cm.write_once(out_gate, rep)
    logger.info("retry finalize %s: accepted %d (round-1 final %d)", cfg.out.name, len(acc), sum(n1.values()))
    return _summary(rep)


def _summary(rep: dict) -> dict:
    return {k: rep[k] for k in ("rates", "counts", "gate_pass", "yield", "accepted_by_round", "safety_coverage",
                                "kept_round1_after_revision", "zero_row_families", "selection_rule")}
