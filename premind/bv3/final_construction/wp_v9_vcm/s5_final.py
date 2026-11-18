"""Stage finalize (CPU): merge all rounds into the final rows and the final gate (R5, generalised by R7).

Per family the LATEST REVIEWED round that has its rows wins (rounds 1..N with a gate file: s5/, s5_r2/, s5_r3/, ...);
an unreviewed round (e.g. round 2 that stopped after assemble) never supplies final rows - its clean families reach the
final set through carry-forward into the next reviewed round. Acceptance and the gate are computed over these
candidates (v8a minima unchanged, I1-I7 on accepted rows); every reviewed round's rates are reported separately.
Still-rejected rows are dropped: s4/rows_<main>_final.json holds only accepted rows (the drop list is kept).
Outputs are write-once; per-round files are never modified.
"""
from __future__ import annotations

from collections import Counter

from final_construction.wp_v9_vcm import common as cm

MAX_ROUNDS = 9


def _rows_file(cfg: cm.Cfg, rnd: int):  # noqa: ANN202
    return cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}{'' if rnd == 1 else f'_r{rnd}'}.json")


def _gate_file(cfg: cm.Cfg, rnd: int):  # noqa: ANN202
    return cfg.p("s5" if rnd == 1 else f"s5_r{rnd}", "gate.json")


def reviewed_rounds(cfg: cm.Cfg) -> list[int]:
    out = [r for r in range(1, MAX_ROUNDS + 1) if _gate_file(cfg, r).exists()]
    if any(not _rows_file(cfg, r).exists() for r in out):
        raise RuntimeError("a reviewed round has no rows file")
    return out


COVERAGE_TYPES = ("G1_visual", "G1G2_visual", "G0_closed", "GT_absent")   # general-rule control removed (2026-10-03)


def safety_coverage(accepted_rows: list[dict]) -> dict:
    """Unsafe share per row type among accepted rows (GT-absent rows pooled); pass = every type >= COVERAGE_MIN.
    G0 (audit 4.4b, 2026-10-03): since the unsafe-G0 redesign EVERY unsafe G0 carries a generic query, so the specific
    closed G0 rows are safe by construction and the G0 risk decision lives in the generic-query population: once
    styled G0 rows exist, the G0_closed share is computed over them (threshold unchanged); the specific G0 share is
    reported as G0_specific (informational, not gated)."""
    from final_construction.wp_v9_vcm import s1_prod, s4_g0query as gq
    by: dict[str, list[bool]] = {t: [] for t in COVERAGE_TYPES}
    styled_g0 = any(r["row_type"] == "G0_closed" and gq.is_styled(r) for r in accepted_rows)
    info: list[bool] = []
    for r in accepted_rows:
        if r["row_type"] in cm.DROPPED_ROW_TYPES:
            continue
        if styled_g0 and r["row_type"] == "G0_closed" and not gq.is_styled(r):
            info.append(r["risk_label"] == "unsafe")
            continue
        t = "GT_absent" if r["row_type"].endswith("_absent") else r["row_type"]
        by.setdefault(t, []).append(r["risk_label"] == "unsafe")
    share = {t: (round(sum(v) / len(v), 4) if v else None) for t, v in by.items()}
    return {"unsafe_share": share, "n": {t: len(v) for t, v in by.items()}, "min": s1_prod.COVERAGE_MIN,
            "pass": all(v is not None and v >= s1_prod.COVERAGE_MIN for v in share.values()),
            "G0_basis": "generic-query G0" if styled_g0 else "all G0",
            "G0_specific": {"n": len(info), "unsafe_share": round(sum(info) / len(info), 4) if info else None}}


def gate_over(cfg: cm.Cfg, cand: list[dict], acc: set[str]) -> dict:
    """Final gate over candidate rows and the accepted row ids (v8a minima, I1-I7 on accepted, prod coverage)."""
    from final_construction.wp_v9_vcm import s5_review
    cats = {"positive": [0, 0], "control": [0, 0]}
    for x in cand:
        if x["deterministic_pass"]:
            cat = "control" if s5_review.row_kind(x) == "no_memory_control" else "positive"
            cats[cat][1] += 1
            cats[cat][0] += x["row_id"] in acc
    rates = {c: (a / n if n else None) for c, (a, n) in cats.items()}
    th = cm.THRESHOLDS
    i17 = [x["row_id"] for x in cand if x["row_id"] in acc and not all(
        v["pass"] for k, v in x["checks"].items() if k[:2] in {f"I{i}" for i in range(1, 8)})]
    coverage = safety_coverage([x for x in cand if x["row_id"] in acc])
    passed = all(rates[c] is not None and rates[c] >= th[f"v8a_{c}_acceptance_min"] for c in cats) and not i17
    if cfg.profile == "prod":                       # P1: every row type >= COVERAGE_MIN unsafe after acceptance
        passed = passed and coverage["pass"]
    return {"rates": rates, "counts": cats, "i1_i7_fail_on_accepted": i17, "safety_coverage": coverage,
            "gate_pass": passed}


def run_finalize(cfg: cm.Cfg, max_round: int | None = None) -> dict:
    """`max_round=1` (--no-review-round, lever 3): round-1 accepted rows only, the gate is computed on them."""
    if cm.is_extra(cfg):                               # round 7: previous final selection + additive rows
        from final_construction.wp_v9_vcm import s5_extra
        return s5_extra.run_finalize_extra(cfg)
    out_rows, out_gate = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final.json"), cfg.p("s5_final", "gate.json")
    if out_gate.exists():
        return cm.read(out_gate)
    rounds = [r for r in reviewed_rounds(cfg) if max_round is None or r <= max_round]
    if not rounds:                       # pilot fix: round 1 may be unreviewed (all families carried into round 2)
        raise RuntimeError("no round has been reviewed")
    rows = {r: cm.read(_rows_file(cfg, r)) for r in rounds}
    gates = {r: cm.read(_gate_file(cfg, r)) for r in rounds}
    source: dict[str, int] = {}
    for r in rounds:                                     # ascending: later reviewed rounds overwrite earlier ones,
        has = {x["family_id"] for x in rows[r]["rows"]}  # except a later round without rows for the family (its
        for fid in set(rows[r]["report"]["families"]) | has:     # revision failed precheck): earlier rows stand
            if fid in has or fid not in source:
                source[fid] = r
    cand = [dict(x, round=r) for r in rounds for x in rows[r]["rows"] if source[x["family_id"]] == r]
    outcome = {r: {o["row_id"]: {**o, "round": r} for o in gates[r]["outcomes"]} for r in rounds}
    outcomes = [outcome[x["round"]][x["row_id"]] for x in cand]
    acc = {o["row_id"] for o in outcomes if o["accepted"]}
    g = gate_over(cfg, cand, acc)
    rates, cats, i17, coverage, passed = (g[k] for k in ("rates", "counts", "i1_i7_fail_on_accepted",
                                                          "safety_coverage", "gate_pass"))
    dropped = [x["row_id"] for x in cand if x["row_id"] not in acc]
    families = {fid: rows[r]["report"]["families"].get(fid, "assembled") for fid, r in source.items()}
    rep = {"gate": "v9 final (latest reviewed round per family)", "rates": rates, "counts": cats, "gate_pass": passed,
           "safety_coverage": coverage,
           "i1_i7_fail_on_accepted": i17, "outcomes": outcomes,
           "round_rates": {f"round{r}": {"rates": gates[r].get("rates"), "counts": gates[r].get("counts")} for r in rounds},
           "family_source_round": source, "revised_families": sorted(f for f, r in source.items() if r > 1),
           "yield": {"candidate_rows": len(cand), "deterministic_pass": sum(x["deterministic_pass"] for x in cand),
                     "accepted": len(acc), "dropped": len(dropped)},
           "accepted_by_round": dict(Counter(o["round"] for o in outcomes if o["accepted"])),
           "no_review_round": max_round == 1}
    cm.write_once(out_rows, {"rows": [x for x in cand if x["row_id"] in acc], "dropped_rows": dropped,
                             "report": {**rows[rounds[0]]["report"], "families": families, "finalized": True,
                                        "rounds": rounds}})
    cm.write_once(out_gate, rep)
    cm.write_once(cfg.p("s5_final", "CANARY_GATE_PASS_v9.json" if passed else "CANARY_GATE_FAIL_v9.json"),
                  {"rates": rates, "round_rates": rep["round_rates"]})
    return {k: rep[k] for k in ("rates", "counts", "gate_pass", "round_rates", "yield", "accepted_by_round",
                                "family_source_round", "safety_coverage", "no_review_round")}
