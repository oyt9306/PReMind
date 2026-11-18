"""Release v3 (train/dev) and the sealed holdout release (owner 2026-10-03; CPU, code-only, no paid call).

After every unit's round-7 selection (s5_final_r7, additive rows) exists:
  1. r8 = dataset-level, LABEL-INDEPENDENT query assignment over ALL units' styled rows (s4_g0query.assign; unsafe G0
     acute and non-acute, safe generic G0, G1_panic_memory + absent): a new write-once code-only selection
     s5_final_r8 for every unit (golds and labels unchanged; query_style recomputed from the text, analysis only;
     the previous query kept as query_before_balance). S2/S2b/S2c re-checked on every changed row.
  2. S1x / S6 must already hold in every final selection (no late fix under a newer selection).
  3. render + export (3 modes) per unit (s6_r8/, release_r8/), then the combined release:
     train/dev -> prod_v9/RELEASE_TRAIN_DEV_v3/ (v1/v2 untouched); holdout -> holdout_v9_sealed/RELEASE_HOLDOUT_v1/
     (CLI output of the holdout profile is redacted to counts and hashes).
The manifest carries S2d (s4_g0query.label_report): per-text label distributions and a query-only probe vs the
majority baseline for risk, action and memory need.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from final_construction.wp_v9_vcm import common as cm

logger = logging.getLogger(__name__)
Q_TAG = "_r8"
RELEASE_HOLDOUT = "RELEASE_HOLDOUT_v1"


def holdout_units(cfg: cm.Cfg) -> dict[str, cm.Cfg]:
    root = cfg.p("s1").parent
    out = {}
    n_fam = len(cm.read(root / "s1" / "plan.json")["families"])
    for j in range(1, n_fam // cm.PROD_CHUNK + 2):
        d = cm.chunk_dir(root, j, "holdout")                     # the active tree (re-authored chunk: _g2)
        uc = cm.Cfg(**{**cfg.__dict__, "chunk": j, "out": d, "s1_dir": root / "s1",
                       "s0_dir": root / "s0", "round": 1, "retry": False})
        if uc.p("s5_final", "gate.json").exists():
            out[d.name] = uc
    return out


def _final(uc: cm.Cfg) -> tuple[str, list[dict], dict]:
    tag = cm.final_tag(uc)
    return tag, cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json"))["rows"], \
        cm.read(uc.p(f"s5_final{tag}", "gate.json"))


def run_query_balance(units: dict[str, cm.Cfg], key: str, q_tag: str = Q_TAG,
                      added: dict[str, list[dict]] | None = None) -> dict:
    """Write s5_final<q_tag> for every unit (all or none): the label-independent dataset-level query assignment over
    the current final selections (+ `added` accepted rows per unit, e.g. the v4 re-review), re-running S2/S2b/S2c and
    Q1/I1-I2 on every assigned query."""
    from final_construction.wp_v9_vcm import s4_acute as ac, s4_g0query as gq
    have = [u for u, uc in units.items() if uc.p(f"s5_final{q_tag}", "gate.json").exists()]
    if have and len(have) != len(units):
        raise RuntimeError(f"partial query balance (r8 only in {have}); the owner decides how to proceed")
    finals = {u: _final(uc) for u, uc in units.items()}
    if have:
        rows = [r for _, rs, _ in finals.values() for r in rs]
        return {"existing": True, "s2d": gq.label_report(rows), "shortcut": gq.shortcut_probe(rows)}
    finals = {u: (t, rs + list((added or {}).get(u, [])), g) for u, (t, rs, g) in finals.items()}
    plans = {f["family_id"]: f for uc in units.values() for f in cm.plan_raw(uc)["families"]}
    mapping = gq.assign([r for _, rs, _ in finals.values() for r in rs], key, plans)
    changed_total = 0
    for u, uc in units.items():
        tag, rows, gate = finals[u]
        fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
        by: dict[str, list[dict]] = defaultdict(list)
        new_rows, changed = [], []
        for r in rows:
            q = mapping.get(r["row_id"])
            if q is not None and q != r["query"]:
                r = {**r, "query": q, "query_style": gq.style_of_query(q), "query_before_balance": r["query"]}
                changed.append(r["row_id"])
            elif q is not None:
                r = {**r, "query_style": gq.style_of_query(q)}
            new_rows.append(r)
            by[r["family_id"]].append(r)
        for fid, rs in by.items():
            if not any(r["row_id"] in mapping for r in rs):
                continue
            res = ac.row_results(fams[fid], rs)
            for r in rs:
                if r["row_id"] in mapping:
                    gen = gq.generic_checks(fams[fid], r["query"])
                    bad = [c for c in (ac.S2, ac.S2B, ac.S2C) if not res[r["row_id"]][c]["pass"]] + \
                        [c for c, v in gen.items() if not v["pass"]]
                    if bad:
                        raise RuntimeError(f"{r['row_id']}: balanced query fails {bad}")
                    r["checks"] = {**r["checks"], **{c: res[r["row_id"]][c] for c in (ac.S2, ac.S2B, ac.S2C)}, **gen}
        info = {"source_selection": f"s5_final{tag}", "styled_rows": sum(r["row_id"] in mapping for r in rows),
                "changed": sorted(changed), "key": key}
        info["added_rows"] = len((added or {}).get(u, []))
        fin = cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json"))
        add_out = [o for r in (added or {}).get(u, []) for o in [r.get("_outcome")] if o]
        new_rows = [{k: v for k, v in r.items() if k != "_outcome"} for r in new_rows]
        cm.write_once(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{q_tag}.json"),
                      {"rows": new_rows, "dropped_rows": fin.get("dropped_rows", []),
                       "report": {**fin["report"], "query_balance": info, "code_only": True}})
        cm.write_once(uc.p(f"s5_final{q_tag}", "gate.json"),
                      {**{k: v for k, v in gate.items() if k not in ("gate", "outcomes")}, "code_only": True,
                       "outcomes": gate["outcomes"] + add_out, "query_balance": info,
                       "gate": f"v9 label-independent query assignment of s5_final{tag} (no call)"})
        changed_total += len(changed)
    rows = [r for uc in units.values() for r in _final(uc)[1]]
    return {"existing": False, "changed": changed_total, "s2d": gq.label_report(rows),
            "shortcut": gq.shortcut_probe(rows)}


def release_v3(cfg: cm.Cfg, units: dict[str, cm.Cfg]) -> dict:
    """Round-7 release: query balance (r8), S1x/S6 verification, render + export, combined release."""
    from final_construction.wp_v9_vcm import s4_acute as ac, s6_export, s6_render, s7_release as rel
    name = RELEASE_HOLDOUT if cfg.profile == "holdout" else rel.RELEASE_DIR_V3
    units = {u: uc for u, uc in units.items() if uc.p("s5_final", "gate.json").exists()}
    if not units:
        raise RuntimeError("no finalized unit")
    missing = [u for u, uc in units.items() if not uc.p("s5_final_r7", "gate.json").exists()]
    if missing:
        raise RuntimeError(f"round 7 is not finalized for {missing}")
    qb = run_query_balance(units, f"{cfg.profile}_v3")
    rep = {"units": {}, "query_balance": {k: v for k, v in qb.items() if k != "s2d"}}
    for u, uc in units.items():
        tag, rows, _ = _final(uc)
        fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[r["family_id"]].append(r)
        bad = [fid for fid, rs in by.items() if rel.s1x_fix(fams[fid], rs)[1:] != ([], {})
               or any(not d[ac.S6]["pass"] for d in ac.row_results(fams[fid], rs).values())]
        if bad:
            raise RuntimeError(f"{u}: S1x/S6 violations in s5_final{tag}: {bad[:5]}")
        for mode in rel.MODES:
            s6_render.run_render(uc, mode, 0.2)
            s6_export.run_export(uc, mode)
        rep["units"][u] = f"s5_final{tag}"
    rep["release"] = rel.build_release(cfg, name, extra={"s2d_query_label_independence": qb["s2d"],
                                                         "shortcut_probe_all_g0_panic": qb.get("shortcut"),
                                                         "query_assignment": "s4_g0query.assign (dataset-level)"},
                                       units_in=units)
    rep["s2d"] = qb["s2d"]
    return rep


def release_v4(cfg: cm.Cfg, units: dict[str, cm.Cfg]) -> dict:
    """v4 = the r8 selection + the accepted round-9 re-reviewed safe generic rows; dataset-level query re-assignment
    (s5_final_r9, code-only), S1x/S6 verification, render + export, RELEASE_TRAIN_DEV_v4 (v1-v3 untouched)."""
    from final_construction.wp_v9_vcm import s4_acute as ac, s5_rereview, s6_export, s6_render, s7_release as rel
    units = {u: uc for u, uc in units.items() if uc.p("s5_final", "gate.json").exists()}
    pending = [u for u, uc in units.items() if s5_rereview.needed(uc) and not uc.p("s5_r9", "gate.json").exists()]
    if pending:
        raise RuntimeError(f"round-9 re-review not finished for {pending}")
    added = {u: s5_rereview.accepted(uc) for u, uc in units.items()}
    qb = run_query_balance(units, f"{cfg.profile}_v4", "_r9", added)
    rep = {"units": {}, "added": {u: len(v) for u, v in added.items()},
           "query_balance": {k: v for k, v in qb.items() if k not in ("s2d", "shortcut")}}
    for u, uc in units.items():
        tag, rows, _ = _final(uc)
        fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[r["family_id"]].append(r)
        bad = [fid for fid, rs in by.items() if rel.s1x_fix(fams[fid], rs)[1:] != ([], {})
               or any(not d[ac.S6]["pass"] for d in ac.row_results(fams[fid], rs).values())]
        if bad:
            raise RuntimeError(f"{u}: S1x/S6 violations in s5_final{tag}: {bad[:5]}")
        for mode in rel.MODES:
            s6_render.run_render(uc, mode, 0.2)
            s6_export.run_export(uc, mode)
        rep["units"][u] = f"s5_final{tag}"
    rep["release"] = rel.build_release(cfg, "RELEASE_TRAIN_DEV_v4",
                                       extra={"s2d_query_label_independence": qb["s2d"],
                                              "shortcut_probe_all_g0_panic": qb["shortcut"],
                                              "panic_memory_absent": "none by owner decision 2026-10-03",
                                              "query_assignment": "s4_g0query.assign (dataset-level, Q1/I1 re-run)"},
                                       units_in=units)
    rep["s2d"], rep["shortcut"] = qb["s2d"], qb["shortcut"]
    return rep


def release_v5(cfg: cm.Cfg, units: dict[str, cm.Cfg], name: str) -> dict:
    """After round 10 (extended safe generic G0) in EVERY unit: one dataset-wide label-independent query assignment
    (s5_final_r11, code-only; S2/S2b/S2c + Q1/I1-I2 re-run), S1x/S6 verification, render + export, combined release
    `name` (train/dev: RELEASE_TRAIN_DEV_v5 with v1-v4 untouched; holdout: RELEASE_HOLDOUT_v1). The manifest carries
    S2d and the check-12 shortcut probe over all G0 + panic-memory rows."""
    from final_construction.wp_v9_vcm import s4_acute as ac, s6_export, s6_render, s7_release as rel
    units = {u: uc for u, uc in units.items() if uc.p("s5_final", "gate.json").exists()}
    missing = [u for u, uc in units.items() if not uc.p(f"s5_final_r{cm.EXTRA2_ROUND}", "gate.json").exists()]
    if not units or missing:
        raise RuntimeError(f"round {cm.EXTRA2_ROUND} is not finalized for {missing or 'any unit'}")
    qb = run_query_balance(units, f"{cfg.profile}_v5", cm.BALANCE_TAG_V5)
    rep = {"units": {}, "query_balance": {k: v for k, v in qb.items() if k not in ("s2d", "shortcut")}}
    for u, uc in units.items():
        tag, rows, _ = _final(uc)
        fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[r["family_id"]].append(r)
        bad = [fid for fid, rs in by.items() if rel.s1x_fix(fams[fid], rs)[1:] != ([], {})
               or any(not d[ac.S6]["pass"] for d in ac.row_results(fams[fid], rs).values())]
        if bad:
            raise RuntimeError(f"{u}: S1x/S6 violations in s5_final{tag}: {bad[:5]}")
        for mode in rel.MODES:
            s6_render.run_render(uc, mode, 0.2)
            s6_export.run_export(uc, mode)
        rep["units"][u] = f"s5_final{tag}"
    rep["release"] = rel.build_release(cfg, name, extra={
        "s2d_query_label_independence": qb["s2d"], "shortcut_probe_all_g0_panic": qb["shortcut"],
        "panic_memory_absent": "none by owner decision 2026-10-03",
        "g0_safety_coverage": g0_coverage([r for uc in units.values() for r in _final(uc)[1]]),
        "query_assignment": "s4_g0query.assign (dataset-level, Q1/I1 re-run) after round 10"}, units_in=units)
    rep["s2d"], rep["shortcut"] = qb["s2d"], qb["shortcut"]
    return rep


def g0_coverage(rows: list[dict]) -> dict:
    """G0 unsafe share split by population (audit 4.4b): generic-query G0 (styled; risk decided under a generic
    query) vs specific closed G0 (ordinary visual questions), per split."""
    from final_construction.wp_v9_vcm import s4_g0query as gq
    out = {}
    for split in ("train", "dev"):
        for pop, keep in (("generic", True), ("specific", False)):
            g0 = [r for r in rows if r["split"] == split and r["row_type"] == "G0_closed" and gq.is_styled(r) == keep]
            unsafe = sum(r["risk_label"] == "unsafe" for r in g0)
            out[f"{split}_{pop}"] = {"rows": len(g0), "unsafe": unsafe,
                                     "unsafe_share": round(unsafe / len(g0), 4) if g0 else None}
        allg0 = [r for r in rows if r["split"] == split and r["row_type"] == "G0_closed"]
        out[f"{split}_all"] = {"rows": len(allg0), "unsafe_share": round(sum(r["risk_label"] == "unsafe"
                                                                            for r in allg0) / max(1, len(allg0)), 4)}
    return out
