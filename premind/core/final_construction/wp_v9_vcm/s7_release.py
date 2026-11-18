"""Final train/dev release (CPU, code-only, no paid call): post-finalize cross-round acute check S1x and the combined
release of the 8 final units (`release_train_dev --profile prod`).

S1x (final audit #10, 2026-10-03): within a unit's FINAL selection, if any selected row of a family is acute, every
selected row of that family must be acute. A non-acute row of such a family (carried over from an earlier round) is
relabelled in code - acute_visual_hazard copied from the family (present, evidence) with a name-free pattern built
from the hazard / safety-action terms of the row's own first sentence, risk unsafe, safety_escalation A9 in the row
and every arm - when its gold is hazard-first by the STRICT test (first sentence holds a hazard-lexicon, safety-
response or safety-action term; evidence words such as "bag" do not count) and the S1-S6 checks then pass; otherwise
the row is DROPPED with the reason. A unit with a change gets the new write-once selection s5_final_r5/ +
rows_<main>_final_r5.json (code_only); render/export (3 modes) then write s6_r5/ and release_r5/.
Combined release: prod_v9/RELEASE_TRAIN_DEV_v1/{train,dev}.jsonl (retrieved), <mode>/{train,dev}.jsonl for the other
two modes, MANIFEST.json (per unit: final selection, source rounds, generation versions, counts by split / row type;
sha256 of every jsonl; excluded row types). Images stay in the unit folders; every path is relative to prod_v9/.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from final_construction.wp_v9_vcm import common as cm

logger = logging.getLogger(__name__)
S1X_TAG = "_r5"
G0Q_TAG = "_r6"                      # code-only unsafe-G0 query diversity (panic / calm lists, owner 2026-10-03)
S1X = "S1x_cross_round_acute"
MODES = ("retrieved", "gt_retrieved", "gt_retrieved_drop")
RELEASE_DIR = "RELEASE_TRAIN_DEV_v1"
RELEASE_DIR_V2 = "RELEASE_TRAIN_DEV_v2"      # v1 + seeded panic/calm unsafe-G0 queries (query_style)
RELEASE_DIR_V3 = "RELEASE_TRAIN_DEV_v3"      # v2 + round-7 additive rows (safe generic G0, G1_panic_memory)


def strict_terms(fam: dict, gold: str) -> list[str]:
    from final_construction.wp_v9_vcm import s4_acute as ac
    first = ac.first_sentence(gold)
    banned = {w.lower() for n in ac.labels(fam) for w in ac.words(n)}
    return sorted({m.group(0).lower() for rx in (ac.LEXICON, ac.RESPONSE, ac.SAFETY_ACTION) for m in rx.finditer(first)}
                  - banned, key=lambda t: (-len(t), t))


def relabel_arms(r: dict) -> dict:
    """The S1x labels on every arm of an (acute-relabelled) row (also after render re-derives the arms)."""
    arms = {k: ({**a, "expected_model_action": "safety_escalation", "b7_subtype": "A9",
                 "gold_action_set": ["safety_escalation"]} if a.get("excluded") is None and "expected_model_action" in a
                else a) for k, a in r["arms"].items()}
    return {**r, "arms": arms}


def s1x_fix(fam: dict, rows: list[dict]) -> tuple[list[dict], list[str], dict[str, str]]:
    """(rows after S1x, relabelled row ids, dropped row id -> reason) for ONE family's selected rows."""
    from final_construction.wp_v9_vcm import s4_acute as ac
    acute = [r for r in rows if (r.get("acute_visual_hazard") or {}).get("present")]
    if not acute:
        return rows, [], {}
    evidence = next((r["acute_visual_hazard"].get("evidence") for r in acute if r["acute_visual_hazard"].get("evidence")),
                    "")
    out, fixed, dropped = [], [], {}
    for r in rows:
        if (r.get("acute_visual_hazard") or {}).get("present"):
            out.append(r)
            continue
        terms = strict_terms(fam, r.get("gold_answer") or "")
        if not terms:
            dropped[r["row_id"]] = "S1x: family acute, row not hazard-first (strict test); not relabelled"
            continue
        pat = "(?i)\\b(?:" + "|".join(re.escape(t) for t in terms) + ")\\b"
        new = relabel_arms({**r, "acute_visual_hazard": {"present": True, "evidence": evidence, "gold_pattern": pat},
                            "risk_label": "unsafe", "physical_safety_or_health_risk": "true",
                            "expected_model_action": "safety_escalation", "b7_subtype": "A9",
                            "gold_action_set": ["safety_escalation"],
                            "safety_priority": "acute_incident_or_injury_escalation", "s1x_fixed": True})
        out.append(new)
        fixed.append(r["row_id"])
    res = ac.row_results(fam, out)
    for rid in list(fixed):
        bad = sorted(c for c, v in res[rid].items() if not v["pass"])
        if bad:
            dropped[rid] = f"S1x: relabelled row fails {','.join(bad)}"
            fixed.remove(rid)
    out = [r if r["row_id"] not in fixed else {**r, "checks": {**r["checks"], S1X: {
        "pass": True, "detail": "label-only fix in code (family acute)"}}} for r in out if r["row_id"] not in dropped]
    return out, fixed, dropped


def unit_cfgs(cfg: cm.Cfg) -> dict[str, cm.Cfg]:
    from final_construction.wp_v9_vcm import s7_retry
    return {u: cm.Cfg(**{**uc.__dict__, "round": 1, "retry": False}) for u, uc in s7_retry.units(cfg).items()}


def run_s1x(uc: cm.Cfg) -> dict:
    """S1x on one unit's final selection; writes s5_final_r5 only when something changes."""
    from final_construction.wp_v9_vcm import s5_final
    if uc.p(f"s5_final{S1X_TAG}", "gate.json").exists():
        g = cm.read(uc.p(f"s5_final{S1X_TAG}", "gate.json"))
        return {"unit": uc.out.name, "changed": True, "existing": True, **g["s1x"]}
    tag = cm.final_tag(uc)
    fin, gate = cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json")), cm.read(uc.p(f"s5_final{tag}", "gate.json"))
    fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
    by: dict[str, list[dict]] = defaultdict(list)
    for r in fin["rows"]:
        by[r["family_id"]].append(r)
    rows, fixed, dropped = [], [], {}
    for fid, rs in by.items():
        o, f, d = s1x_fix(fams[fid], rs)
        rows += o
        fixed += f
        dropped.update(d)
    if not fixed and not dropped:
        return {"unit": uc.out.name, "changed": False, "selection": f"s5_final{tag}"}
    s1x = {"fixed": sorted(fixed), "dropped": dict(sorted(dropped.items())), "source_selection": f"s5_final{tag}"}
    keep = {r["row_id"] for r in rows}
    outcomes = [o for o in gate["outcomes"] if o["row_id"] not in dropped]
    rep = {**{k: gate[k] for k in ("rates", "counts", "selection_rule") if k in gate}, "outcomes": outcomes,
           "gate": f"v9 S1x code-only fix of s5_final{tag} (no call)", "code_only": True, "s1x": s1x,
           "repaired_rows": gate.get("repaired_rows", []), "family_source_round": gate.get("family_source_round", {}),
           "safety_coverage": s5_final.safety_coverage(rows), "gate_pass": gate.get("gate_pass"),
           "yield": {"accepted": len(keep), "accepted_previous_final": len(fin["rows"]), "relabelled": len(fixed),
                     "dropped": len(dropped)},
           "split_accepted": dict(Counter(r["split"] for r in rows)), "retry": True}
    cm.write_once(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{S1X_TAG}.json"),
                  {"rows": rows, "dropped_rows": fin.get("dropped_rows", []) + sorted(dropped),
                   "report": {**fin["report"], "s1x": s1x, "code_only": True}})
    cm.write_once(uc.p(f"s5_final{S1X_TAG}", "gate.json"), rep)
    logger.info("S1x %s: %d relabelled, %d dropped", uc.out.name, len(fixed), len(dropped))
    return {"unit": uc.out.name, "changed": True, **s1x}


def run_g0_style(uc: cm.Cfg) -> dict:
    """Code-only: every selected unsafe G0 row gets its seeded panic (acute) / calm (non-acute) query and
    query_style; gold, labels and memory behaviour unchanged. Writes s5_final_r6 only when something changes."""
    from final_construction.wp_v9_vcm import s4_acute as ac, s4_g0query as gq
    if uc.p(f"s5_final{G0Q_TAG}", "gate.json").exists():
        return {"unit": uc.out.name, "changed": True, "existing": True}
    tag = cm.final_tag(uc)
    fin, gate = cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json")), cm.read(uc.p(f"s5_final{tag}", "gate.json"))
    fams = {f["family_id"]: f for f in cm.plan_raw(uc)["families"]}
    acute_fam = {r["family_id"] for r in fin["rows"] if (r.get("acute_visual_hazard") or {}).get("present")}
    rows, changed, before, b7_fixed = [], [], Counter(), []
    for r in fin["rows"]:
        if r["row_type"] != "G0_closed" or r["risk_label"] != "unsafe":
            rows.append(r)
            continue
        acute = r["family_id"] in acute_fam
        before[r["query"]] += 1
        ev = [(r.get("acute_visual_hazard") or {}).get("evidence") or ""] if acute else []
        q = gq.choose(fams[r["family_id"]], r["row_id"], acute, r["gold_answer"], [e for e in ev if e])
        new = {**r, "query": q, "query_style": gq.style_of_query(q), "query_before_style_fix": r["query"]}
        if acute and (r["b7_subtype"], r["expected_model_action"]) != ("A9", "safety_escalation"):
            new = relabel_arms({**new, "expected_model_action": "safety_escalation", "b7_subtype": "A9",
                                "gold_action_set": ["safety_escalation"],
                                "safety_priority": "acute_incident_or_injury_escalation", "b7_normalized": True})
            b7_fixed.append(r["row_id"])
        res = ac.row_results(fams[r["family_id"]], [x for x in fin["rows"] if x["family_id"] == r["family_id"]
                                                    and x["row_id"] != r["row_id"]] + [new])[r["row_id"]]
        bad = sorted(c for c in (ac.S2, ac.S2B, ac.S2C) if not res[c]["pass"])
        if bad:
            raise RuntimeError(f"{r['row_id']}: generic query fails {bad}")
        rows.append({**new, "checks": {**r["checks"], **{c: res[c] for c in (ac.S2, ac.S2B, ac.S2C)}}})
        changed.append(r["row_id"])
    if not changed:
        return {"unit": uc.out.name, "changed": False, "selection": f"s5_final{tag}"}
    style = Counter(r["query_style"] for r in rows if r["row_id"] in set(changed))
    info = {"changed": sorted(changed), "source_selection": f"s5_final{tag}", "styles": dict(style),
            "acute_b7_normalized_to_a9": b7_fixed, "distinct_before": len(before), "distinct_after": len({r["query"] for r in rows if r["row_id"] in set(changed)})}
    cm.write_once(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{G0Q_TAG}.json"),
                  {"rows": rows, "dropped_rows": fin.get("dropped_rows", []),
                   "report": {**fin["report"], "g0_query_style": info, "code_only": True}})
    cm.write_once(uc.p(f"s5_final{G0Q_TAG}", "gate.json"),
                  {**{k: v for k, v in gate.items() if k != "gate"}, "gate": f"v9 G0 query style code-only fix of "
                   f"s5_final{tag} (no call)", "code_only": True, "g0_query_style": info})
    return {"unit": uc.out.name, "changed": True, **{k: v for k, v in info.items() if k != "changed"},
            "rows": len(changed)}


def _relpath(root: Path, base: Path, p: str) -> str:
    return str((root / p).resolve().relative_to(base.resolve()))


def build_release(cfg: cm.Cfg, name: str = RELEASE_DIR, extra: dict | None = None,
                  units_in: dict[str, cm.Cfg] | None = None) -> dict:
    """Combined train/dev release of the final units (write-once)."""
    from final_construction.wp_v9_vcm import s4_acute as ac
    base = cfg.p("s1").parent
    out = base / name
    if (out / "MANIFEST.json").exists():
        return {"status": "exists", "root": str(out)}
    recs: dict[str, dict[str, list[dict]]] = {m: defaultdict(list) for m in MODES}
    units, missing = {}, []
    for u, uc in (units_in or unit_cfgs(cfg)).items():
        tag = cm.final_tag(uc)
        if not tag:
            continue
        fin = cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json"))["rows"]
        info = {"final_selection": f"s5_final{tag}", "code_only_s1x": tag == S1X_TAG, "release_dir": f"{u}/release{tag}",
                "source_round": dict(Counter(str(r.get("round", 1)) for r in fin)),
                "gen_version": dict(Counter(r.get("gen_version") for r in fin)), "unit_manifest_sha256": {}}
        for mode in MODES:
            root = uc.p(f"release{tag}", mode)
            info["unit_manifest_sha256"][mode] = cm.sha_file(root / "MANIFEST.json")
            info.setdefault("excluded_by_rule", {})[mode] = cm.read(root / "MANIFEST.json").get(
                "excluded_by_rule", {}).get("rows", 0)
            for split_file in sorted(root.glob("*.jsonl")):
                for line in split_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    paths = [rec["query"]] + list(rec["contexts"])
                    missing += [p for p in paths if not (root / p).exists()]
                    rec.update(sample_id=f"{u}/{rec['sample_id']}", query=_relpath(root, base, rec["query"]),
                               contexts=[_relpath(root, base, p) for p in rec["contexts"]])
                    rec["evaluator"] = {**rec["evaluator"], "unit": u, "final_selection": f"s5_final{tag}"}
                    recs[mode][rec["split"]].append(rec)
        main = [r for s in recs["retrieved"].values() for r in s if r["evaluator"]["unit"] == u]
        info["rows"] = dict(Counter(r["split"] for r in main))
        info["row_types"] = dict(Counter(r["row_type"] for r in main))
        info["families"] = len({r["family_id"] for r in main})
        units[u] = info
    if missing:
        raise RuntimeError(f"release image paths missing: {missing[:5]} ({len(missing)})")
    # S1x across the whole release: an acute family has only acute rows (rows of the final selections)
    viol = 0
    for u, uc in (units_in or unit_cfgs(cfg)).items():
        tag = cm.final_tag(uc)
        by: dict[str, list[dict]] = defaultdict(list)
        for r in cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json"))["rows"] if tag else []:
            by[r["family_id"]].append(r)
        viol += sum(len({(r.get("acute_visual_hazard") or {}).get("present", False) for r in rs}) > 1 for rs in by.values())
    out.mkdir(parents=True, exist_ok=True)
    sha = {}
    for mode in MODES:
        d = out if mode == "retrieved" else out / mode
        d.mkdir(parents=True, exist_ok=True)
        for split, rs in sorted(recs[mode].items()):
            p = d / f"{split}.jsonl"
            if p.exists():
                raise RuntimeError(f"write-once output exists: {p}")
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rs, key=lambda x: x["sample_id"])))
            sha[str(p.relative_to(out))] = cm.sha_file(p)
    main = [r for s in recs["retrieved"].values() for r in s]
    manifest = {"release": f"S2 v9 VCM train/dev {name.rsplit('_', 1)[1]} (final units)",
                "added_row_kinds": dict(Counter(f"{r['row_type']}|{r.get('query_style')}|{r['risk']}" for r in
                                                (x for s in recs["retrieved"].values() for x in s)
                                                if r["row_type"].startswith("G1_panic") or r["evaluator"].get(
                                                    "row_id", "").endswith("_generic_safe"))),
                "image_paths": "relative to prod_v9/",
                "query_style": dict(Counter(str(r.get("query_style")) for r in main)),
                "unsafe_g0_distinct_queries": len({r["query_text"] for r in main if r.get("query_style")}),
                "modes": {m: ("." if m == "retrieved" else m) for m in MODES},
                "units": units, "jsonl_sha256": sha,
                "totals": {m: {s: len(v) for s, v in recs[m].items()} for m in MODES},
                "rows": len(main), "families": len({r["family_id"] for r in main}),
                "row_types": dict(Counter(r["row_type"] for r in main)),
                "review_modes": dict(Counter(r["evaluator"].get("review_mode", "gpt_images") for r in main)),
                "excluded_row_types": {t: cm.DROP_REASON for t in cm.DROPPED_ROW_TYPES},
                "general_rule_rows": sum(r["row_type"] in cm.DROPPED_ROW_TYPES for r in main),
                "excluded_by_rule": {m: sum(u.get("excluded_by_rule", {}).get(m, 0) for u in units.values())
                                     for m in MODES},
                "null_memory_behavior": {m: sum(r["evaluator"].get("expected_memory_behavior") is None
                                                for s in recs[m].values() for r in s) for m in MODES},
                "s1x_family_violations": viol, "checks": [*ac.CHECKS, S1X],
                "prereg_sha256": cm.sha_file(cfg.prereg) if cfg.prereg.exists() else None}
    manifest.update(extra or {})
    cm.write_once(out / "MANIFEST.json", manifest)
    return {"root": str(out), **{k: manifest[k] for k in ("totals", "rows", "families", "row_types",
                                                         "s1x_family_violations", "general_rule_rows")}}


def run_release(cfg: cm.Cfg) -> dict:
    """`release_train_dev --profile prod`: S1x per unit, render + export (3 modes) of changed units, combined release."""
    from final_construction.wp_v9_vcm import s6_export, s6_render, s7_release3
    if cfg.profile == "holdout":                       # sealed holdout: rounds 7 + 10, then the balanced release
        return s7_release3.release_v5(cfg, s7_release3.holdout_units(cfg), s7_release3.RELEASE_HOLDOUT)
    units = unit_cfgs(cfg)
    if any(uc.p("s5_final_r10", "gate.json").exists() for uc in units.values()):
        return s7_release3.release_v5(cfg, units, "RELEASE_TRAIN_DEV_v5")
    if any(uc.p("s5_r9", "gate.json").exists() for uc in units.values()):
        return s7_release3.release_v4(cfg, units)
    if any(uc.p("s5_final_r7", "gate.json").exists() for uc in units.values()):
        return s7_release3.release_v3(cfg, units)
    rep = {"s1x": {}}
    for u, uc in units.items():
        if not cm.final_tag(uc):
            continue
        r = run_s1x(uc)
        rep["s1x"][u] = r
        g = run_g0_style(uc)
        rep.setdefault("g0_query_style", {})[u] = g
        if r["changed"] or g["changed"]:
            for mode in MODES:
                s6_render.run_render(uc, mode, 0.2)
                s6_export.run_export(uc, mode)
    rep["release"] = build_release(cfg, RELEASE_DIR_V2)          # v1 stays untouched
    return rep
