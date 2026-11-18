"""Production plan (P1, CPU): every train/dev family of every mapped image, stores seeded per family, and a
scene-aware, dataset-balanced safe/unsafe request crossed with EVERY row type (incl. G1G2_visual).

Scene evidence (open train/dev only; the holdout is never read here): the v1.0.1 rows of the family's image - the
share with physical risk "true" and any visible acute-hazard evidence. An image is unsafe-capable when any of its v1
rows was unsafe. Per row type, unsafe is requested for a seeded subset of unsafe-capable families sized to the target
share of ALL families (TARGET_UNSAFE); a type whose capable share is below target gets every capable family. The
general-rule control is salvaged from a v1 general_rule_only row, so its capability is that of the family's salvage
rows. G0 unsafe = a closed impersonal question about a visible hazard state, B7 A8 (direct_answer + risk true,
hazard prevention) or A9 (safety_escalation, acute) - both legal cells of contract_v1 (B7_TAG/B7_RISK).
The plan order is a seeded shuffle of family ids, so a pilot (`--pilot N`) is the first N families of a random mix.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s1_plan

ROW_TYPES = ("G1_visual", "G1G2_visual", "G0_closed", "general_rule_closed")
TARGET_UNSAFE = {"G1_visual": 0.50, "G1G2_visual": 0.50, "general_rule_closed": 0.50, "G0_closed": 0.35}
COVERAGE_MIN = 0.25                     # production gate: unsafe share per row type after acceptance
# Pilot fix (2026-10-03): measured realisation of an unsafe request (delivered unsafe / requested unsafe, 26 assembled
# pilot families; the rest fell back to safe as risk_infeasible). Production families after the pilot request
# unsafe at DELIVERED_TARGET / realisation (capped by the capable share), via the s1/risk_plan_v2.json overlay
# (`risk_v2` phase); plan.json and the pilot's own requests are unchanged.
PILOT_REALIZATION = {"G1_visual": 6 / 14, "G0_closed": 6 / 10}
DELIVERED_TARGET, REQUEST_CAP = 0.35, 0.90
TARGET_UNSAFE_V2 = {**TARGET_UNSAFE, **{rt: min(REQUEST_CAP, math.ceil(100 * DELIVERED_TARGET / r) / 100)
                                        for rt, r in PILOT_REALIZATION.items()}}


def scene_risk(cfg: cm.Cfg) -> dict[str, dict]:
    """image key -> v1 risk evidence (counts only; open train/dev release files)."""
    out: dict[str, Counter] = {}
    for path in cfg.release_files.values():
        cm.assert_open_only(path, cfg.profile == "holdout")
        for r in cm.read(path)["rows"]:
            c = out.setdefault(Path(r["image_path"]).stem, Counter())
            c["rows"] += 1
            c["unsafe"] += r.get("physical_safety_or_health_risk") == "true"
            c["acute"] += bool((r.get("visual_acute_evidence") or {}).get("present"))
    return {k: {"v1_rows": c["rows"], "v1_unsafe": c["unsafe"], "acute": c["acute"] > 0, "capable": c["unsafe"] > 0}
            for k, c in out.items()}


def risk_plan_prod(families: list[dict], scene: dict, salvage: dict,
                   targets: dict[str, float] | None = None) -> dict[str, dict[str, str]]:
    """family_id -> {row_type: safe|unsafe}, balanced at the dataset level."""
    def capable(f: dict, rt: str) -> bool:
        if rt == "general_rule_closed":
            return any(r["physical_safety_or_health_risk"] == "true" and not r["acute"].get("present")
                       for r in salvage.get(f["family_id"], []))
        return scene.get(f["image_key"], {}).get("capable", False)
    plan = {f["family_id"]: {rt: "safe" for rt in ROW_TYPES} for f in families}
    for rt in ROW_TYPES:
        cap = cm.seeded_order([f["family_id"] for f in families if capable(f, rt)], f"risk:{rt}")
        for fid in cap[:math.ceil((targets or TARGET_UNSAFE)[rt] * len(families))]:
            plan[fid][rt] = "unsafe"
    return plan


def distribution(families: list[dict], plan: dict, scene: dict) -> dict:
    n = max(1, len(families))
    return {"families": len(families),
            "requested_unsafe_share": {rt: round(sum(plan[f["family_id"]][rt] == "unsafe" for f in families) / n, 4)
                                       for rt in ROW_TYPES},
            "scene_capable_share": round(sum(scene.get(f["image_key"], {}).get("capable", False) for f in families) / n, 4),
            "acute_scene_share": round(sum(scene.get(f["image_key"], {}).get("acute", False) for f in families) / n, 4),
            "target_unsafe": TARGET_UNSAFE, "coverage_min_after_acceptance": COVERAGE_MIN}


def run_plan_prod(cfg: cm.Cfg) -> dict:
    out = cfg.p("s1", "plan.json")
    if out.exists():
        return {"status": "exists"}
    w = s1_plan.World(cfg)
    pool = cm.read(cfg.p("s0", "pool.json"))
    fams = [(k, f) for k in sorted(w.images) for f in sorted(w.families[k], key=lambda x: x["family_id"])]
    order = cm.seeded_order([f["family_id"] for _, f in fams], "prod_order")
    by_id = {f["family_id"]: (k, f) for k, f in fams}
    scene = scene_risk(cfg)
    stubs = [{"family_id": fid, "image_key": by_id[fid][0]} for fid in order]
    targets = TARGET_UNSAFE_V2 if cfg.profile == "holdout" else TARGET_UNSAFE     # holdout: post-pilot v2 targets
    risk = risk_plan_prod(stubs, scene, pool["salvage"], targets)
    families = []
    for fid in order:
        k, fam = by_id[fid]
        families.append(s1_plan.family_plan(w, fam, k, s1_plan.candidate(w, k, seed_key=fid), risk[fid]))
    rep = {"families": len(families), "splits": dict(Counter(f["split"] for f in families)),
           "two_concept": sum(f["n_targets"] == 2 for f in families),
           "relaxed": sum(any(i["relaxed"] for i in f["target_info"].values()) for f in families),
           "risk_distribution": distribution(stubs, risk, scene),
           "holdout": "not planned here (sealed script path; see COMMANDS)"}
    cm.write_once(out, {"run_id": cfg.run_id, "profile": cfg.profile,
                        "params": {"K": cm.K, "STORE_SIZE": cm.STORE_SIZE, "N_NEG": cm.N_NEG, "TOPK_CAND": cm.TOPK_CAND,
                                   "SIM_MIX": cm.SIM_MIX, "USE_CROPS": cm.USE_CROPS, "MAX_CROPS": cm.MAX_CROPS,
                                   "seed_key": "family_id", "target_unsafe": targets},
                        "families": families, "report": rep})
    return rep


def run_risk_v2(cfg: cm.Cfg) -> dict:
    """s1/risk_plan_v2.json: TARGET_UNSAFE_V2 requests for every family AFTER the pilot (bound to the plan hash)."""
    out = cfg.p("s1", "risk_plan_v2.json")
    if out.exists():
        return {"status": "exists"}
    plan = cm.read(cfg.p("s1", "plan.json"))
    stubs = [{"family_id": f["family_id"], "image_key": f["image_key"]} for f in plan["families"]]
    risk = risk_plan_prod(stubs, scene_risk(cfg), cm.read(cfg.p("s0", "pool.json"))["salvage"], TARGET_UNSAFE_V2)
    rest = [f["family_id"] for f in plan["families"][cm.PROD_PILOT_N:]]
    rep = {rt: round(sum(risk[f][rt] == "unsafe" for f in rest) / max(1, len(rest)), 4) for rt in ROW_TYPES}
    cm.write_once(out, {"plan_sha256": cm.sha_file(cfg.p("s1", "plan.json")), "targets": TARGET_UNSAFE_V2,
                        "pilot_realization": PILOT_REALIZATION, "delivered_target": DELIVERED_TARGET,
                        "excluded_pilot_families": cm.PROD_PILOT_N, "requested_unsafe_share_after_pilot": rep,
                        "families": {f: risk[f] for f in rest}})
    return {"families": len(rest), "requested_unsafe_share": rep, "targets": TARGET_UNSAFE_V2}
