"""Regeneration of the pilot (30) and production chunk 1 (50) under the current production rules (owner 2026-10-03).

Runs: `--chunk-regen pilot` -> prod_v9/regen_pilot/, `--chunk-regen 1` -> prod_v9/regen_chunk01/; round 1 only
(--no-review-round semantics), the same family ids, every paid call tagged "@g2" (common.rid), rows carry
gen_version "g2" (originals "g1"). Old outputs are never touched (separate trees, write-once).
Overlays (all frozen files are read, none is modified):
  * plan.json, retrieval_siglip2.json, relations: as for every production chunk (shared prod_v9/s1).
  * chunk01: concept_labels_v3.json + risk_plan_v2.json already cover these families (computed for plan[30:]).
  * pilot: those two files do NOT contain the pilot (they start after it), so `regen_overlay` writes the NEW write-once
    s1/regen_pilot_overlay.json with the SAME functions and constants (s1_labels.labels_for with the v3 HANDLES;
    s1_prod.risk_plan_prod over the whole plan with TARGET_UNSAFE_V2), after verifying that this recomputation
    reproduces the stored labels_v3 / risk_plan_v2 entries of every post-pilot family exactly.
Salvage (local Qwen, unpaid) is reused when the salvage request of the regen family is identical to the original's
(`salvage_reuse`, CPU); everything paid is regenerated. `release_sources` picks, per family, ONE generation for the
final manifest: the regen tree once it is finalized, else the original - never mixed inside a family. Within that
tree the retry selection s5_final_r2/ is used when it exists (production retry round), else s5_final/.
"""
from __future__ import annotations

import json
from pathlib import Path

from final_construction.wp_v9_vcm import common as cm

OVERLAY = "regen_pilot_overlay.json"


def regen_cfg(cfg: cm.Cfg, target: str) -> cm.Cfg:
    return cm.Cfg(**{**cfg.__dict__, "regen": target, "chunk": None, "pilot": None, "s1_dir": cfg.p("s1"),
                     "s0_dir": cfg.s0_dir or cfg.p("s0"),
                     "out": cfg.p("s1").parent / f"regen_{target}"})


def source_cfg(cfg: cm.Cfg) -> cm.Cfg:
    """The original generation of a regen target (pilot tree = prod_v9/, chunk 1 = prod_v9/chunk_01/)."""
    root = cfg.p("s1").parent
    if cfg.regen == "pilot":
        return cm.Cfg(**{**cfg.__dict__, "regen": None, "pilot": cm.PROD_PILOT_N, "out": root, "s1_dir": None})
    return cm.Cfg(**{**cfg.__dict__, "regen": None, "chunk": int(cfg.regen[-2:]), "out": root / cfg.regen.replace(
        "chunk", "chunk_")})


def _recompute(cfg: cm.Cfg) -> tuple[dict, dict]:
    from final_construction.wp_v9_vcm import s1_labels, s1_prod
    plan = cm.read(cfg.p("s1", "plan.json"))
    stubs = [{"family_id": f["family_id"], "image_key": f["image_key"]} for f in plan["families"]]
    risk = s1_prod.risk_plan_prod(stubs, s1_prod.scene_risk(cfg), cm.read(cfg.p("s0", "pool.json"))["salvage"],
                                  s1_prod.TARGET_UNSAFE_V2)
    labels = {f["family_id"]: s1_labels.labels_for(f, s1_labels.HANDLES) for f in plan["families"]}
    return risk, labels


def run_overlay(cfg: cm.Cfg) -> dict:
    out = cfg.p("s1", OVERLAY)
    if out.exists():
        return {"status": "exists"}
    risk, labels = _recompute(cfg)
    rv2, lv3 = cm.read(cfg.p("s1", "risk_plan_v2.json")), cm.read(cfg.p("s1", "concept_labels_v3.json"))
    bad = [f for f, v in rv2["families"].items() if risk[f] != v] + \
        [f for f, v in lv3["families"].items() if labels[f] != v]
    if bad:
        raise RuntimeError(f"recomputation does not reproduce risk_plan_v2 / labels_v3 for {bad[:5]}")
    pilot = [f["family_id"] for f in cm.read(cfg.p("s1", "plan.json"))["families"][:cm.PROD_PILOT_N]]
    cm.write_once(out, {"plan_sha256": cm.sha_file(cfg.p("s1", "plan.json")),
                        "risk_plan_v2_sha256": cm.sha_file(cfg.p("s1", "risk_plan_v2.json")),
                        "labels_v3_sha256": cm.sha_file(cfg.p("s1", "concept_labels_v3.json")),
                        "verified_against_post_pilot_families": len(rv2["families"]),
                        "families": {f: {"risk_plan": risk[f], "labels": labels[f]} for f in pilot}})
    return {"families": len(pilot), "verified": len(rv2["families"])}


def apply_pilot_overlay(cfg: cm.Cfg, plan: dict) -> None:
    p = cfg.p("s1", OVERLAY)
    if not p.exists():
        raise RuntimeError(f"pilot regeneration needs {p}: run `regen_overlay --profile prod` first")
    doc = cm.read(p)
    if doc["plan_sha256"] != cm.sha_file(cfg.p("s1", "plan.json")):
        raise RuntimeError(f"{p} was computed for another plan.json")
    for fam in plan["families"]:
        o = doc["families"][fam["family_id"]]
        fam["risk_plan"] = o["risk_plan"]
        for k, r in o["labels"].items():
            fam["concepts"][k].update(name=r["name"], relation=r["relation"], handle=r["handle"])


def _salvage_key(cfg: cm.Cfg, fam: dict) -> str:
    from final_construction.wp_v9_vcm import s2_local
    row = s2_local.salvage_row(cfg, fam)
    req = s2_local.salvage_request(fam, row) if row else None
    return json.dumps(req and {k: req[k] for k in ("instructions", "payload", "images")}, sort_keys=True)


def salvage_reuse(cfg: cm.Cfg) -> dict:
    """Copy the original salvage output (local Qwen) when the regen family's salvage request is identical (unused
    since the general-rule control was removed, 2026-10-03: then a no-op)."""
    if cm.no_general_rule(cfg):
        return {"reused": 0, "left_for_gpu_salvage": 0, "skipped": "general-rule control removed"}
    src = source_cfg(cfg)
    old = {f["family_id"]: f for f in cm.plan_raw(src)["families"]}
    reused, todo = 0, 0
    for fam in cm.plan_raw(cfg)["families"]:
        fid = fam["family_id"]
        dst, orig = (c.p("s2", "salvage", cm.safe_name(fid) + ".json") for c in (cfg, src))
        if dst.exists():
            continue
        if orig.exists() and _salvage_key(cfg, fam) == _salvage_key(src, old[fid]):
            cm.write_once(dst, {**cm.read(orig), "reused_from": str(orig)})
            reused += 1
        else:
            todo += 1
    return {"reused": reused, "left_for_gpu_salvage": todo}


def release_sources(cfg: cm.Cfg) -> dict[str, dict]:
    """family -> {"gen_version", "tree", "final", "rows", "source_round"} for the final manifest: a finalized regen tree
    replaces the original family wholesale; families of an unfinished regen keep the original (never mixed); inside
    the tree, the retry selection (s5_final_r2/, rows_<main>_final_r2.json) when finalized, else s5_final/."""
    root = cfg.p("s1").parent
    out = {}
    for target, (a, b) in cm.REGEN_TARGETS.items():
        rc = regen_cfg(cfg, target)
        done = rc.p("s5_final", "gate.json").exists()
        tree = rc.out if done else source_cfg(rc).out
        for f in cm.read(cfg.p("s1", "plan.json"))["families"][a:b]:
            out[f["family_id"]] = {"gen_version": cm.GEN_REGEN if done else cm.GEN_ORIGINAL, "tree": str(tree)}
    for d in sorted(root.glob("chunk_*")):
        k = int(d.name.split("_")[1])
        a = cm.PROD_PILOT_N + cm.PROD_CHUNK * (k - 1)
        for f in cm.read(cfg.p("s1", "plan.json"))["families"][a:a + cm.PROD_CHUNK]:
            out.setdefault(f["family_id"], {"gen_version": cm.GEN_ORIGINAL, "tree": str(d)})
    for fid, src in out.items():                       # retry selection (s5_final_r2) per tree when finalized
        tc = cm.Cfg(**{**cfg.__dict__, "out": Path(src["tree"]), "s1_dir": cfg.p("s1")})
        tag = cm.final_tag(tc)
        g = tc.p(f"s5_final{tag}", "gate.json")
        src.update(final=f"s5_final{tag}", rows=str(tc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json")),
                   source_round=(cm.read(g).get("family_source_round", {}).get(fid, 1) if tag and g.exists() else 1),
                   repaired_rows=(sorted(r for r in cm.read(g).get("repaired_rows", []) if f":{fid}:" in r)
                                  if tag in ("_r3", "_r4", "_r5", "_r6", "_r7", "_r8") else []))
    return out
