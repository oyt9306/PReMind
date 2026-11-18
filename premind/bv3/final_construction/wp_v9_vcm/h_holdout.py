"""Sealed VCM holdout build (audit WP-P0b, profile `holdout`, root cm.HOLDOUT_ROOT; owner approval 2026-10-03).

Source: release_v1_0_1/holdout_sealed/final_holdout_v1_0_1.json, verified against its seal manifest (file sha +
aggregate a91b41a5...) before any read. Every holdout stage writes only inside the sealed root; every CLI result is
passed through `redact` (numbers, booleans and identifier-like strings only - never a sentence), errors go to a sealed
log with only the exception type printed; cards / spot-checks refuse the profile.
Stage 0 (`holdout_stage0.sh`): holdout_refs (query image -> governed refs from the benchmark_v2 generation manifests,
the same source as retriever_audit_v1/inputs/provenance) -> pool (the same concept pool as train/dev; identity rule:
ZERO holdout query image shared with train/dev by file stem or sha256; seen/unseen concept flags for the evaluator)
-> describe (reused from the canary when identities and entry images are identical, else Qwen GPU) -> holdout_detect
(YOLOE as retriever_audit_v1, GPU) -> crops -> embed siglip2/wemm/clip (GPU) -> finetype -> plan (production rules,
TARGET_UNSAFE_V2, labels v3 handles) -> retrieve siglip2. Chunks of 50 (`holdout_chunk.sh`): round 1 only, then
render (3 modes), export (CoViP layout) and `holdout_check` (projected tokens > cm.HOLDOUT_BUDGET_STOP -> stop file).
`holdout_seal` writes the hashes-only seal manifest.
"""
from __future__ import annotations

import collections
import datetime
import glob
import json
import re
import traceback
from pathlib import Path

from final_construction.wp_v9_vcm import common as cm

BATCHES = cm.PROD / "benchmark_v2" / "batches"
SEEN_PLANS = [cm.HERE / "prod_v9" / "s1" / "plan.json", cm.OUT_DEFAULT / "s1" / "plan.json"]
PHASES = ("holdout_refs", "holdout_reuse_describe", "holdout_detect", "holdout_check", "holdout_seal")
_SAFE = re.compile(r"^[\w:./@#|+=,\-\[\]]{0,200}$")


def redact(obj: object) -> object:
    """Counts-and-hashes view of a phase result: any string that is not identifier-like (spaces etc.) is dropped."""
    if isinstance(obj, dict):                           # the prelaunch line holds only hashes and counts
        return {str(k): (v if k == "prelaunch_line" else redact(v)) for k, v in obj.items() if _SAFE.match(str(k))}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return obj if _SAFE.match(obj) else "<redacted>"
    return obj


def sealed_error(cfg: cm.Cfg, exc: BaseException) -> str:
    log = cfg.p("s7", "errors", datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(traceback.format_exception(exc)))
    return f"HOLDOUT ERROR {type(exc).__name__} (details only in the sealed log {log.name})"


def verify_source(cfg: cm.Cfg) -> None:
    path = cfg.release_files.get("holdout")
    if set(cfg.release_files) != {"holdout"} or path is None:
        raise RuntimeError("holdout profile: release_files must be exactly the sealed holdout file")
    seal = Path(path).with_name("HOLDOUT_SEAL_MANIFEST_v1_0_1.json")
    if seal.exists():                                    # real source: file sha + aggregate (mock worlds have none)
        m = cm.read(seal)
        if m["aggregate_sha256"] != cm.HOLDOUT_AGGREGATE_V1 or m["files"][Path(path).name] != cm.sha_file(path):
            raise RuntimeError("sealed holdout source does not match its seal manifest")


def _rows(cfg: cm.Cfg) -> list[dict]:
    verify_source(cfg)
    cm.assert_open_only(cfg.release_files["holdout"], holdout_ok=cfg.profile == "holdout")
    return cm.read(cfg.release_files["holdout"])["rows"]


def run_refs(cfg: cm.Cfg) -> dict:
    """Holdout query image -> governed object references, from the generation manifests (as cover.py for open)."""
    if cfg.refs.exists():
        return {"status": "exists"}
    idx: dict[str, list] = collections.defaultdict(list)
    for f in sorted(glob.glob(str(BATCHES / "**" / "manifest_*.json"), recursive=True)):
        try:
            m = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        for c in m.get("calls", []) if isinstance(m, dict) else []:
            refs = (c.get("scenario_build_plan") or {}).get("governed_object_references")
            if refs:
                idx[c.get("sample_id")].append(refs)
    out, status = {}, collections.Counter()
    keep = ("identity_id", "entity_kind", "category", "reference_role", "reference_source", "db_reference_path", "path",
            "reference_equals_db_image", "source_family")
    for key in sorted({Path(r["image_path"]).stem for r in _rows(cfg)}):
        refs = idx.get(key.split("__", 1)[-1])
        status["mapped" if refs else "unmapped"] += 1
        if refs:
            out[key] = [{k: g.get(k) for k in keep} for g in refs[-1]]
    cm.write_once(cfg.refs, out)
    return {"images": sum(status.values()), **status}


def holdout_pool_builder(cfg: cm.Cfg) -> tuple[dict, list[dict], list[str]]:
    """The same concept pool as train/dev (retriever_audit_v1 builders) + holdout query units from the sealed refs."""
    from final_construction.wp_g_final_benchmark_v1 import batch_common_v2 as bc
    from final_construction.wp_v9_vcm import s0_pool
    ad = s0_pool._audit_module("audit_data")
    sources = tuple(s for s in ad.FULL_SOURCES if s[0] in s0_pool.POOL_SOURCES)
    with bc.patched((ad, "FULL_SOURCES", sources)):
        pool = ad.build_full_pool(ad.build_registry_pool())
    merges = pool.pop("__merges__", [])
    refs, pods = cm.read(cfg.refs), {r["instance_id"]: r for r in map(json.loads, ad.PODS_CROPS.open())}
    units = []
    for img in sorted({r["image_path"] for r in _rows(cfg)}):
        tg = refs.get(Path(img).stem) or []
        tids = [g["identity_id"] for g in tg]
        for g in tg:
            iid = g["identity_id"]
            if iid not in pool:
                continue
            ident, gp = pool[iid], ad.resolve(g["path"])
            grp, gsha, gdh = ad.gen_group(ident, gp, pods.get(iid)), ad.sha256(gp), ad.dhash(gp)
            g2 = [i for i, v in enumerate(ident["views"]) if not ad.pixel_shared(v, gp, grp, gsha, gdh)]
            units.append({"unit_id": f"{Path(img).stem}::{iid}", "split": "holdout", "image": img, "target": iid,
                          "n_targets": len(tg), "co_targets": sorted(set(tids) - {iid}), "role": g["entity_kind"],
                          "category": ident["category"], "gen_input": gp, "gen_group": grp, "g2_target_views": g2,
                          "g2_kind": ad._g2_kind([ident["views"][i] for i in g2])})
    return pool, units, merges


def _open_images(cfg: cm.Cfg) -> tuple[set[str], set[str]]:
    stems, shas = set(), set()
    for path in cfg.open_files.values():
        cm.assert_open_only(path)
        for r in cm.read(path)["rows"]:
            stems.add(Path(r["image_path"]).stem)
            shas.add(r.get("image_sha256") or cm.sha_file(Path(r["image_path"])))
    return stems, shas


def run_pool(cfg: cm.Cfg, builder) -> dict:  # noqa: ANN001
    from final_construction.wp_v9_vcm import s0_pool
    verify_source(cfg)
    rep = s0_pool.run_pool(cfg, builder)
    pool = cm.read(cfg.p("s0", "pool.json"))
    stems, shas = _open_images(cfg)
    ov_stem = sum(k in stems for k in pool["images"])
    ov_sha = sum(im["image_sha256"] in shas for im in pool["images"].values())
    rule = {"holdout_query_images": len(pool["images"]), "open_query_images": len(stems),
            "overlap_by_stem": ov_stem, "overlap_by_sha256": ov_sha}
    if not cfg.p("s0", "IDENTITY_RULE.json").exists():
        cm.write_once(cfg.p("s0", "IDENTITY_RULE.json"), rule)
    if ov_stem or ov_sha:
        raise RuntimeError(f"identity rule violated: {ov_stem + ov_sha} holdout query images shared with train/dev")
    seen_t, seen_s, srcs = set(), set(), []
    for p in SEEN_PLANS:
        if Path(p).exists():
            srcs.append(str(p))
            for f in cm.read(p)["families"]:
                seen_t |= {f["concepts"][t]["iid"] for t in f["targets"]}
                seen_s |= {c["iid"] for c in f["concepts"].values()}
    if not cfg.p("s0", "SEEN_CONCEPTS.json").exists():
        cm.write_once(cfg.p("s0", "SEEN_CONCEPTS.json"), {"sources": srcs, "seen_as_target": sorted(seen_t),
                                                          "seen_in_store": sorted(seen_s)})
    return {**(rep if isinstance(rep, dict) else {}), "identity_rule": rule, "seen_sources": len(srcs)}


def seen_flags(cfg: cm.Cfg, fam: dict) -> dict:
    p = cfg.p("s0", "SEEN_CONCEPTS.json")
    s = cm.read(p) if p.exists() else {"seen_as_target": [], "seen_in_store": []}
    t, st = set(s["seen_as_target"]), set(s["seen_in_store"])
    return {"query_image_unseen": True, "targets": {k: {"identity_id": fam["concepts"][k]["iid"],
                                                        "seen_as_target": fam["concepts"][k]["iid"] in t,
                                                        "seen_in_store": fam["concepts"][k]["iid"] in st}
                                                    for k in fam["targets"]}}


def reuse_describe(cfg: cm.Cfg) -> dict:
    """Copy the canary's descriptions when every holdout identity and entry image is identical (no GPU needed)."""
    out = cfg.p("s0", "descriptions.json")
    if out.exists():
        return {"status": "exists"}
    ids = cm.read(cfg.p("s0", "pool.json"))["identities"]
    can = cm.OUT_DEFAULT / "s0"
    cids = cm.read(can / "pool.json")["identities"]
    same = all(i in cids and cids[i].get("entry_sha256") == v.get("entry_sha256") for i, v in ids.items())
    if not same:
        return {"reused": False, "note": "run the GPU describe phase"}
    cm.write_once(out, {**cm.read(can / "descriptions.json"), "reused_from": str(can / "descriptions.json")})
    return {"reused": True, "identities": len(ids)}


def run_detect(cfg: cm.Cfg, device: str = "0") -> dict:  # pragma: no cover - GPU job (env ppercep)
    from final_construction.wp_v9_vcm import s0_pool
    if cfg.audit_crops.exists():
        return {"status": "exists"}
    dc = s0_pool._audit_module("detect_crops")
    classes = sorted({i["category"].replace("_", " ") for i in
                      json.loads((cm.AUDIT_DIR / "build/pool_registry.json").read_text()).values()})
    model, out = dc.load_detector("yoloe", classes, device), {}
    for p in sorted(im["image"] for im in cm.read(cfg.p("s0", "pool.json"))["images"].values()):
        r = model.predict(source=p, conf=dc.CONF, verbose=False, save=False, device=device)[0]
        boxes = [[round(v, 1) for v in xy] + [round(c, 4), classes[int(k)]] for xy, c, k in
                 zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                 if min(xy[2] - xy[0], xy[3] - xy[1]) >= dc.MIN_SIDE]
        out[p] = sorted(boxes, key=lambda b: -b[4])[:dc.MAX_BOXES]
    meta = {"detector": "yoloe", "conf": dc.CONF, "max_boxes": dc.MAX_BOXES, "min_side": dc.MIN_SIDE, "classes": classes,
            "n_images": len(out), "n_without_boxes": sum(not v for v in out.values())}
    cm.write_once(cfg.audit_crops, {"meta": meta, "boxes": out})
    return {k: meta[k] for k in ("n_images", "n_without_boxes")}


def run_plan(cfg: cm.Cfg) -> dict:
    from final_construction.wp_v9_vcm import s1_labels, s1_prod
    rep = s1_prod.run_plan_prod(cfg)                        # production rules; TARGET_UNSAFE_V2 for this profile
    lab = s1_labels.run_labels(cfg, 3, first=0)              # v3 neutral handles for every holdout family
    return {"plan": rep, "labels": lab}


def holdout_check(cfg: cm.Cfg) -> dict:
    k = cfg.chunk
    if not k:
        raise RuntimeError("holdout_check needs --chunk K")
    out = cfg.p("s1", f"HOLDOUT_CHECK_chunk{k:02d}.json")
    if out.exists():
        return cm.read(out)
    root = cfg.p("s1").parent
    n_fam = len(cm.read(cfg.p("s1", "plan.json"))["families"])
    done = [j for j in range(1, n_fam // cm.PROD_CHUNK + 2)
            if (cm.chunk_dir(root, j, "holdout") / "s5_final/gate.json").exists()]      # re-authored chunks: _g2
    if done[:k] != list(range(1, k + 1)):
        raise RuntimeError(f"holdout chunks 1..{k} must all be finalized")
    gen = {f["family_id"]: cm.holdout_gen(cm.Cfg(**{**cfg.__dict__, "out": cm.chunk_dir(root, j, "holdout")}))
           for j in done for f in cm.plan_raw(cm.Cfg(**{**cfg.__dict__, "chunk": j}))["families"]}
    spent, active, superseded = active_spend(cfg, gen)
    fams = list(gen)
    # fix 2026-10-03: the mean is over the ACTIVE trees only (a re-authored chunk's superseded "@h1" spend is sunk);
    # projected = every token spent (incl. superseded and lost) + remaining families x active mean
    mean = sum(active.get(f, 0) for f in fams) / max(1, len(fams))
    projected = spent + max(0, n_fam - len(fams)) * mean
    rep = {"chunk": k, "families_done": len(fams), "families_total": n_fam, "spent": spent,
           "superseded_spent": superseded, "mean_basis": "active trees only (superseded generations excluded)",
           "running_mean_tokens_per_family": round(mean), "projected_total_tokens": round(projected),
           "budget_stop": cm.HOLDOUT_BUDGET_STOP, "stop": projected > cm.HOLDOUT_BUDGET_STOP}
    cm.write_once(out, rep)
    if rep["stop"]:
        cm.write_once(cfg.p("s1", f"HOLDOUT_STOP_chunk{k:02d}.json"), rep)
    return rep


def active_spend(cfg: cm.Cfg, gen: dict[str, int]) -> tuple[int, dict[str, int], int]:
    """(all tokens of this run id, family -> tokens of its ACTIVE holdout generation, superseded tokens): a tag
    "<fid>@h<g>..." belongs to generation g (no "@h" = 1); `gen` maps finished families to their active generation."""
    spent, active, superseded = 0, {}, 0
    if cfg.usage_log.exists():
        for line in cfg.usage_log.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            parts = r["tag"].split("::") if r else []
            if len(parts) < 3 or parts[0] != cfg.run_id:
                continue
            t = int(r["total_tokens"] or 0)
            spent += t
            fid = parts[2].split("@")[0].split("#")[0]
            m = re.search(r"@h(\d+)", parts[2])
            g = int(m.group(1)) if m else 1
            if fid in gen and g != gen[fid]:
                superseded += t
            elif fid in gen:
                active[fid] = active.get(fid, 0) + t
    return spent, active, superseded


def seal(cfg: cm.Cfg) -> dict:
    """Hashes-only seal manifest over every chunk's exported release (no content)."""
    root = cfg.p("s1").parent
    out = root / "HOLDOUT_SEAL_MANIFEST_v9.json"
    if out.exists():
        return cm.read(out)
    chunks = {}
    n_fam = len(cm.read(cfg.p("s1", "plan.json"))["families"])
    for d in [cm.chunk_dir(root, j, "holdout") for j in range(1, n_fam // cm.PROD_CHUNK + 2)]:   # active trees only
        if not d.exists():
            continue
        tag = cm.final_tag(cm.Cfg(**{**cfg.__dict__, "out": d, "s1_dir": root / "s1"}))
        ms = sorted(d.glob(f"release{tag}/*/MANIFEST.json"))
        chunks.update({d.name: {}} if ms else {})
        for m in ms:
            man = cm.read(m)
            chunks[d.name][m.parent.name] = {"manifest_sha256": cm.sha_file(m), "rows": man["rows"],
                                             "jsonl_sha256": man["jsonl_sha256"], "selection": f"s5_final{tag}"}
    body = {"run_id": cfg.run_id, "source_aggregate_sha256_v1": cm.HOLDOUT_AGGREGATE_V1, "chunks": chunks,
            "rows_retrieved": sum(sum(v.get("retrieved", {}).get("rows", {}).values()) for v in chunks.values()),
            "combined_release_manifest_sha256": cm.sha_file(root / "RELEASE_HOLDOUT_v1" / "MANIFEST.json")
            if (root / "RELEASE_HOLDOUT_v1" / "MANIFEST.json").exists() else None}
    cm.write_once(out, {**body, "aggregate_sha256": cm.h(json.dumps(body, sort_keys=True))})
    return {"chunks": len(chunks), "rows_retrieved": body["rows_retrieved"], "aggregate_sha256": cm.read(out)[
        "aggregate_sha256"]}


def prereg_section(cfg: cm.Cfg) -> dict | None:
    if cfg.profile != "holdout":
        return None
    return {"root": str(cm.HOLDOUT_ROOT), "source": str(cm.HOLDOUT_FILE), "source_aggregate_sha256": cm.HOLDOUT_AGGREGATE_V1,
            "ledger": str(cfg.usage_log), "cap": cm.HOLDOUT_CAP, "budget_stop": cm.HOLDOUT_BUDGET_STOP,
            "round1_only": True, "risk_targets": "s1_prod.TARGET_UNSAFE_V2", "labels": "concept_labels_v3 (HANDLES)",
            "identity_rule": "zero holdout query images shared with train/dev (stem and sha256); seen/unseen concept "
                             "flags per row (evaluator key)", "output": "counts and hashes only (h_holdout.redact)",
            "review_mode": "gpt_query_img_texts: GPT review_primary/adjudicate see the query image + the memory TEXTS "
                           "only (no memory images); local Qwen review_second keeps all K+1 images (owner 2026-10-03)",
            "removed_row_types": {t: cm.DROP_REASON for t in cm.DROPPED_ROW_TYPES}, "salvage": "skipped (unused)",
            "row_types": ["G1_visual", "G1G2_visual", "G0_closed", "G1_visual_absent", "G1G2_visual_absent"],
            "checks_added": "S1-S4 (s4_acute) in precheck and assemble",
            "expected_tokens_per_family": HOLDOUT_EXPECTED_PER_FAMILY}


# Expected paid tokens per holdout family (2026-10-03, ledger of the regen round 1, 180 families: 28,379), minus the
# removed general-rule review (0.57 rows x 854) and the memory images of GPT review (primary 9.39 x 331, adjudicate
# 0.35 x 9.33 x 282) under review_mode gpt_query_img_texts.
HOLDOUT_EXPECTED_PER_FAMILY = round(28_379 - 0.57 * 854 - 9.39 * 331 - 0.35 * 9.33 * 282)


def run_phase(cfg: cm.Cfg, a, mock: object | None) -> object | None:  # noqa: ANN001
    """Holdout-specific dispatch; None = use the generic phase. Refuses phases that would show content."""
    ph = a.phase
    if cfg.profile != "holdout":
        if ph in PHASES:
            raise SystemExit(f"{ph} is a holdout phase")
        return None
    if ph in ("cards", "probes", "zs_prepare", "zs_run", "zs_score", "relations", "labels", "labels_v3", "risk_v2",
              "regen_overlay", "salvage_reuse", "pilot_check", "pilot_report", "prod_check"):
        raise SystemExit(f"{ph} is refused for the sealed holdout profile")
    if ph == "holdout_refs":
        return run_refs(cfg)
    if ph == "pool":
        return run_pool(cfg, mock.pool_builder if mock else holdout_pool_builder)
    if ph == "holdout_reuse_describe":
        return reuse_describe(cfg)
    if ph == "holdout_detect":
        return run_detect(cfg)
    if ph == "plan":
        return run_plan(cfg)
    if ph == "holdout_check":
        return holdout_check(cfg)
    if ph == "holdout_seal":
        return seal(cfg)
    return None
