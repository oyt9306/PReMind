"""Stage 0: union concept pool -> 448x448 entry images -> Qwen3-VL-8B descriptions -> concept crops -> embeddings.

pool     (CPU) retriever_audit_v1.audit_data full pool (registry + yollava, myvlm, custom101, database_multi merged
               into myvlm by name/sha256, dreambooth, PODS, person_augment; NO CelebA, no archive person_DB_augment),
               minus custom101 quarantined images and fully-quarantined concepts (the 6 registry identities);
               one gallery view per identity (non-generation view if one exists, else the generation input) and
               one 448x448 preprocessing; train/dev query images -> governed refs; v1.0.1 family records (open only).
describe (GPU) Qwen3-VL-8B BF16 per identity: fine_type (fixed enum), gender/age group, visible text, appearance.
crops    (CPU) query-view crops reused from retriever_audit_v1 (YOLOE detections, its crop preprocessing).
embed    (GPU) WeMM-Embedding-2B (main, audit selection) or CLIP ViT-L/14-336 (legacy) loaded exactly as
               retriever_audit_v1/encoders.py; gallery = 448 entry images, queries = query images + crops.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import taxonomy as tx

logger = logging.getLogger(__name__)
POOL_SOURCES = ("yollava", "myvlm", "custom101", "database_multi", "dreambooth", "pods", "person_augment")
RECORD_KEYS = ("record_id", "text", "keyphrase", "fact_type", "fact_predicate")


def _audit_module(name: str):  # noqa: ANN202
    if str(cm.AUDIT_DIR) not in sys.path:
        sys.path.insert(0, str(cm.AUDIT_DIR))
    return __import__(name)


def default_pool_builder(cfg: cm.Cfg) -> tuple[dict, list[dict], list[str]]:
    """(identities, train/dev query units, merges) from the retriever audit builders, restricted to POOL_SOURCES."""
    from final_construction.wp_g_final_benchmark_v1 import batch_common_v2 as bc
    ad = _audit_module("audit_data")
    sources = tuple(s for s in ad.FULL_SOURCES if s[0] in POOL_SOURCES)
    for p in cfg.release_files.values():
        cm.assert_open_only(p, cfg.profile == "holdout")
    with bc.patched((ad, "FULL_SOURCES", sources), (ad, "SPLIT_FILES", dict(cfg.release_files)), (ad, "REFS", cfg.refs)):
        pool = ad.build_full_pool(ad.build_registry_pool())
        merges = pool.pop("__merges__", [])
        units = ad.build_queries(pool)["units"]
    return pool, units, merges


def _quarantine(cfg: cm.Cfg) -> tuple[set[str], set[str]]:
    q = cm.read(cfg.quarantine)
    images = {os.path.realpath(cfg.custom101_dir / e["relative_path"]) for e in q["excluded"]}
    fully = {d.name for d in cfg.custom101_dir.iterdir() if d.is_dir()} - set(q["kept_images_per_concept"])
    return images, fully


def _custom101_concept(path: str, base: Path) -> str | None:
    root = os.path.realpath(base) + "/"
    return path[len(root):].split("/")[0] if path.startswith(root) else None


def gallery_view(views: list[dict]) -> tuple[dict, bool]:
    """Fixed view rule (R2 MINOR, set before the audit): non-generation view if one exists, else the generation input."""
    ordered = sorted(views, key=lambda v: v["path"])
    non_gen = [v for v in ordered if not v.get("ever_generation_input")]
    return (non_gen or ordered)[0], not non_gen


SALVAGE_KEYS = ("row_id", "query", "gold_answer_draft", "physical_safety_or_health_risk", "expected_model_action")


def _salvage(r: dict) -> dict:
    """v1.0.1 general_rule_only row of the family: candidate for the Qwen-rewritten general-rule partner (HYBRID)."""
    return {**{k: r[k] for k in SALVAGE_KEYS}, "b7_subtype": (r.get("b7_action_mapping") or {}).get("subtype"),
            "acute": r.get("visual_acute_evidence") or {"present": False, "evidence": ""},
            "rule": ((r.get("general_knowledge_evidence") or {}).get("rule_statement") or "")}


def _families(cfg: cm.Cfg) -> tuple[dict, dict, dict]:
    fams, records, salvage = {}, {}, {}
    for split, path in cfg.release_files.items():
        cm.assert_open_only(path, cfg.profile == "holdout")
        for r in cm.read(path)["rows"]:
            key = Path(r["image_path"]).stem
            entry = {"family_id": r["family_id"], "person_id": r["person_id"], "scene_id": r["scene_id"], "split": split}
            if entry not in fams.setdefault(key, []):
                fams[key].append(entry)
            records.setdefault(r["family_id"], [{k: rec.get(k) for k in RECORD_KEYS} for rec in r["records"]])
            if r.get("variant_id") == "general_rule_only":
                salvage.setdefault(r["family_id"], []).append(_salvage(r))
    return fams, records, {f: sorted(v, key=lambda x: x["row_id"]) for f, v in salvage.items()}


def run_pool(cfg: cm.Cfg, builder: Callable[[cm.Cfg], tuple] = default_pool_builder) -> dict:
    out = cfg.p("s0", "pool.json")
    if out.exists():
        return {"status": "exists"}
    pool, units, merges = builder(cfg)
    q_images, fully = _quarantine(cfg)
    identities, dropped = {}, {"quarantined_concept": [], "no_views": []}
    for iid, ident in sorted(pool.items()):
        concepts = {_custom101_concept(v["path"], cfg.custom101_dir) for v in ident["views"]} - {None}
        if concepts & fully:
            dropped["quarantined_concept"].append(iid)
            continue
        views = [v for v in ident["views"] if os.path.realpath(v["path"]) not in q_images]
        if not views:
            dropped["no_views"].append(iid)
            continue
        view, is_gen = gallery_view(views)
        entry = preprocess_entry(Path(view["path"]), cfg.p("s0", "entries"))
        identities[iid] = {"identity_id": iid, "kind": ident["role"], "category": ident["category"],
                           "source_family": ident["source_family"], "in_registry": ident["in_registry"],
                           "n_views": len(views), "gallery_src": view["path"], "gallery_src_sha256": view["sha256"],
                           "gallery_is_generation_input": is_gen, **entry}
    fams, records, salvage = _families(cfg)
    images: dict[str, dict] = {}
    for u in units:
        key = Path(u["image"]).stem
        im = images.setdefault(key, {"image": u["image"], "split": u["split"], "targets": [], "dropped_targets": []})
        (im["targets"] if u["target"] in identities else im["dropped_targets"]).append(u["target"])
    for im in images.values():
        im["targets"], im["image_sha256"] = sorted(im["targets"]), cm.sha_file(Path(im["image"]))
    keep = {k: v for k, v in images.items() if v["targets"] and k in fams}
    report = {"identities": len(identities), "kinds": dict(Counter(i["kind"] for i in identities.values())),
              "dropped": {k: len(v) for k, v in dropped.items()}, "images_mapped": len(images),
              "images_kept": len(keep), "images_dropped_all_targets_quarantined": len(images) - len(keep),
              "two_concept_images": sum(len(v["targets"]) == 2 for v in keep.values())}
    cm.write_once(out, {"identities": identities, "dropped": dropped, "merges": merges, "images": keep,
                        "families": {k: fams[k] for k in keep},
                        "records": {f["family_id"]: records[f["family_id"]] for k in keep for f in fams[k]},
                        "salvage": {f["family_id"]: salvage.get(f["family_id"], []) for k in keep for f in fams[k]},
                        "report": report})
    return report


# ---------------------------------------------------------------- describe (GPU job; generate_fn injectable)
DESCRIBE_PROMPT = (
    "Describe the main subject of this image for a personal-concept database (appearance only). Return JSON with: "
    "fine_type = the single best type from the allowed list; gender and age_group only if the subject is a person "
    "(otherwise not_applicable); visible_text = any brand or printed text readable on the subject (empty if none); "
    "description = one or two sentences on distinctive visual appearance (shape, colours, markings, material).")


def describe_schema() -> dict:
    s = {"type": "string"}
    return {"type": "object", "additionalProperties": False,
            "required": ["fine_type", "gender", "age_group", "visible_text", "description"],
            "properties": {"fine_type": {"type": "string", "enum": list(tx.FINE_TYPES)},
                           "gender": {"type": "string", "enum": list(tx.GENDERS)},
                           "age_group": {"type": "string", "enum": list(tx.AGE_GROUPS)},
                           "visible_text": s, "description": s}}


def normalize_description(ident: dict, raw: object) -> dict:
    d = raw if isinstance(raw, dict) else {}
    ft = d.get("fine_type") if d.get("fine_type") in tx.FINE_TYPES else None
    kind, override = ident["kind"], False
    if ft is None or tx.kind_of(ft) != kind:
        override = True
        cat = str(ident["category"]).lower()
        ft = ("person" if kind == "person" else next((t for t in tx.PET_TYPES if t in cat), "other_animal")
              if kind == "pet" else "other_object")
    person = kind == "person"
    return {"fine_type": ft, "fine_type_overridden": override, "valid_json": bool(d),
            "gender": d.get("gender", "unclear") if person else "not_applicable",
            "age_group": d.get("age_group", "unclear") if person else "not_applicable",
            "visible_text": str(d.get("visible_text", "")).strip(), "description": str(d.get("description", ""))}


def run_describe(cfg: cm.Cfg, generate: Callable[[list[dict]], list[str]] | None = None, gpt_fn=None) -> dict:  # noqa: ANN001
    from final_construction.wp_v9_vcm import llm
    out = cfg.p("s0", "descriptions.json")
    if out.exists():
        return {"status": "exists"}
    ids = cm.read(cfg.p("s0", "pool.json"))["identities"]
    reqs = [{"id": i, "instructions": DESCRIBE_PROMPT, "payload": {"task": "describe the concept image"},
             "schema": describe_schema(), "name": "s2v9_describe", "images": [("CONCEPT IMAGE", ids[i]["entry"])],
             "max_out": 600} for i in sorted(ids)]
    got = llm.run_json(cfg, "describe", reqs, gpt_fn=gpt_fn, generate=generate)
    res = {i: normalize_description(ids[i], got.get(i)) for i in sorted(ids)}
    cm.write_once(out, {"backend": cfg.backends["describe"], "model": str(cm.QWEN8B),
                        "prompt_sha256": cm.h(DESCRIBE_PROMPT), "descriptions": res})
    return {"described": len(res), "overridden": sum(r["fine_type_overridden"] for r in res.values()),
            "invalid_json": sum(not r["valid_json"] for r in res.values()),
            "fine_types": dict(Counter(r["fine_type"] for r in res.values()))}


# ---------------------------------------------------------------- crops (CPU: reuse the retriever audit's detections)
def run_crops(cfg: cm.Cfg) -> dict:
    """Query-view crops exactly as retriever_audit_v1 crop mode (the pre-registered main-retriever configuration):
    its YOLOE detections (build/crops.json), boxes in confidence order, raw box crop of the RGB image (no margin),
    at most MAX_CROPS; the whole image is always the first view (FINAL K RULE: no use of the concept count)."""
    from PIL import Image
    out = cfg.p("s0", "query_crops.json")
    if out.exists():
        return {"status": "exists"}
    pool = cm.read(cfg.p("s0", "pool.json"))
    audit = cm.read(cfg.audit_crops)
    missing = [k for k, im in pool["images"].items() if im["image"] not in audit["boxes"]]
    if missing:
        raise RuntimeError(f"{len(missing)} query images absent from the audit crops file")
    res = {}
    for key, im in sorted(pool["images"].items()):
        res[key] = []
        with Image.open(im["image"]) as src:
            rgb = src.convert("RGB")
            for n, b in enumerate(audit["boxes"][im["image"]][:cm.MAX_CROPS]):
                path = cfg.p("s0", "query_crops", f"{cm.h(key)[:16]}_{n}.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                rgb.crop(tuple(b[:4])).save(path)
                res[key].append({"box": b[:4], "conf": b[4], "class": b[5], "crop": str(path)})
    cm.write_once(out, {"source": str(cfg.audit_crops), "source_sha256": cm.sha_file(cfg.audit_crops),
                        "detector_meta": audit["meta"], "max_crops": cm.MAX_CROPS, "crops": res})
    return {"images": len(res), "without_boxes": sum(not v for v in res.values()),
            "crops": sum(len(v) for v in res.values())}


# ---------------------------------------------------------------- embed (GPU job; embed_fn injectable)
def query_inputs(cfg: cm.Cfg) -> list[tuple[str, str]]:
    pool = cm.read(cfg.p("s0", "pool.json"))
    crops = cm.read(cfg.p("s0", "query_crops.json"))["crops"] if cm.USE_CROPS else {}
    q = [(k, im["image"]) for k, im in sorted(pool["images"].items())]
    q += [(f"{k}#crop:{n}", c["crop"]) for k, cs in sorted(crops.items()) for n, c in enumerate(cs)]
    return q


def run_embed(cfg: cm.Cfg, retriever: str, embed: Callable[[list, bool], "object"], meta: dict) -> dict:
    """embed(PIL images, is_query) -> (n, d) L2-normalised array. Gallery = 448 entry images; queries = query images
    and their detector crops (s0/query_crops.json required when USE_CROPS)."""
    import numpy as np
    from PIL import Image, ImageOps
    npz, js = cfg.p("s0", f"embed_{retriever}.npz"), cfg.p("s0", f"embed_{retriever}.json")
    if js.exists():
        return {"status": "exists"}
    pool = cm.read(cfg.p("s0", "pool.json"))
    if cm.USE_CROPS and not cfg.p("s0", "query_crops.json").exists():
        raise RuntimeError("s0/query_crops.json missing: run the crops phase before embed")

    def load(p: str):  # noqa: ANN202
        with Image.open(p) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    ids = sorted(pool["identities"])
    gal = np.asarray(embed([load(pool["identities"][i]["entry"]) for i in ids], False), dtype=np.float32)
    q = query_inputs(cfg)
    qv = np.asarray(embed([load(p) for _, p in q], True), dtype=np.float32)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, gallery=gal, queries=qv)
    cm.write_once(js, {"retriever": retriever, **cm.RETRIEVERS[retriever], "meta": meta, "gallery_ids": ids,
                       "query_keys": [k for k, _ in q], "npz_sha256": cm.sha_file(npz)})
    return {"gallery": len(ids), "queries": len(q), "dim": int(gal.shape[1])}


def load_embeddings(cfg: cm.Cfg, retriever: str) -> tuple[dict, dict]:
    import numpy as np
    meta = cm.read(cfg.p("s0", f"embed_{retriever}.json"))
    npz = cfg.p("s0", f"embed_{retriever}.npz")
    if cm.sha_file(npz) != meta["npz_sha256"]:
        raise RuntimeError(f"embedding file changed: {npz}")
    arr = np.load(npz)
    return ({i: arr["gallery"][n] for n, i in enumerate(meta["gallery_ids"])},
            {k: arr["queries"][n] for n, k in enumerate(meta["query_keys"])})


def real_embedder(retriever: str):  # noqa: ANN201  # pragma: no cover - GPU job
    """Loaded exactly as retriever_audit_v1/encoders.py (SigLIP2 and CLIP-L/336: env omni; WeMM: env qwen35)."""
    enc = _audit_module("encoders")
    fn, meta = {"siglip2": enc.load_siglip2, "wemm": enc.load_wemm, "clip": enc.load_clip}[retriever]("cuda")
    return fn, meta


def preprocess_entry(src: Path, dst_dir: Path) -> dict:
    """One view rule, one preprocessing (R2 B4): EXIF-upright RGB, centre pad to square with cm.PAD_RGB, 448x448 LANCZOS.
    The file name is content-addressed (no identity name reaches any path a model could see)."""
    from PIL import Image, ImageOps, ImageStat
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, hgt = im.size
    gray = im.convert("L")
    strips = [gray.crop((0, 0, w, max(1, hgt // 20))), gray.crop((0, hgt - max(1, hgt // 20), w, hgt)),
              gray.crop((0, 0, max(1, w // 20), hgt)), gray.crop((w - max(1, w // 20), 0, w, hgt))]
    stats = [ImageStat.Stat(s) for s in strips]
    border_mean = sum(s.mean[0] for s in stats) / 4
    border_std = sum(s.stddev[0] for s in stats) / 4
    side = max(w, hgt)
    canvas = Image.new("RGB", (side, side), cm.PAD_RGB)
    canvas.paste(im, ((side - w) // 2, (side - hgt) // 2))
    out = canvas.resize((cm.ENTRY_SIZE, cm.ENTRY_SIZE), Image.LANCZOS)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / f"tmp_{cm.h(str(src))[:16]}.png"
    out.save(tmp, format="PNG")
    digest = cm.sha_file(tmp)
    dst = dst_dir / f"entry_{digest[:20]}.png"
    if dst.exists():
        tmp.unlink()
    else:
        tmp.replace(dst)
    return {"entry": str(dst), "entry_sha256": digest, "orig_w": w, "orig_h": hgt,
            "border_mean": round(border_mean, 3), "border_std": round(border_std, 3)}
