"""resplit_v2_scene: shared paths, rules and read-only loaders (CPU only, dry run; no release is written).

Owner decisions 2026-10-13 (see PLAN.md): merge RELEASE_TRAIN_DEV_v7h + RELEASE_HOLDOUT_v3j, scene-disjoint
test (room, alameda, nyc), test person targets disjoint from train/dev person targets and Unsplash-only.
Reuses resplit_v1.io_rows (multi-mode row join), resplit_v1.leakage (pHash, union-find components,
cross-split re-check), resplit_v1.features + resplit_v1.optimize (stratified component assignment) and
holdout_audit_v3/results/resplit_keys.json (view groups)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FC = HERE.parent
if str(FC) not in sys.path:
    sys.path.insert(0, str(FC))

from resplit_v1.config import Pool  # noqa: E402
from resplit_v1.io_rows import Rec, load_all  # noqa: E402

B3 = FC.parent
VCM = FC / "wp_v9_vcm"
SRC_TD = VCM / "prod_v9" / "RELEASE_TRAIN_DEV_v7h"
SRC_H = VCM / "holdout_v9_sealed" / "RELEASE_HOLDOUT_v3j"
POOLS = (Pool("RELEASE_TRAIN_DEV_v7h", SRC_TD, "pool_t7h", ("train", "dev")),
         Pool("RELEASE_HOLDOUT_v3j", SRC_H, "pool_h3j", ("holdout",)))
POOL_FILES = (VCM / "canary_v9" / "s0" / "pool.json", VCM / "holdout_v9_sealed" / "s0" / "pool.json")
DESCRIPTIONS = VCM / "holdout_v9_sealed" / "s0" / "descriptions.json"
PERSONA_SPECS = (FC / "eval_image_correction_v1" / "regen_v1" / "PERSONA_SPECS.json",
                 FC / "eval_image_correction_v1" / "regen_v2" / "PERSONA_SPECS.json")
RESPLIT_KEYS = FC / "holdout_audit_v3" / "results" / "resplit_keys.json"
RECORDS = B3 / "records.jsonl"
OUT = HERE / "out"

TEST_SCENES = ("room", "alameda", "nyc")
TRAIN_SCENES = ("berlin", "london", "counter", "bonsai", "bicycle", "kitchen", "garden")
FLAGGED_UNSPLASH = ("aug-person-014", "aug-person-015", "aug-person-018", "aug-person-019", "aug-person-020", "aug-person-028",
                    "aug-person-033")
UNSPLASH_PREFIX = "aug-person-"
OLD_COUNTS = {"train": 848, "dev": 123, "test": 896}
DEV_SHARE = 0.11                     # owner: 10-12 % of train-scene rows
SEED = 20261013
MODES_ORDER = ("retrieved", "gt_retrieved", "gt_retrieved_drop")

# acute hazard type keywords on evaluator.acute_visual_hazard.image_verdict.what (multi-label, lower-cased)
HAZARD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fire_smoke": ("fire", "flame", "smoke", "burning", "burn", "blaze", "scald", "steam", "stove", "pan"),
    "collapse": ("collapse", "lying", "slumped", "unresponsive", "motionless", "unconscious", "faint",
                 "in distress", "injured", "on the floor", "on the ground", "clutching"),
    "electrical": ("electric", "wiring", "wire", "spark", "outlet", "socket", "power strip", "heater", "cable",
                   "cord", "plug"),
    "chemical_fumes": ("chemical", "fume", "vapor", "vapour", "acid", "chlorine", "bleach", "spray", "gas",
                       "toxic", "aerosol", "cleaner", "detergent", "pill", "medication"),
    "machine": ("shredder", "processor", "blender", "machine", "saw", "mixer", "grinder", "drill", "fan blade",
                "mower"),
    "fall": ("fall", "falling", "slip", "trip", "losing balance", "loses balance", "ladder", "stairs",
             "edge", "climb", "tumbl"),
    "water": ("water", "wet", "flood", "spill", "bath", "pool", "sink", "drown"),
    "sharp": ("knife", "blade", "glass", "shard", "scissors", "needle"),
}


def load_rows() -> list[Rec]:
    """All logical rows of both pools (every render mode joined by evaluator.row_id)."""
    return load_all(POOLS)


def read_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def family_scenes() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in POOL_FILES:
        for lst in read_json(f)["families"].values():
            for x in lst:
                if out.setdefault(x["family_id"], x["scene_id"]) != x["scene_id"]:
                    raise ValueError(f"family {x['family_id']} has two scenes")
    return out


def family_records() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for f in POOL_FILES:
        for fam, recs in read_json(f)["records"].items():
            out.setdefault(fam, recs)
    return out


def person_meta() -> dict[str, dict[str, str]]:
    """identity -> {gender, age_group, source, description} from the Qwen descriptions (aug/yollava/multi) and the
    regen persona specs (synth / synth2)."""
    out: dict[str, dict[str, str]] = {}
    for k, v in read_json(DESCRIPTIONS)["descriptions"].items():
        if v.get("fine_type") == "person":
            out[k] = {"gender": v.get("gender"), "age_group": v.get("age_group"), "source": "qwen_descriptions",
                      "description": v.get("description", "")[:160]}
    for f in PERSONA_SPECS:
        for k, v in read_json(f)["identities"].items():
            out[k] = {"gender": v.get("gender"), "age_group": v.get("age_group"), "source": f.parent.name,
                      "apparent_age": v.get("apparent_age"), "replaces": v.get("replaces"),
                      "description": (v.get("old_description_qwen") or "")[:160]}
    return out


def base_sha(r: Rec) -> str:
    """Pre-regeneration source image sha (regen_v1 / regen_v2 rows map back to the original donor image)."""
    for k in ("regen_v1", "regen_v2"):
        if isinstance(r.main.get(k), dict) and r.main[k].get("orig_source_image_sha256"):
            return r.main[k]["orig_source_image_sha256"]
    return r.ev["source_image_sha256"]


def donor_frames() -> dict[str, str]:
    """image sha256 -> raw_cell.source_frame.sha256 (records.jsonl); images without a donor frame are absent."""
    out: dict[str, str] = {}
    with open(RECORDS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sf = (r.get("raw_cell") or {}).get("source_frame")
            if sf and sf.get("sha256"):
                out[r["image_sha256"]] = sf["sha256"]
    return out


def identity_kinds(recs: list[Rec]) -> dict[str, str]:
    """identity -> person / pet / object from every context concept's fine_type."""
    ft: dict[str, str] = {}
    for r in recs:
        for mode_row in r.rows.values():
            for c in mode_row["evaluator"]["context_concepts"]:
                ft.setdefault(c["identity_id"], c["fine_type"])
    pets = {"cat", "dog", "parrot", "bird", "rabbit", "hamster"}
    return {k: "person" if v == "person" else "pet" if v in pets else "object" for k, v in ft.items()}


def targets(r: Rec) -> list[str]:
    return list(r.ev.get("target_identity_ids") or [])


def person_targets(r: Rec, kinds: dict[str, str]) -> list[str]:
    return [t for t in targets(r) if kinds.get(t) == "person"]


def is_acute(r: Rec) -> bool:
    return (r.ev["acute_visual_hazard"].get("image_verdict") or {}).get("severity") == "acute"


def hazard_types(r: Rec) -> list[str]:
    """Keyword types of image_verdict.what for rows whose image verdict is acute (empty otherwise)."""
    v = r.ev["acute_visual_hazard"].get("image_verdict") or {}
    if v.get("severity") != "acute":
        return []
    w = str(v.get("what") or "").lower()
    hit = [k for k, kws in HAZARD_KEYWORDS.items() if any(re.search(r"\b" + re.escape(x), w) for x in kws)]
    return hit or ["other"]


def fact_categories(r: Rec, recs_by_fam: dict[str, list[dict[str, Any]]]) -> list[str]:
    """fact_type:fact_predicate of the family record(s) whose keyphrase matches a decisive keyphrase."""
    kps = [k.lower() for k in r.ev.get("decisive_keyphrases") or []]
    out = []
    fam_recs = recs_by_fam.get(r.main["family_id"], [])
    for kp in kps:
        m = [x for x in fam_recs if x.get("keyphrase", "").lower() == kp] or \
            [x for x in fam_recs if kp in x.get("text", "").lower() or x.get("keyphrase", "").lower() in kp]
        out.append(f"{m[0]['fact_type']}:{m[0]['fact_predicate']}" if m else "unmatched")
    return out


def counter(xs: Any) -> dict[str, int]:
    return dict(sorted(Counter(xs).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
        fh.write("\n")


__all__ = ["HERE", "FC", "POOLS", "OUT", "TEST_SCENES", "TRAIN_SCENES", "FLAGGED_UNSPLASH", "UNSPLASH_PREFIX",
           "OLD_COUNTS", "MODES_ORDER", "DEV_SHARE", "SEED", "HAZARD_KEYWORDS", "RESPLIT_KEYS", "SRC_H", "Rec", "load_rows",
           "read_json", "family_scenes", "family_records", "person_meta", "base_sha", "donor_frames",
           "identity_kinds", "targets", "person_targets", "is_acute", "hazard_types", "fact_categories",
           "counter", "write_json"]
