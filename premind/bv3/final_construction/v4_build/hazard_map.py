"""Acute hazard types per query image (owner 2026-10-13): HAZARD_BY_IMAGE of v4_build/paper_split_fig/
make_split_figure_v1.py (read-only import; per acute image sha256[:6], read from the image verdict text), with the owner's
Option P merge fall + person_down -> person_distress. Acute is image-level. b-lite rows are keyed by their original image.
Used by build_release_v4 (acute_hazard_split rule), leakage_v4 (type-overlap check) and axis_tags (hazard axis)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
MAP_FILE = HERE / "paper_split_fig" / "make_split_figure_v1.py"
_path, _common = list(sys.path), sys.modules.get("common")
sys.path.insert(0, str(MAP_FILE.parent))
from make_split_figure_v1 import HAZARD_BY_IMAGE  # noqa: E402
# make_split_figure_v1 puts resplit_v2_scene on sys.path and imports its `common`; restore both so other modules named
# `common` (e.g. review_v1/common.py) are not shadowed in the same process
sys.path[:] = _path
if _common is None:
    sys.modules.pop("common", None)
else:
    sys.modules["common"] = _common

# F3 (owner 2026-10-14, build_v4f): the acute_add maps (render sha256[:6] -> types) load by default, so acute_types works
# on every acute_add row without injection. They are kept in their own dict: HAZARD_BY_IMAGE stays the base map that
# other modules extend or compare against (ha_common.merged_map, checks_v4c, ext2_run); the base map wins on a key.
ACUTE_ADD_MAP_FILES = (HERE / "acute_add" / "HAZARD_BY_IMAGE_ACUTE_ADD.json",
                       HERE / "acute_add" / "HAZARD_BY_IMAGE_ACUTE_ADD_EXT2.json")
ACUTE_ADD_MAP: dict[str, list[str]] = {}
for _f in ACUTE_ADD_MAP_FILES:
    _m = json.loads(_f.read_text())["map"]
    _clash = sorted(k for k in set(_m) & (set(HAZARD_BY_IMAGE) | set(ACUTE_ADD_MAP))
                    if {**ACUTE_ADD_MAP, **HAZARD_BY_IMAGE}[k] != _m[k])
    if _clash:
        raise ImportError(f"hazard map collision ({_f.name}): {_clash}")
    ACUTE_ADD_MAP.update(_m)
ACUTE_ADD_MAP_SHA256 = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ACUTE_ADD_MAP_FILES}

MERGE = {"fall": "person_distress", "person_down": "person_distress"}
TRAIN_ALLOWED = frozenset({"person_distress"})   # owner rev 5: the only acute type allowed in the training release
MAP_SHA256 = hashlib.sha256(MAP_FILE.read_bytes()).hexdigest()


def _ev(row: dict) -> dict:
    ev = row["evaluator"]
    return json.loads(ev) if isinstance(ev, str) else ev


def image_key(row: dict) -> str:
    return ((row.get("blite_v4") or {}).get("orig_source_image_sha256") or _ev(row)["source_image_sha256"])[:6]


def is_acute(row: dict) -> bool:
    return (_ev(row)["acute_visual_hazard"].get("image_verdict") or {}).get("severity") == "acute"


def acute_types(row: dict, merge: bool = True) -> Optional[list[str]]:
    """Sorted types of an acute row's image (KeyError if an acute image is missing from the map); None if not acute."""
    if not is_acute(row):
        return None
    k = image_key(row)
    t = HAZARD_BY_IMAGE[k] if k in HAZARD_BY_IMAGE else ACUTE_ADD_MAP[k]
    return sorted({MERGE.get(x, x) for x in t} if merge else set(t))


def forbidden(rule: dict[str, Any], split: str) -> set[str]:
    return set(rule["test_only"]) if split in ("train", "dev") else set(rule["train_only"])


__all__ = ["HAZARD_BY_IMAGE", "ACUTE_ADD_MAP", "MERGE", "MAP_FILE", "MAP_SHA256", "ACUTE_ADD_MAP_FILES", "ACUTE_ADD_MAP_SHA256", "image_key", "is_acute", "acute_types", "forbidden", "TRAIN_ALLOWED"]
