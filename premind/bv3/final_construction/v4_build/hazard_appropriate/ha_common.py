"""Shared helpers for the hazard-appropriateness metric (CPU only; read-only use of v4_build/hazard_map).

Hazard types per acute image = hazard_map.HAZARD_BY_IMAGE (make_split_figure_v1) merged in-process (never written
back) with acute_add/HAZARD_BY_IMAGE_ACUTE_ADD.json and HAZARD_BY_IMAGE_ACUTE_ADD_EXT2.json; fall + person_down ->
person_distress; key = hazard_map.image_key (b-lite rows: blite_v4.orig_source_image_sha256[:6], else
evaluator.source_image_sha256[:6]). Seen types = hazard_map.TRAIN_ALLOWED (person_distress).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
V4B = HERE.parent
FC = V4B.parent
for _p in (str(FC), str(V4B), str(V4B / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_map as HZ  # noqa: E402

RELEASE_V4D = FC / "wp_v9_vcm" / "holdout_v9_sealed" / "RELEASE_TEST_v4d"
ADD_FILES = (V4B / "acute_add" / "HAZARD_BY_IMAGE_ACUTE_ADD.json", V4B / "acute_add" / "HAZARD_BY_IMAGE_ACUTE_ADD_EXT2.json")
SEEN_TYPES = frozenset(HZ.TRAIN_ALLOWED)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def merged_map() -> dict[str, list[str]]:
    m = dict(HZ.HAZARD_BY_IMAGE)
    for f in ADD_FILES:
        add = json.loads(f.read_text())["map"]
        clash = set(add) & set(m)
        if clash:
            raise ValueError(f"{f.name}: keys collide with the base map: {sorted(clash)[:5]}")
        m.update(add)
    return m


MAP = merged_map()


def ev(row: dict) -> dict:
    e = row["evaluator"]
    return json.loads(e) if isinstance(e, str) else e


def acute_types(row: dict) -> Optional[list[str]]:
    """Merged, sorted hazard types of an acute row (KeyError if unmapped); None for a non-acute row."""
    if not HZ.is_acute(row):
        return None
    return sorted({HZ.MERGE.get(x, x) for x in MAP[HZ.image_key(row)]})


def visible_hazard(row: dict) -> str:
    return ((ev(row)["acute_visual_hazard"].get("image_verdict") or {}).get("what") or "").strip()


def seen_split(types: list[str]) -> str:
    """'seen' iff every type was seen in training (mixed images count as unseen, as in awareness.py)."""
    return "seen" if all(t in SEEN_TYPES for t in types) else "unseen"


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def release_rows(release: Path = RELEASE_V4D, split: str = "holdout") -> list[dict]:
    return jl(Path(release) / f"{split}.jsonl")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split((text or "").strip()) if s.strip()]


__all__ = ["HZ", "FC", "V4B", "HERE", "RELEASE_V4D", "SEEN_TYPES", "MAP", "merged_map", "ev", "acute_types",
           "visible_hazard", "seen_split", "jl", "release_rows", "sentences"]
