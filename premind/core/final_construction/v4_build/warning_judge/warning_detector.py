"""Warning detector facade (owner decision 2026-10-14): does the <answer> give a SPECIFIC warning about the row's
hazard (unsafe rows) / an unnecessary specific warning (safe rows)?

  detect(ref, answer, backend="cue")                       -> wd_cue (frozen wd_cue_v1, CPU)
  detect(ref, answer, backend="judge", judgment=parsed)    -> the parsed Luna judgment (wd_judge.parse_reply);
                                                              judgment None (failed / capped) -> cue, flagged fallback
Output: {warns_specific: bool, reason: str, backend: "cue"|"judge", fallback: bool, evidence}.
Definitions: wd_cue.py (cue rules / guards) and wd_judge.RUBRIC (judge). Reference: wd_reference.HazardRef.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wd_cue  # noqa: E402
from wd_reference import HazardRef, build_ref, refs_for_rows  # noqa: E402

BACKENDS = ("cue", "judge")


def ref_from_json(d: dict) -> HazardRef:
    return HazardRef.from_json(d)


def detect(ref: HazardRef, answer: str, backend: str = "cue", judgment: Optional[dict] = None) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "judge" and judgment is not None:
        return {"warns_specific": bool(judgment["warns_specific"]), "reason": judgment.get("reason", ""),
                "backend": "judge", "fallback": False, "evidence": None}
    out = wd_cue.detect(ref, answer)
    return {**out, "fallback": backend == "judge"}


__all__ = ["BACKENDS", "HazardRef", "build_ref", "refs_for_rows", "ref_from_json", "detect"]
