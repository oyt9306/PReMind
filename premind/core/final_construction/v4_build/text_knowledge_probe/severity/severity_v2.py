"""Harm-severity re-judge v2 (owner/orchestrator 2026-10-15): v1 over-called ~9/25 sampled 'critical' verdicts.
Same item / prompt / reply format, client, cap, ledger, cache and clip as severity_v1 (imported, not edited); only the
system prompt changes to RUBRIC_SEVERITY_v2.md (critical = death / serious injury LIKELY, not merely possible).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import severity_v1 as S1
from severity_v1 import (DEFAULT_MODEL, LABELS, SYSTEM_HEAD, FakeBackend, build_item, finalize,  # noqa: F401
                         item_key, item_text, judge, parse_reply)

VERSION = "tkp_severity_v2"
RUBRIC_PATH = S1.HERE / "RUBRIC_SEVERITY_v2.md"
RUBRIC_SHA256 = "b7f608885900eabeb68b49e249bf5310d049c2f5dc288ff0887dfe7b36caaa5f"


def rubric_text() -> str:
    raw = RUBRIC_PATH.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != RUBRIC_SHA256:
        raise RuntimeError(f"{RUBRIC_PATH} sha256 {got} != pinned {RUBRIC_SHA256} (rubric changed: new version)")
    return raw.decode()


class SeverityLunaBackend(S1.SeverityLunaBackend):
    """severity_v1.SeverityLunaBackend with the v2 rubric (prefix, ledger purpose, cache key)."""

    def __init__(self, usage_log: Path, *, usd_cap: float, model: str = DEFAULT_MODEL, concurrency: int = 8,
                 cache_path: Optional[Path] = None, call: Any = None) -> None:
        super().__init__(usage_log, usd_cap=usd_cap, model=model, concurrency=concurrency, cache_path=None, call=call)
        self.prefix = SYSTEM_HEAD + rubric_text()
        self.purpose, self.rubric, self.rubric_sha = VERSION, VERSION, RUBRIC_SHA256
        self.cache_path, self.cache = (Path(cache_path) if cache_path else None), {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("model") == model and rec.get("rubric_sha256") == self.rubric_sha:
                    self.cache[rec["key"]] = rec["reply"]


def estimate_usd(texts: list[str], model: str = DEFAULT_MODEL, out_tokens: int = 343) -> dict[str, float]:
    """As severity_v1.estimate_usd with the v2 system prompt."""
    AP = S1.AP
    sys_tok = AP.text_tokens(SYSTEM_HEAD + rubric_text())
    tin = sum(sys_tok + AP.text_tokens(t) for t in texts)
    hi = AP.ASSUMED_REASONING_TOKENS["medium"] + 120
    return {"n_calls": len(texts), "input_tokens": tin,
            "usd_measured_rate": round(AP.cost_usd(model, tin, out_tokens * len(texts)), 4),
            "usd_upper_bound": round(AP.cost_usd(model, tin, hi * len(texts)), 4)}


__all__ = ["VERSION", "RUBRIC_SHA256", "LABELS", "build_item", "item_text", "item_key", "parse_reply", "finalize",
           "SeverityLunaBackend", "FakeBackend", "judge", "estimate_usd"]
