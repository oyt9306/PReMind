"""Read-only loading of the source releases (all render modes)."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import MODES, Pool

logger = logging.getLogger(__name__)


@dataclass
class Rec:
    """One logical row (evaluator.row_id) of a source pool with its rows in every render mode."""
    uid: str
    pool: Pool
    orig_split: str
    order: int
    rows: dict[str, dict[str, Any]]      # mode -> release row (main mode "retrieved" always present)
    notes: dict[str, Any] = field(default_factory=dict)   # harmonization notes (ROW_PROVENANCE.jsonl)

    @property
    def main(self) -> dict[str, Any]:
        return self.rows["retrieved"]

    @property
    def ev(self) -> dict[str, Any]:
        return evaluator(self.main)


def evaluator(row: dict[str, Any]) -> dict[str, Any]:
    ev = row["evaluator"]
    return json.loads(ev) if isinstance(ev, str) else ev


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def release_files(release: Path, split: str) -> dict[str, Path]:
    return {mode: release / sub / f"{split}.jsonl" for mode, sub in MODES.items()}


def load_pool(pool: Pool, start: int = 0) -> list[Rec]:
    """Every row of `pool` (file order kept); variant rows are joined to main rows by evaluator.row_id."""
    recs: list[Rec] = []
    for split in pool.splits:
        files = release_files(pool.release, split)
        main = _read_jsonl(files["retrieved"])
        by_mode: dict[str, dict[str, dict]] = {}
        for mode, f in files.items():
            if mode == "retrieved":
                continue
            if not f.is_file():
                raise FileNotFoundError(f)
            by_mode[mode] = {evaluator(r)["row_id"]: r for r in _read_jsonl(f)}
        main_ids = {evaluator(r)["row_id"] for r in main}
        for mode, d in by_mode.items():
            extra = set(d) - main_ids
            if extra:
                raise ValueError(f"{pool.name}/{mode}/{split}: {len(extra)} rows not in the main file")
        for r in main:
            uid = evaluator(r)["row_id"]
            if r.get("split") != split:
                raise ValueError(f"{pool.name}: row {uid} has split {r.get('split')} in {split}.jsonl")
            rows = {"retrieved": r, **{m: d[uid] for m, d in by_mode.items() if uid in d}}
            recs.append(Rec(uid, pool, split, start + len(recs), rows))
    logger.info("loaded %s: %d rows", pool.name, len(recs))
    return recs


def load_all(pools: tuple[Pool, ...]) -> list[Rec]:
    recs: list[Rec] = []
    for p in pools:
        recs.extend(load_pool(p, start=len(recs)))
    dup = len(recs) - len({r.uid for r in recs})
    if dup:
        raise ValueError(f"{dup} duplicate evaluator.row_id across pools")
    return recs


def source_info(pools: tuple[Pool, ...]) -> dict[str, Any]:
    """sha256 of every source file (manifest provenance)."""
    out: dict[str, Any] = {}
    for p in pools:
        files = {"MANIFEST.json": p.release / "MANIFEST.json"}
        for split in p.splits:
            for mode, f in release_files(p.release, split).items():
                files[str(f.relative_to(p.release))] = f
        out[p.name] = {"path": str(p.release), "sha256": {k: sha_file(v) for k, v in sorted(files.items())}}
    return out


__all__ = ["Rec", "evaluator", "sha_file", "load_pool", "load_all", "source_info", "release_files"]
