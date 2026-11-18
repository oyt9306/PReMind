"""Held-out MONITOR slice of the training release (owner 2026-10-13: no dev split; RELEASE_TRAIN_v9 = train + former
dev). A component-level ~8 % slice of the train rows is never trained on and is evaluated during training ONLY for the
over-abstention monitor / confirmed stop rule and the pre-registered training diagnostics; it never picks a checkpoint
(the final step is reported) and never enters a reported table.

Components: union-find over rows sharing a family_id, pair_id, evidence_pair_id or query image (evaluator
source_image_sha256), so no family / evidence pair / query image straddles the slice boundary. Label = the smallest
family_id of the component (the same in every render mode). Components are taken in the order of
sha256(f"{seed}:{label}") until the slice holds >= ``frac`` of the rows; components larger than ``max_share`` of the
rows are skipped (one huge component must not become the whole slice).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

SUB = {"retrieved": "", "gt_retrieved": "gt_retrieved/", "gt_retrieved_drop": "gt_retrieved_drop/"}
DEFAULT_FRAC, DEFAULT_SEED, MAX_SHARE = 0.08, 0, 0.02


def _ev(r: dict) -> dict:
    e = r.get("evaluator") or {}
    return json.loads(e) if isinstance(e, str) else e


def _keys(r: dict) -> list[str]:
    out = [f"family:{r['family_id']}"]
    for k in ("pair_id", "evidence_pair_id"):
        if r.get(k):
            out.append(f"{k}:{r[k]}")
    sha = _ev(r).get("source_image_sha256")
    if sha:
        out.append(f"img:{sha}")
    return out


def components(rows: list[dict]) -> dict[str, str]:
    """sample_id -> component label (smallest family_id in the component)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        ks = _keys(r)
        for k in ks[1:]:
            a, b = find(ks[0]), find(k)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(find(_keys(r)[0]), []).append(r)
    label = {root: min(x["family_id"] for x in rs) for root, rs in groups.items()}
    return {r["sample_id"]: label[find(_keys(r)[0])] for r in rows}


def select(rows: list[dict], frac: float = DEFAULT_FRAC, seed: int = DEFAULT_SEED,
           max_share: float = MAX_SHARE) -> tuple[set[str], dict]:
    if not 0 < frac < 0.5:
        raise ValueError("monitor slice fraction must be in (0, 0.5)")
    comp = components(rows)
    size: dict[str, int] = {}
    for c in comp.values():
        size[c] = size.get(c, 0) + 1
    n, cap = len(rows), max(1, int(max_share * len(rows)))
    chosen: list[str] = []
    got = 0
    for c in sorted(size, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest()):
        if got >= frac * n:
            break
        if size[c] > cap:
            continue
        chosen.append(c)
        got += size[c]
    cs = set(chosen)
    ids = {sid for sid, c in comp.items() if c in cs}
    by = {r["sample_id"]: r for r in rows}
    info = {"frac": frac, "seed": seed, "n_rows": n, "n_slice": len(ids), "n_components": len(size),
            "n_slice_components": len(cs), "slice_components": sorted(cs),
            "conditions": _count(by[i].get("context_condition") for i in ids),
            "row_types": _count(by[i].get("row_type") for i in ids),
            "ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()}
    return ids, info


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


@lru_cache(maxsize=16)
def _cached(path: str, frac: float, seed: int) -> tuple[frozenset, str]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    ids, info = select(rows, frac, seed)
    return frozenset(ids), json.dumps(info)


def slice_ids(release: Path, mode: str, frac: float = DEFAULT_FRAC, seed: int = DEFAULT_SEED) -> tuple[frozenset, dict]:
    """Monitor-slice sample_ids of ``release``'s train file in ``mode`` (+ info)."""
    ids, info = _cached(str(Path(release) / f"{SUB[mode]}train.jsonl"), frac, seed)
    return ids, json.loads(info)
