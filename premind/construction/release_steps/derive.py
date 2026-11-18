"""Row-drop release step (owner 2026-10-05): RELEASE_*_v7d/v3d -> v7e/v3e and v7e/v3e -> v7f/v3f.

v7e drops 39 rows: every row on an image flagged L2b (ambiguous acute label) and implicit-request (G1_panic_memory) rows
failing the strict visible-acute-emergency rule (flags of the v7d label audit). v7f drops the 18 rows of the
memory-need audit (G0 / implicit-request rows on the same image whose Stage-1 memory need is not decidable).
The drop lists are the recorded audit decisions (inputs/drop_list_v7e.json, inputs/drop_list_v7f.json); the audits
themselves (GPT / local-VLM review) are not re-run. Kept partners of a dropped row get evaluator.pair_incomplete = true.
Image paths are unchanged. Write-once: refuses an existing output directory.

    python -m premind.construction.release_steps.derive --step v7e \
        --src-td .../RELEASE_TRAIN_DEV_v7d --src-ho .../RELEASE_HOLDOUT_v3d --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"v7e": ("RELEASE_TRAIN_DEV_v7e", "RELEASE_HOLDOUT_v3e"), "v7f": ("RELEASE_TRAIN_DEV_v7f", "RELEASE_HOLDOUT_v3f")}
PAIRS = ("evidence_pair_id", "pair_id")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def derive(src: Path, dst: Path, drop: set, note: dict) -> dict:
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst} exists (write-once)")
    dst.mkdir()
    shas, counts = {}, {}
    for f in sorted(src.rglob("*.jsonl")):
        rel = f.relative_to(src)
        if rel.parts[0] == "media":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel.name in ("MEDIA_INDEX.jsonl",):
            shutil.copy2(f, out)
        else:
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            key = (lambda r: r.get("row_id") or r.get("evaluator", {}).get("row_id"))
            dropped_pairs = {(k, r.get(k)) for r in rows if key(r) in drop for k in PAIRS if r.get(k)}
            kept = []
            for r in rows:
                if key(r) in drop:
                    continue
                if "evaluator" in r and any((k, r.get(k)) in dropped_pairs for k in PAIRS if r.get(k)):
                    r = {**r, "evaluator": {**r["evaluator"], "pair_incomplete": True}}
                kept.append(r)
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
            counts[str(rel)] = (len(rows), len(kept))
        shas[str(rel)] = sha(out)
    m = json.loads((src / "MANIFEST.json").read_text())
    m.update({"release": dst.name, "derived_from": [src.name], "jsonl_sha256": shas, "row_changes_v7e": note,
              "row_counts_v7e": counts})
    (dst / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    return counts


def drop_ids(note: dict) -> set:
    return {x for k, v in note.items() if k.startswith("dropped_") and isinstance(v, list) for x in v}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=sorted(NAMES))
    ap.add_argument("--src-td", required=True, type=Path)
    ap.add_argument("--src-ho", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path, help="parent dir of the two new release dirs")
    a = ap.parse_args(argv)
    note = json.loads((HERE / "inputs" / f"drop_list_{a.step}.json").read_text())
    drop = drop_ids(note)
    assert len(drop) == note["n_dropped"], (len(drop), note["n_dropped"])
    td, ho = NAMES[a.step]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(derive(a.src_td, a.out_dir / td, drop, note), indent=1))
    print(json.dumps(derive(a.src_ho, a.out_dir / ho, drop, note), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
