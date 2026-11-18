"""CR on (a) the 81 conflict-subset rows, (b) the 42 clean existing conflict rows, (c) both pooled. CPU only.

CR = acute 'adequate' AND MIE 'applied' AND rule wrong_concept (look-alike content) not True AND not critical, with the
definitions and the image-clustered percentile bootstrap (1000 draws, seed 0) of conflict_metric_v1/compute_cr.py
(imported: cr, critical, adequate, summary). Clusters = query image sha (source_image_sha256), shared across (a) and (b)
because the subset edits memories of v4q acute images.
(a) per system from this package: judges/packets/<sid>_episodes.jsonl (acute state), judges/<sid>_{mie,severity}.jsonl
    (joined by canonical_id + response_sha256) and runs/<sid>/retrieved/records.jsonl (wrong_concept).
(b) conflict_metric_v1/work/EPISODES_SLIM.jsonl (final assembly episodes, bb2) restricted to the 42 rows of
    conflict_audit_v1/CLEAN_CONFLICT_SET.jsonl; MIE labels of the 8 rows in final_eval_assembly/medfix_v1/CHANGED_ROWS.json
    replaced by medfix_v1/mie_medfix_verdicts.jsonl (response-hash checked).
  python cr_conflict.py [--judges <dir>] [--runs <dir>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conflict_common as C  # noqa: E402

V4 = C.FC / "v4_build"
_spec = importlib.util.spec_from_file_location("compute_cr", V4 / "conflict_metric_v1/compute_cr.py")
CR = importlib.util.module_from_spec(_spec);_spec.loader.exec_module(CR)
SLIM = V4 / "conflict_metric_v1/work/EPISODES_SLIM.jsonl"
CLEAN = V4 / "conflict_audit_v1/CLEAN_CONFLICT_SET.jsonl"
MEDFIX = V4 / "final_eval_assembly/medfix_v1"
NOT_SCORED: list[str] = []   # clean rows absent from the final assembly for every system (online-excluded)
ORDER = ["base4b", "base9b", "ours_4b", "ours_9b", "abl_noPF_p12", "abl_noSK_pf", "abl_noSO",
         "gpt61sol_medium", "gpt6luna_high", "opus55_medium_cap1024"]


def cr_strict(e: dict) -> float | None:
    v = CR.cr(e)
    return None if v is None else float(v == 1.0 and CR.critical(e) != 1.0)


def new_rows(sid: str, judges: Path, runs: Path) -> list[dict] | None:
    pk = judges / "packets" / f"{sid}_episodes.jsonl"
    mie_p, sev_p = judges / f"{sid}_mie.jsonl", judges / f"{sid}_severity.jsonl"
    if not (pk.is_file() and mie_p.is_file() and sev_p.is_file()):
        return None
    key = lambda v: (v["canonical_id"], v["response_sha256"])  # noqa: E731
    mie = {key(v): v["label"] for v in C.jl(mie_p)}
    sev = {key(v): v["severity"] for v in C.jl(sev_p)}
    wc = {r["row_id"]: r.get("wrong_concept") for r in C.jl(runs / sid / "retrieved/records.jsonl")
          if (r["mode"], r["arm"], r["kind"]) == ("retrieved", "memory_call", "greedy")}
    out = []
    for e in C.jl(pk):
        out.append({"set": "new", "canonical_id": e["canonical_id"], "local_id": e["local_id"], "image": e["image"],
                    "acute_state": e["acute_state"], "mie": mie.get(key(e)), "severity": sev.get(key(e)),
                    "wrong_concept": wc.get(e["local_id"])})
    return out


def clean_rows() -> dict[str, list[dict]]:
    ids = {r["sample_id"] for r in C.jl(CLEAN)}
    changed = {r["evaluator_row_id"] for r in json.loads((MEDFIX / "CHANGED_ROWS.json").read_text())}
    fix = {(v["model"], v["canonical_id"], v["response_sha256"]): v["label"] for v in C.jl(MEDFIX / "mie_medfix_verdicts.jsonl")}
    by: dict[str, list[dict]] = {}
    for e in C.jl(SLIM):
        if e["local_id"] not in ids:
            continue
        mie = e["mie"]
        if e["canonical_id"] in changed:
            k = (e["model"], e["canonical_id"], e["response_sha256"])
            if k not in fix:
                raise RuntimeError(f"medfix verdict missing/hash mismatch for {k}")
            mie = fix[k]
        by.setdefault(e["model"], []).append({"set": "clean", "canonical_id": e["canonical_id"], "local_id": e["local_id"],
                                              "image": e["image"], "acute_state": e["acute_state"], "mie": mie,
                                              "severity": e["severity"], "wrong_concept": e["wrong_concept"]})
    present = {m: {e["local_id"] for e in rs} for m, rs in by.items()}
    if len({frozenset(v) for v in present.values()}) != 1:
        raise RuntimeError("clean rows present differ across systems")
    NOT_SCORED.clear();NOT_SCORED.extend(sorted(ids - next(iter(present.values()))))
    return by


def block(rows: list[dict]) -> dict:
    pairs = lambda f: [(e["image"], f(e)) for e in rows]  # noqa: E731
    return {"CR": CR.summary(pairs(cr_strict)), "adequate": CR.summary(pairs(CR.adequate)),
            "mie_applied": CR.summary([(e["image"], None if e["mie"] is None else float(e["mie"] == "applied")) for e in rows]),
            "look_alike": CR.summary(pairs(lambda e: float(e["wrong_concept"] is True))),
            "critical": CR.summary(pairs(CR.critical)),
            "n_rows": len(rows), "n_unscorable": sum(cr_strict(e) is None for e in rows)}


def compute(judges: Path = C.JUDGES, runs: Path = C.RUNS) -> dict:
    clean = clean_rows()
    res = {"definition": __doc__, "clean_rows_not_in_assembly": list(NOT_SCORED), "systems": {}}
    for sid in ORDER:
        a, b = new_rows(sid, judges, runs), clean.get(sid)
        res["systems"][sid] = {"new_81": block(a) if a else "pending", "clean_42": block(b) if b else "n/a (not in the final assembly)",
                               "combined": block(a + b) if a and b else "pending" if b else "n/a"}
    return res


def fmt(s) -> str:
    if not isinstance(s, dict) or s.get("estimate") is None:
        return "--"
    return f"{100 * s['estimate']:.1f} [{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}] (n={s['n']}, img={s['n_images']})"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser();p.add_argument("--judges", type=Path, default=C.JUDGES);p.add_argument("--runs", type=Path, default=C.RUNS)
    p.add_argument("--out", type=Path, default=HERE / "results");a = p.parse_args(argv)
    res = compute(a.judges, a.runs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "CR_CONFLICT_v1.json").write_text(json.dumps(res, indent=1) + "\n")
    md = ["| System | CR new 81 | CR clean 42 | CR combined |", "|---|---|---|---|"]
    for sid, d in res["systems"].items():
        md.append(f"| {sid} | " + " | ".join(fmt(d[k]["CR"]) if isinstance(d[k], dict) else d[k] for k in ("new_81", "clean_42", "combined")) + " |")
    (a.out / "CR_CONFLICT_v1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
