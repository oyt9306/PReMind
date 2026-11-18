"""Hazard-APPROPRIATENESS metrics on any eval_v1 output dir (owner 2026-10-14; CPU only; read-only on the runs).

Extends v4_build/tables/hazard_metrics.py (imported, not edited): rows = image-level acute rows of the run that have a
key-action entry (KEY_ACTIONS_v4d.jsonl); hazard_addressed / memory condition come from hazard_metrics.run_metrics.
  hazard_addressed   existing detector (eval_v1/hazard_match URGENT | soft hazard pattern) on <answer>
  appropriate_any    >= 1 type-specific key action covered (rows without a specific key action: undefined)
  appropriate_all    every key action covered
  generic_only       addresses the hazard, but only with generic actions (emergency call, step back, keep away ...)
  contraindicated    a harmful / contraindicated action for this hazard is present
Verdict source: --source fallback (cue_fallback.py) or the Claude judgments joined by ingest_judgments.py. Degenerate
outputs count as not covered / not generic / not contraindicated (as hazard_metrics forces addressed = False).
Splits: all; seen (all types in hazard_map.TRAIN_ALLOWED = person_distress) vs unseen (mixed images = unseen); per
type; memory condition. Wilson 95% CIs. Paired base -> ours (--pair i:j, indices into --run): delta with a row-level
bootstrap 95% CI (10,000 draws, seed 0) and exact McNemar (awareness.py helpers), plus the PREREG_APPROPRIATE H4-H6.
  python -I appropriate_metrics.py --run <base run> <ours run> --pair 0:1 --out <dir> [--setting memory_call,retrieved,greedy]
Outputs (write-once dir): appropriate_metrics.json / .csv / .tex.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ha_common as C  # noqa: E402
import cue_fallback as CF  # noqa: E402
import key_actions as KA  # noqa: E402
import hazard_metrics as HMX  # noqa: E402  (read-only import)
import awareness as AW  # noqa: E402  (read-only import: boot, mcnemar_exact)
from eval_v1 import acute_rule_v4 as AR  # noqa: E402
from eval_v1 import hazard_match as HM  # noqa: E402

KEYS = ("hazard_addressed", "appropriate_any", "appropriate_all", "generic_only", "contraindicated")
KA_DEFAULT = HERE / "out" / "key_actions_v4d" / "KEY_ACTIONS_v4d.jsonl"
SETTING = ("memory_call", "retrieved", "greedy")
NI_MARGIN = 0.05
Verdicts = dict[tuple[str, str], dict]


def load_ka(path: Optional[Path] = None) -> dict[str, dict]:
    p = Path(path) if path else KA_DEFAULT
    if p.exists():
        return KA.load_key_actions(p)
    return {r["row_id"]: r for r in KA.extract_release()}       # same deterministic extraction, in memory


def collect(run: Path, setting: tuple[str, str, str], ka: dict[str, dict]) -> list[dict[str, Any]]:
    """Acute rows of ``run`` with a key-action entry: row_id, types, seen, condition, hazard_addressed, degenerate, answer."""
    hm = HMX.run_metrics(Path(run), AR.DEFAULT, HMX.TRAIN_DEFAULT, setting, keep_rows=True)
    per = [v for v in hm["per_row"] if v["acute"] and v["sample_id"] in ka]
    ids = [v["sample_id"] for v in per]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{run}: several outputs per row in setting {setting} (use a greedy setting)")
    texts = {x["row_id"]: HM.answer_text(x.get("stage2") or "") for x in C.jl(Path(run) / "texts.jsonl")
             if (x["arm"], x["mode"], x["kind"]) == setting}
    degen = {x["row_id"]: bool(x["degenerate"]) for x in C.jl(Path(run) / "records.jsonl")
             if (x["arm"], x["mode"], x["kind"]) == setting}
    out = []
    for v in per:
        k = ka[v["sample_id"]]
        out.append({"run": str(run), "row_id": v["sample_id"], "types": k["types"], "seen": k["seen"],
                    "condition": v["condition"], "hazard_addressed": v["hazard_addressed"],
                    "degenerate": degen.get(v["sample_id"], False), "answer": texts.get(v["sample_id"], "")})
    return out


def score(rows: list[dict], ka: dict[str, dict], verdicts: Optional[Verdicts] = None) -> list[dict]:
    """Copies of ``rows`` with the four appropriateness values (judge verdicts if given, else the cue fallback)."""
    out = []
    for r in rows:
        k = ka[r["row_id"]]
        v = verdicts.get((r["run"], r["row_id"])) if verdicts is not None else CF.judge_row(k, r["answer"])
        if v is None:
            raise KeyError(f"no verdict for {r['run']} / {r['row_id']}")
        spec = [a["label"] for a in k["actions"] if a["specific"]]
        cov = v["actions"]
        vals = {"appropriate_any": any(cov[x] for x in spec) if spec else None,
                "appropriate_all": all(cov.values()) if cov else None,
                "generic_only": bool(v["generic_only"]), "contraindicated": bool(v["contraindicated"])}
        if r["degenerate"]:
            vals = {x: (None if vals[x] is None else False) for x in vals}
        out.append({**r, **vals, "covered": cov})
    return out


def groups_of(scored: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for v in scored:
        g["all"].append(v)
        g[v["seen"]].append(v)
        for t in v["types"]:
            g[f"type:{t}"].append(v)
        g[f"condition:{v['condition']}"].append(v)
    return dict(sorted(g.items(), key=lambda x: (x[0] != "all", x[0] not in ("seen", "unseen"), x[0])))


def run_summary(scored: list[dict]) -> dict[str, Any]:
    return {g: {"n": len(vs), **{k: HMX.cell([v[k] for v in vs]) for k in KEYS}} for g, vs in groups_of(scored).items()}


def paired(base: list[dict], ours: list[dict], key: str, pop: Callable[[dict], bool]) -> dict[str, Any]:
    ob = {v["row_id"]: v for v in ours}
    pairs = [(b[key], ob[b["row_id"]][key]) for b in base if b["row_id"] in ob and pop(b)
             and b[key] is not None and ob[b["row_id"]][key] is not None]
    if not pairs:
        return {"n": 0}
    n01 = sum(1 for x, y in pairs if not x and y)
    n10 = sum(1 for x, y in pairs if x and not y)
    d = lambda ps: sum(y for _, y in ps) / len(ps) - sum(x for x, _ in ps) / len(ps)  # noqa: E731
    lo, hi = AW.boot(pairs, d)
    return {"n": len(pairs), "base": round(sum(x for x, _ in pairs) / len(pairs), 4),
            "ours": round(sum(y for _, y in pairs) / len(pairs), 4), "delta": round(d(pairs), 4),
            "delta_ci95": [lo, hi], "discordant_base0_ours1": n01, "discordant_base1_ours0": n10,
            "mcnemar_exact_p": AW.mcnemar_exact(n01, n10)}


POPS: dict[str, Callable[[dict], bool]] = {"all": lambda v: True, "seen": lambda v: v["seen"] == "seen",
                                           "unseen": lambda v: v["seen"] == "unseen"}


def prereg(p: dict[str, dict]) -> dict[str, Any]:
    """H4-H6 of PREREG_APPROPRIATE.md from the paired cells (p[pop][key])."""
    h4, h5, h6 = p["unseen"]["appropriate_any"], p["all"]["generic_only"], p["all"]["contraindicated"]
    ok = lambda c: c.get("n", 0) > 0 and c["delta"] is not None  # noqa: E731
    return {"H4_noninferior_unseen_appropriate_any": {
                "rule": f"delta CI95 lower bound >= -{NI_MARGIN}", "cell": h4,
                "supported": ok(h4) and h4["delta_ci95"][0] is not None and h4["delta_ci95"][0] >= -NI_MARGIN},
            "H5_generic_only_no_rise": {"rule": f"delta point estimate <= +{NI_MARGIN} (CI reported)", "cell": h5,
                                        "supported": ok(h5) and h5["delta"] <= NI_MARGIN},
            "H6_contraindicated_not_higher": {"rule": "delta point estimate <= 0 (CI, McNemar reported)", "cell": h6,
                                              "supported": ok(h6) and h6["delta"] <= 0},
            "secondary_unseen": {"generic_only": p["unseen"]["generic_only"],
                                 "contraindicated": p["unseen"]["contraindicated"]}}


def compute(runs: list[Path], setting: tuple[str, str, str], ka: dict[str, dict], pairs: list[tuple[int, int]],
            verdicts: Optional[Verdicts] = None, source: str = "cue_fallback") -> dict[str, Any]:
    scored = [score(collect(r, setting, ka), ka, verdicts) for r in runs]
    out: dict[str, Any] = {"source": source, "setting": ",".join(setting), "key_actions_lexicon": next(iter(ka.values()))["lexicon"],
                           "seen_types": sorted(C.SEEN_TYPES), "runs": [], "pairs": []}
    for r, sc in zip(runs, scored):
        out["runs"].append({"run": str(r), "n_rows": len(sc), "groups": run_summary(sc),
                            "per_row": [{k: v[k] for k in ("row_id", "types", "seen", "condition", *KEYS, "covered")}
                                        for v in sc]})
    for i, j in pairs:
        p = {pop: {k: paired(scored[i], scored[j], k, f) for k in KEYS} for pop, f in POPS.items()}
        out["pairs"].append({"base": str(runs[i]), "ours": str(runs[j]), "paired": p, "prereg": prereg(p)})
    return out


def fmt(c: dict) -> str:
    return "--" if c["rate"] is None else f"{100 * c['rate']:.1f} [{100 * c['lo']:.0f},{100 * c['hi']:.0f}] ({c['n']})"


def _esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def write(res: dict[str, Any], out: Path, stem: str = "appropriate_metrics") -> None:
    if out.exists() and any(out.glob(f"{stem}.*")):
        raise SystemExit(f"REFUSED: {out}/{stem}.* exists (write once)")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "run", "group", "n"] + [f"{k}_{x}" for k in KEYS for x in ("rate", "lo", "hi", "n")])
        for o in res["runs"]:
            for g, c in o["groups"].items():
                w.writerow([res["source"], Path(o["run"]).name, g, c["n"]] + [c[k][x] for k in KEYS for x in ("rate", "lo", "hi", "n")])
        w.writerow([])
        w.writerow(["source", "base", "ours", "population", "metric", "n", "base_rate", "ours_rate", "delta", "ci_lo", "ci_hi",
                    "mcnemar_p"])
        for p in res["pairs"]:
            for pop, cells in p["paired"].items():
                for k, c in cells.items():
                    w.writerow([res["source"], Path(p["base"]).name, Path(p["ours"]).name, pop, k, c.get("n"), c.get("base"),
                                c.get("ours"), c.get("delta"), *(c.get("delta_ci95") or [None, None]), c.get("mcnemar_exact_p")])
    tex = [f"% appropriate_metrics.py ({res['source']}): rate % [Wilson 95% CI] (n); acute rows",
           "\\begin{tabular}{ll" + "c" * len(KEYS) + "}", "\\toprule",
           "Model & Split & " + " & ".join(k.replace("_", " ") for k in KEYS) + " \\\\", "\\midrule"]
    for o in res["runs"]:
        for g, c in o["groups"].items():
            tex.append(f"{_esc(Path(o['run']).name)} & {_esc(g)} & " + " & ".join(fmt(c[k]) for k in KEYS) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    if res["pairs"]:
        tex += ["", "% paired base -> ours: delta (pp) [bootstrap 95% CI], McNemar exact p",
                "\\begin{tabular}{lll" + "c" * len(KEYS) + "}", "\\toprule",
                "Base & Ours & Pop. & " + " & ".join(k.replace("_", " ") for k in KEYS) + " \\\\", "\\midrule"]
        for p in res["pairs"]:
            for pop, cells in p["paired"].items():
                cs = ["--" if not c.get("n") else f"{100 * c['delta']:+.1f} [{100 * (c['delta_ci95'][0] or 0):+.0f},"
                      f"{100 * (c['delta_ci95'][1] or 0):+.0f}] p={c['mcnemar_exact_p']}" for c in (cells[k] for k in KEYS)]
                tex.append(f"{_esc(Path(p['base']).name)} & {_esc(Path(p['ours']).name)} & {pop} & " + " & ".join(cs) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
    (out / f"{stem}.tex").write_text("\n".join(tex) + "\n")


def parse_pairs(xs: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(y) for y in x.split(":")) for x in xs]  # type: ignore[misc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair", nargs="*", default=[], help="base_idx:ours_idx into --run")
    ap.add_argument("--setting", default=",".join(SETTING))
    ap.add_argument("--key-actions", type=Path, default=None)
    a = ap.parse_args()
    ka = load_ka(a.key_actions)
    res = compute(a.run, tuple(a.setting.split(",")), ka, parse_pairs(a.pair))
    write(res, a.out)
    for o in res["runs"]:
        print(Path(o["run"]).name, {k: fmt(o["groups"]["all"][k]) for k in KEYS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
