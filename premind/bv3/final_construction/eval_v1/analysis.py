"""VCM aggregate tables and family-cluster uncertainty intervals (CPU only)."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from trainer_v2 import metrics_v9 as MV

GROUPS = ("row_type", "grounding", "risk", "condition", "query_style", "request_form", "mode", "knowledge_stratum")
METRICS = ("reward_mean", "action_acc", "risk_acc", "cite_f1", "wrong_concept_rate", "correct_withhold",
           "false_withhold", "withhold_precision", "withhold_recall", "pair_accuracy", "pair_accuracy_strict",
           "hazard_first_rate", "clarify_share", "abstain_share")


def _extra(recs: list[dict]) -> dict:
    absent = [r for r in recs if r["withhold_target"]]
    acute = [r for r in absent if r.get("acute_visual")]
    def rate(values: list[bool | None]) -> float | None:
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else None
    return {"hazard_first_rate": rate([r.get("hazard_first") for r in acute]),
            "n_hazard_first_rate": len(acute),
            "clarify_share": rate([r.get("action_pred") == "clarify" for r in absent]),
            "n_clarify_share": len(absent),
            "abstain_share": rate([r.get("action_pred") == "abstain" for r in absent]),
            "n_abstain_share": len(absent)}


def summary(recs: list[dict]) -> dict:
    return {**MV.summarize(recs), **_extra(recs)}


def bootstrap(recs: list[dict], metric: str, *, draws: int = 1000, seed: int = 0) -> list[float] | None:
    if draws < 1:
        raise ValueError("draws must be positive")
    families: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        families[str(r["family_id"])].append(r)
    if not families:
        return None
    groups = list(families.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        sample = [r for _ in groups for r in rng.choice(groups)]
        v = summary(sample).get(metric)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(.025 * (len(vals) - 1))], 4), round(vals[int(.975 * (len(vals) - 1))], 4)]


def table(recs: list[dict], *, draws: int = 1000, seed: int = 0) -> dict:
    if draws < 1:
        raise ValueError("draws must be positive")
    settings: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        settings[f"{r['arm']}|{r['mode']}|{r['kind']}"].append(r)
    result = {}
    for name, rows in sorted(settings.items()):
        block = {"all": _cell(rows, draws, seed), "groups": {}}
        for key in GROUPS:
            by: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by[str(r.get(key) if r.get(key) is not None else "none")].append(r)
            block["groups"][key] = {k: _cell(v, draws, seed) for k, v in sorted(by.items())}
        result[name] = block
    return result


def _cell(recs: list[dict], draws: int, seed: int) -> dict:
    stats = summary(recs)
    families: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        families[str(r["family_id"])].append(r)
    groups = list(families.values())
    rng = random.Random(seed)
    sims: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(draws):
        sampled = [r for _ in groups for r in rng.choice(groups)]
        s = summary(sampled)
        for m in METRICS:
            if s.get(m) is not None:
                sims[m].append(s[m])
    ci = {}
    for m, values in sims.items():
        values.sort()
        ci[m] = ([round(values[int(.025 * (len(values) - 1))], 4),
                  round(values[int(.975 * (len(values) - 1))], 4)] if values else None)
    return {"metrics": stats, "ci95": ci}


def paired(a: list[dict], b: list[dict], *, metric: str = "reward_mean", draws: int = 1000,
           seed: int = 0) -> dict:
    def key(r: dict) -> tuple:
        return (r["row_id"], r["arm"], r["mode"], r["kind"], r.get("sample_idx", 0))
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError("paired runs must have identical unique item/arm/mode/decoding keys")
    if any(aa[k]["family_id"] != bb[k]["family_id"] for k in aa):
        raise ValueError("paired runs disagree on family id")
    families: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for k in aa:
        families[str(aa[k]["family_id"])].append((aa[k], bb[k]))
    groups = list(families.values())
    rng = random.Random(seed)
    vals = []
    for sampled in [groups] + [[rng.choice(groups) for _ in groups] for _ in range(draws)]:
        pairs = [pair for fam in sampled for pair in fam]
        va, vb = summary([x for x, _ in pairs]).get(metric), summary([y for _, y in pairs]).get(metric)
        if va is not None and vb is not None:
            vals.append(round(vb - va, 4))
    ci = None
    if len(vals) > 1:
        sims = sorted(vals[1:])
        ci = [sims[int(.025 * (len(sims) - 1))], sims[int(.975 * (len(sims) - 1))]]
    return {"metric": metric, "delta_b_minus_a": vals[0] if vals else None, "ci95": ci,
            "n_items": len(aa), "n_families": len(groups)}


def markdown(report: dict) -> str:
    lines = ["# VCM Evaluation Report", "", "Rates are shown as estimate [family-cluster bootstrap 95% CI];",
             "denominators are in `eval_report.json`. Greedy decoding is the primary analysis.", ""]
    def fmt(cell: dict, metric: str) -> str:
        v, ci = cell["metrics"].get(metric), cell["ci95"].get(metric)
        return "—" if v is None else f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v:.3f}"
    lines += ["## Main table: arm × render mode", "", "| Arm | Mode | Decoding | n | Action | Risk | Cite F1 | Reward |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, block in report["table"].items():
        arm, mode, kind = name.split("|")
        c = block["all"]
        lines.append(f"| {arm} | {mode} | {kind} | {c['metrics']['n']} | {fmt(c,'action_acc')} | "
                     f"{fmt(c,'risk_acc')} | {fmt(c,'cite_f1')} | {fmt(c,'reward_mean')} |")
    for title, metrics, group in (("Safety table", ("risk_acc", "hazard_first_rate", "wrong_concept_rate"), "risk"),
                                  ("Withhold table", ("correct_withhold", "false_withhold",
                                                      "withhold_precision", "withhold_recall", "pair_accuracy",
                                                      "clarify_share", "abstain_share"), "condition"),
                                  ("Knowledge strata", ("action_acc", "pair_accuracy"), "knowledge_stratum")):
        lines += ["", f"## {title}", "", "| Setting | Group | n | " + " | ".join(metrics) + " |",
                  "|---|---|---:|" + "---:|" * len(metrics)]
        for name, block in report["table"].items():
            for label, cell in block["groups"][group].items():
                lines.append(f"| {name} | {label} | {cell['metrics']['n']} | " +
                             " | ".join(fmt(cell, m) for m in metrics) + " |")
    lines += ["", "## Breakdowns", ""]
    for key in GROUPS:
        lines += [f"### {key}", "", "| Setting | Value | n | Action | Withhold | Pair accuracy |",
                  "|---|---|---:|---:|---:|---:|"]
        for name, block in report["table"].items():
            for label, cell in block["groups"][key].items():
                lines.append(f"| {name} | {label} | {cell['metrics']['n']} | {fmt(cell,'action_acc')} | "
                             f"{fmt(cell,'correct_withhold')} | {fmt(cell,'pair_accuracy')} |")
        lines.append("")
    return "\n".join(lines) + "\n"
