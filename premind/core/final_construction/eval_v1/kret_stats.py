"""K-retention aggregates: family-cluster bootstrap CIs and paired base-vs-trained deltas (CPU only).

Same resampling algorithm, percentile indices and seed handling as ``eval_v1.analysis`` (``_cell`` / ``paired``).
It is re-stated here because ``analysis.summary`` is hard-wired to ``metrics_v9`` VCM records (withhold, pair and
citation fields that K-retention records do not have); the summary function is the only difference.
"""
from __future__ import annotations

import random
from collections import defaultdict

METRICS = ("memory_follow_rate", "memory_follow_lenient_rate", "action_acc", "pc_hit_rate", "decisive_fact_hit_rate",
           "wrong_record_hit_rate", "contrary_rate", "risk_acc", "format_rate", "pc_reward_rate", "reward_mean")
GROUPS = ("grounding", "risk_label", "expected_action")
SIDE_BY_SIDE = ("action_acc", "pc_hit_rate", "memory_follow_rate", "decisive_fact_hit_rate", "memory_follow_lenient_rate",
                "wrong_record_hit_rate")
GROUP_COLS = ("memory_follow_rate", "memory_follow_lenient_rate", "action_acc", "decisive_fact_hit_rate", "risk_acc")
LABELS = {"memory_follow_rate": "Memory-follow (strict)", "memory_follow_lenient_rate": "Memory-follow (lenient)",
          "decisive_fact_hit_rate": "Decisive-fact hit", "wrong_record_hit_rate": "Wrong-record hit",
          "action_acc": "Action", "pc_hit_rate": "Support hit (strict)",
          "contrary_rate": "Contrary", "risk_acc": "Risk", "format_rate": "Format",
          "pc_reward_rate": "PC reward hit", "reward_mean": "Reward v4"}


def _rate(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return round(sum(map(float, valid)) / len(valid), 4) if valid else None


def summary(recs: list[dict]) -> dict:
    """Point estimates; ``contrary_rate`` is over rows that define contrary patterns (denominator reported)."""
    out = {"n": len(recs), "n_families": len({r["family_id"] for r in recs})}
    for metric, field in (("memory_follow_rate", "memory_follow"), ("action_acc", "action_acc"),
                          ("memory_follow_lenient_rate", "memory_follow_lenient"),
                          ("decisive_fact_hit_rate", "decisive_fact_hit"), ("wrong_record_hit_rate", "wrong_record_hit"),
                          ("pc_hit_rate", "pc_hit"), ("contrary_rate", "contrary_hit"), ("risk_acc", "risk_acc"),
                          ("format_rate", "format_strict"), ("pc_reward_rate", "pc_reward_hit"),
                          ("reward_mean", "reward_total")):
        vals = [r.get(field) for r in recs]
        out[metric] = _rate(vals)
        out[f"n_{metric}"] = sum(v is not None for v in vals)
    return out


def _families(recs: list[dict]) -> list[list[dict]]:
    fam: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        fam[str(r["family_id"])].append(r)
    return list(fam.values())


def _pct(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    return [round(values[int(.025 * (len(values) - 1))], 4), round(values[int(.975 * (len(values) - 1))], 4)]


def cell(recs: list[dict], *, draws: int, seed: int) -> dict:
    """Point estimates plus family-cluster bootstrap 95% CIs for every metric."""
    if draws < 1:
        raise ValueError("draws must be positive")
    groups = _families(recs)
    rng = random.Random(seed)
    sims: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(draws if groups else 0):
        s = summary([r for _ in groups for r in rng.choice(groups)])
        for m in METRICS:
            if s.get(m) is not None:
                sims[m].append(s[m])
    return {"metrics": summary(recs), "ci95": {m: _pct(v) for m, v in sims.items()}}


def table(recs: list[dict], *, draws: int, seed: int) -> dict:
    """``variant|kind`` -> overall cell and per-group cells."""
    settings: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        settings[f"{r['variant']}|{r['kind']}"].append(r)
    result = {}
    for name, rows in sorted(settings.items()):
        block = {"all": cell(rows, draws=draws, seed=seed), "groups": {}}
        for key in GROUPS:
            by: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by[str(r.get(key))].append(r)
            block["groups"][key] = {k: cell(v, draws=draws, seed=seed) for k, v in sorted(by.items())}
        result[name] = block
    return result


def _key(r: dict) -> tuple:
    return (r["row_id"], r["variant"], r["kind"], r.get("sample_idx", 0))


def paired(a: list[dict], b: list[dict], *, metric: str, draws: int, seed: int) -> dict:
    """Delta (b - a) with a paired family-cluster bootstrap CI; keys must match exactly."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    aa, bb = {_key(r): r for r in a}, {_key(r): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError("paired runs must have identical unique row/variant/decoding keys")
    if any(aa[k]["family_id"] != bb[k]["family_id"] for k in aa):
        raise ValueError("paired runs disagree on family id")
    fam: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for k in aa:
        fam[str(aa[k]["family_id"])].append((aa[k], bb[k]))
    groups = list(fam.values())
    rng = random.Random(seed)
    vals = []
    for sampled in [groups] + [[rng.choice(groups) for _ in groups] for _ in range(draws if groups else 0)]:
        pairs = [p for f in sampled for p in f]
        va, vb = summary([x for x, _ in pairs]).get(metric), summary([y for _, y in pairs]).get(metric)
        if va is not None and vb is not None:
            vals.append(round(vb - va, 4))
    ci = _pct(vals[1:]) if len(vals) > 1 else None
    return {"metric": metric, "delta_b_minus_a": vals[0] if vals else None, "ci95": ci,
            "n_items": len(aa), "n_families": len(groups)}


def _fmt(c: dict, metric: str) -> str:
    v, ci = c["metrics"].get(metric), c["ci95"].get(metric)
    return "—" if v is None else f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v:.3f}"


def markdown(report: dict) -> str:
    """KRET_REPORT.md body."""
    meta = report["meta"]
    data = meta.get("data", {})
    lines = ["# K-retention (text-memory) evaluation", "",
             "Evaluation only: shelved text-track rows (text-cued G1 / G1+G2), passive exact-five rendering. "
             "Rates are estimate [family-cluster bootstrap 95% CI]; greedy decoding.", "",
             f"- split: `{data.get('split')}`; rows kept {data.get('n_rows_kept')} of {data.get('n_rows')}; "
             f"excluded {data.get('excluded')}",
             f"- model: `{meta.get('model')}`; LoRA: `{meta.get('lora')}`",
             f"- scoring override: {data.get('scoring_override')}", "",
             "Strict memory-follow = action correct AND a support pattern matches AND no contrary pattern matches.",
             "Lenient memory-follow = action correct AND decisive-fact hit (query-novel keyphrase/fact token overlap "
             f">= {meta.get('lenient_threshold', 'THRESHOLD')}, kret_lenient.py). Wrong-record hit = the same test "
             "fires on a non-decisive shown record (false-positive control).", ""]
    for title, metrics in (("Strict vs lenient content", SIDE_BY_SIDE), ("All metrics", METRICS)):
        lines += [f"## {title}", "", "| Variant | Decoding | n | " + " | ".join(LABELS[m] for m in metrics) + " |",
                  "|---|---|---:|" + "---:|" * len(metrics)]
        for name, block in report["table"].items():
            variant, kind = name.split("|")
            c = block["all"]
            lines.append(f"| {variant} | {kind} | {c['metrics']['n']} | " + " | ".join(_fmt(c, m) for m in metrics)
                         + " |")
        lines.append("")
    for key in GROUPS:
        lines += ["", f"## By {key}", "", "| Setting | Value | n | " + " | ".join(LABELS[m] for m in GROUP_COLS) + " |",
                  "|---|---|---:|" + "---:|" * len(GROUP_COLS)]
        for name, block in report["table"].items():
            for label, c in block["groups"][key].items():
                lines.append(f"| {name} | {label} | {c['metrics']['n']} | " + " | ".join(
                    _fmt(c, m) for m in GROUP_COLS) + " |")
    return "\n".join(lines) + "\n"


def paired_markdown(result: dict) -> str:
    lines = ["# K-retention paired comparison", "", "Delta is run B minus run A (e.g. trained minus base); "
             "CI uses paired family-cluster resampling.", "",
             "| Setting | Metric | Delta | 95% CI | Matched items | Families |", "|---|---|---:|---:|---:|---:|"]
    for setting, by_metric in result.items():
        for metric, c in by_metric.items():
            lines.append(f"| {setting} | {metric} | {c['delta_b_minus_a']} | {c['ci95']} | {c['n_items']} | "
                         f"{c['n_families']} |")
    return "\n".join(lines) + "\n"
