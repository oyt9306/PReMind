"""VCM evaluation metrics (seed of the E0 evaluator) and the over-abstain monitor (WP-T5).

Input = per-episode records built by ``episode_record`` from a reward_v5 ledger (one shared parse and one shared
withhold definition, withhold_v9). Every rate is reported with its denominator ``n_<metric>`` (rows where defined):
  action_acc      action in the branch's gold_action_set;
  risk_acc        rows whose risk term is NOT masked (GT-absent rows without a visible acute hazard are masked and
                  excluded, so on condition=gt_absent it covers the acute rows only);
  cite_f1 / cite_exact   IDENTIFICATION: memories shown (passive / called) AND a non-empty gold set;
  spurious_cite_rate     memories shown AND an empty gold set (G0 / general-rule / GT-absent): any citation;
                  (set_f1 with an empty gold is 1 for no citation and 0 otherwise; it is not averaged into cite_f1);
  distractor_rate any cited index outside the gold set (memories shown);
  wrong_concept_rate (confusers shown); correct_withhold (GT-absent, SPEC (a)-(d)); false_withhold (non-withhold
  targets); withhold precision / recall; g0_call_rate (memory call, rows needing no memory); call precision / recall;
  pair_accuracy   evidence pairs, matched by (arm, mode, decoding, sample index, evidence_pair_id): present member
                  correct use (action + exact citation + no wrong-concept) AND absent member correct withhold;
                  pair_accuracy_strict also requires full personal_content (support-pattern) on the present member.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Optional

from premind.rewards import reward_v5 as R5

GROUP_KEYS = ("row_type", "condition")


def episode_record(item: Any, ledger: dict, *, mode: str = "retrieved", kind: str = "greedy",
                   decision: Optional[str] = None, sample_idx: int = 0) -> dict:
    """Flat record of one scored episode (no texts)."""
    p, tgt, comps = ledger.get("parse") or {}, ledger["target"], ledger["components"]
    cited, gold = set(ledger.get("cited") or []), set(tgt["gold"])
    pc = comps["personal_content"]
    return {"row_id": item.row_id, "arm": item.arm, "mode": mode, "kind": kind, "sample_idx": sample_idx,
            "n_gold": len(gold), "n_cited": len(cited),
            "row_type": item.row.get("row_type"), "condition": item.row.get("condition"),
            "n_targets": item.row.get("n_targets"), "evidence_pair_id": item.row.get("evidence_pair_id"),
            "memory_required": ledger["memory_required"], "withhold_target": bool(tgt["withhold"]),
            "branch": ledger["branch"], "decision": decision if decision is not None else ledger.get("decision"),
            "degenerate": comps["degenerate"]["applicable"], "format_ok": bool(comps["format"]["value"]),
            "action_ok": p.get("action_pred") in tgt["gold_action_set"],
            "risk_ok": (p.get("risk_pred") == tgt["risk_label"]) if comps["risk"]["applicable"] else None,
            "shown": ledger["branch"] != "not_called", "cite_f1": R5.set_f1(cited, gold),
            "cite_exact": cited == gold, "distractor": bool(cited - gold),
            "wrong_concept": p.get("wrong_concept_fired") if comps["wrong_concept"]["applicable"] else None,
            "personal_full": (pc["value"] == R5.W["personal_content"]) if pc["applicable"] else None,
            "withholds": bool(p.get("withholds")), "correct_withhold": p.get("correct_withhold"),
            "total": ledger["total"]}


def _rate(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(map(float, xs)) / len(xs), 4) if xs else None


def _n(xs: list) -> int:
    return sum(x is not None for x in xs)


def withhold_counts(recs: Iterable[dict]) -> dict[str, int]:
    c = {"tp": 0, "fp": 0, "fn": 0, "neg": 0, "pos": 0}
    for r in recs:
        if r["withhold_target"]:
            c["pos"] += 1
            c["tp" if r["withholds"] else "fn"] += 1
        else:
            c["neg"] += 1
            c["fp"] += int(r["withholds"])
    return c


def _div(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def correct_use(r: dict, strict: bool = False) -> bool:
    ok = not r["degenerate"] and r["action_ok"] and r["cite_exact"] and not r["wrong_concept"]
    return ok and (r["personal_full"] is not False if strict else True)


def pair_accuracy(recs: list[dict], strict: bool = False) -> tuple[Optional[float], int]:
    """Evidence pairs matched by (arm, mode, kind, sample_idx, evidence_pair_id); rows without a partner are ignored."""
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in recs:
        if r["evidence_pair_id"]:
            side = "absent" if r["withhold_target"] else "present"
            key = (r["arm"], r["mode"], r["kind"], r.get("sample_idx", 0), r["evidence_pair_id"])
            pairs[key].setdefault(side, []).append(r)
    ok = [all(correct_use(x, strict) for x in p["present"]) and all(x["correct_withhold"] for x in p["absent"])
          for p in pairs.values() if "present" in p and "absent" in p]
    return _rate(ok), len(ok)


def summarize(recs: list[dict]) -> dict:
    """All metrics over one set of records; each rate ``m`` comes with ``n_m`` (its denominator)."""
    w = withhold_counts(recs)
    call = [r for r in recs if r["arm"] == "memory_call"]
    called = [r for r in call if r["decision"] == "call_pdb"]
    need = [r for r in call if r["memory_required"]]
    shown = [r for r in recs if r["shown"]]
    ident = [r for r in shown if r.get("n_gold", 1)]
    empty = [r for r in shown if not r.get("n_gold", 1)]
    cols = {"reward_mean": [r["total"] for r in recs], "format_rate": [r["format_ok"] for r in recs],
            "degenerate_rate": [r["degenerate"] for r in recs], "action_acc": [r["action_ok"] for r in recs],
            "risk_acc": [r["risk_ok"] for r in recs], "cite_f1": [r["cite_f1"] for r in ident],
            "cite_exact": [r["cite_exact"] for r in ident],
            "spurious_cite_rate": [r.get("n_cited", 0) > 0 for r in empty],
            "distractor_rate": [r["distractor"] for r in shown],
            "wrong_concept_rate": [r["wrong_concept"] for r in recs],
            "correct_withhold": [r["correct_withhold"] for r in recs if r["withhold_target"]],
            "false_withhold": [r["withholds"] for r in recs if not r["withhold_target"]],
            "withhold_precision": [r["withhold_target"] for r in recs if r["withholds"]],
            "withhold_recall": [r["withholds"] for r in recs if r["withhold_target"]],
            "g0_call_rate": [r["decision"] == "call_pdb" for r in call if not r["memory_required"]],
            "call_precision": [r["memory_required"] for r in called],
            "call_recall": [r["decision"] == "call_pdb" for r in need]}
    out: dict[str, Any] = {"n": len(recs)}
    for k, xs in cols.items():
        out[k], out[f"n_{k}"] = _rate(xs), _n(xs)
    for name, strict in (("pair_accuracy", False), ("pair_accuracy_strict", True)):
        out[name], out[f"n_{name}"] = pair_accuracy(recs, strict)
    assert w["tp"] + w["fp"] == out["n_withhold_precision"]
    return out


def report(recs: list[dict], keys: tuple[str, ...] = GROUP_KEYS) -> dict:
    """Overall + per (arm, mode, kind) + per group key within each (arm, mode, kind)."""
    out: dict[str, Any] = {"overall": summarize(recs), "by_setting": {}}
    settings: dict[str, list] = defaultdict(list)
    for r in recs:
        settings[f"{r['arm']}|{r['mode']}|{r['kind']}"].append(r)
    for s, rs in sorted(settings.items()):
        block: dict[str, Any] = {"all": summarize(rs)}
        for k in keys:
            groups: dict[str, list] = defaultdict(list)
            for r in rs:
                groups[str(r[k])].append(r)
            block[k] = {g: summarize(v) for g, v in sorted(groups.items())}
        out["by_setting"][s] = block
    return out


MD_COLS = ("action_acc", "risk_acc", "cite_f1", "spurious_cite_rate", "wrong_concept_rate", "correct_withhold",
           "false_withhold", "withhold_precision", "g0_call_rate", "call_precision", "call_recall", "pair_accuracy",
           "pair_accuracy_strict")


def cell(m: dict, c: str) -> str:
    return "-" if m.get(c) is None else f"{m[c]} ({m.get('n_' + c, '')})"


def markdown(rep: dict, title: str) -> str:
    lines = [f"# {title}", "", "Cells: rate (n = rows where the metric is defined). See metrics_v9 docstring.", "",
             "| setting | group | n | " + " | ".join(MD_COLS) + " |", "|" + "---|" * (len(MD_COLS) + 3)]
    for s, block in rep["by_setting"].items():
        rows = [("all", block["all"])] + [(f"{k}={g}", m) for k in GROUP_KEYS for g, m in block[k].items()]
        for name, m in rows:
            lines.append(f"| {s} | {name} | {m['n']} | " + " | ".join(cell(m, c) for c in MD_COLS) + " |")
    return "\n".join(lines) + "\n"


class OverAbstainMonitor:
    """WP-T5: per-step withhold precision / recall, false-withhold on non-GT-absent rows, G0 call rate; hard stop when
    the sliding-window withhold precision drops below ``min_precision`` (needs ``min_withholds`` predicted withholds
    in the window, so a single early step cannot trip it)."""

    def __init__(self, window: int = 5, min_precision: float = 0.7, min_withholds: int = 8,
                 hard_stop: bool = False) -> None:
        self.window, self.min_precision, self.min_withholds, self.hard_stop = window, min_precision, min_withholds, hard_stop
        self.hist: deque = deque(maxlen=window)
        self.stopped = False

    def update(self, recs: list[dict]) -> dict:
        self.hist.append(withhold_counts(recs))
        agg = {k: sum(h[k] for h in self.hist) for k in ("tp", "fp", "fn", "neg", "pos")}
        step = summarize(recs)
        prec_w = _div(agg["tp"], agg["tp"] + agg["fp"])
        trip = (agg["tp"] + agg["fp"] >= self.min_withholds and prec_w is not None and prec_w < self.min_precision)
        self.stopped = self.stopped or (trip and self.hard_stop)
        return {"withhold_precision": step["withhold_precision"], "withhold_recall": step["withhold_recall"],
                "false_withhold": step["false_withhold"], "g0_call_rate": step["g0_call_rate"],
                "correct_withhold": step["correct_withhold"], "window_withhold_precision": prec_w,
                "window_n_withholds": agg["tp"] + agg["fp"], "below_threshold": trip, "hard_stop": self.stopped}
