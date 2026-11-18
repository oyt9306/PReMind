"""Per-row hazard-specific KEY ACTIONS from the gold answers of the acute rows (deterministic; CPU; no model, no API).

1. hazard part of the gold = its leading sentences that address the hazard: up to three leading sentences without any
   lexicon action (scene description, "Yes."/"No.") are skipped as preamble; then sentences are taken while each
   matches at least one action of the row's types (+ generic actions) and does not open the personal part ("Then",
   "After that", "Once help ...", a question, "Your memories ...", ...).
2. key actions = lexicon actions whose cues match the hazard part: type-specific ones first (<= 3, in gold order), then
   generic ones fill up to 3 actions in total.
3. flags: no_specific (hazard part has no type-specific action: data issue), no_action (nothing matched), unmapped.

  python -I key_actions.py [--release RELEASE_TEST_v4d] [--out out/key_actions_v4d]   (write-once dir)
Outputs: KEY_ACTIONS_v4d.jsonl (row_id, types, seen, visible_hazard, hazard_part, actions[{label, specific, desc,
cues}], flags), KEY_ACTIONS_SUMMARY.json, KEY_ACTIONS_SUMMARY.md (per-type table + flagged golds).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ha_common as C  # noqa: E402
import lexicon as L  # noqa: E402

MAX_ACTIONS = 3
MAX_PREAMBLE = 3
_PERSONAL_START = re.compile(
    r"^(?:then\b|after (?:that|this|it)\b|afterwards\b|once (?:help|everyone|it is|it's|safe|clear|the (?:police|hazard|area)|"
    r"first aid|emergency|they are|you)|before (?:any|deciding|handing)|your memories|i (?:can't|cannot|don't|need)|"
    r"(?:who|what|where|which|whom|how)\b|(?:does|do|is|are|has|can|should)\s)", re.I)


def matches(text: str, actions: tuple[L.Action, ...]) -> list[tuple[int, L.Action]]:
    """(first match position, action) for every action with a cue hit in ``text``."""
    out = []
    for a in actions:
        pos = [m.start() for c in a.cues for m in [re.search(c, text, re.I)] if m]
        if pos:
            out.append((min(pos), a))
    return sorted(out, key=lambda x: x[0])


def split_hazard(gold: str, types: list[str]) -> tuple[str, str]:
    """(preamble, action sentences) of the gold's hazard part; key actions are matched on the action sentences only."""
    acts = L.actions_for(types)
    sents = C.sentences(gold)
    i, kept = 0, []
    while i < len(sents) and i < MAX_PREAMBLE and not matches(sents[i], acts):
        i += 1
    if i == len(sents) or not matches(sents[i], acts):   # no action in the leading sentences
        return " ".join(sents[:1]), ""
    for j, s in enumerate(sents[i:]):
        if (j > 0 and _PERSONAL_START.search(s)) or s.endswith("?") or not matches(s, acts):
            break
        kept.append(s)
    return " ".join(sents[:i]), " ".join(kept)


def hazard_part(gold: str, types: list[str]) -> str:
    return " ".join(x for x in split_hazard(gold, types) if x)


def extract(row: dict) -> dict[str, Any]:
    flags: list[str] = []
    try:
        types = C.acute_types(row) or []
    except KeyError:
        types, flags = [], ["unmapped"]
    pre, act = split_hazard(row["gold_answer"], types)
    part = " ".join(x for x in (pre, act) if x)
    hits = matches(act, L.actions_for(types))
    spec = [a for _, a in hits if a.specific][:MAX_ACTIONS]
    gen = [a for _, a in hits if not a.specific][:max(0, MAX_ACTIONS - len(spec))]
    acts = spec + gen
    if not acts:
        flags.append("no_action")
    elif not spec:
        flags.append("no_specific")
    return {"row_id": row["sample_id"], "types": types, "seen": C.seen_split(types) if types else None,
            "visible_hazard": C.visible_hazard(row), "hazard_part": part,
            "actions": [{"label": a.label, "specific": a.specific, "desc": a.desc, "cues": list(a.cues)} for a in acts],
            "flags": flags, "lexicon": L.LEXICON_VERSION}


def extract_release(release: Path = C.RELEASE_V4D, split: str = "holdout") -> list[dict]:
    return [extract(r) for r in C.release_rows(release, split) if C.HZ.is_acute(r)]


def load_key_actions(path: Path) -> dict[str, dict]:
    return {r["row_id"]: r for r in C.jl(path)}


def summary(ka: list[dict]) -> dict[str, Any]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in ka:
        for t in r["types"] or ["unmapped"]:
            by[t].append(r)
    per = {}
    for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        lab = Counter(a["label"] for r in rs for a in r["actions"])
        per[t] = {"n_rows": len(rs), "n_with_specific": sum(any(a["specific"] for a in r["actions"]) for r in rs),
                  "mean_actions": round(sum(len(r["actions"]) for r in rs) / len(rs), 2),
                  "mean_specific": round(sum(sum(a["specific"] for a in r["actions"]) for r in rs) / len(rs), 2),
                  "labels": dict(lab.most_common())}
    flagged = [{"row_id": r["row_id"], "types": r["types"], "flags": r["flags"], "hazard_part": r["hazard_part"]}
               for r in ka if r["flags"]]
    return {"lexicon": L.LEXICON_VERSION, "n_acute": len(ka),
            "n_with_specific": sum(any(a["specific"] for a in r["actions"]) for r in ka),
            "per_type": per, "flagged": flagged}


def summary_md(s: dict[str, Any]) -> str:
    lines = [f"# Key actions per hazard type ({s['lexicon']}; {s['n_acute']} acute rows, "
             f"{s['n_with_specific']} with >= 1 type-specific action)", "",
             "| type | rows | with specific | mean actions | mean specific | labels (count) |", "|---|---:|---:|---:|---:|---|"]
    for t, c in s["per_type"].items():
        labs = ", ".join(f"{k} ({v})" for k, v in c["labels"].items())
        lines.append(f"| {t} | {c['n_rows']} | {c['n_with_specific']} | {c['mean_actions']} | {c['mean_specific']} | {labs} |")
    lines += ["", f"## Flagged golds ({len(s['flagged'])})", ""]
    lines += [f"- `{f['row_id']}` {f['types']} {f['flags']}: {f['hazard_part']}" for f in s["flagged"]] or ["none"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=C.RELEASE_V4D)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "key_actions_v4d")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"REFUSED: {a.out} not empty (write once)")
    a.out.mkdir(parents=True, exist_ok=True)
    ka = extract_release(a.release, a.split)
    (a.out / "KEY_ACTIONS_v4d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ka))
    s = summary(ka)
    (a.out / "KEY_ACTIONS_SUMMARY.json").write_text(json.dumps(s, indent=1) + "\n")
    (a.out / "KEY_ACTIONS_SUMMARY.md").write_text(summary_md(s))
    print(summary_md(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
