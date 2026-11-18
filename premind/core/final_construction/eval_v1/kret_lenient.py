"""Lenient secondary content metric for K-retention: ``decisive_fact_hit`` (CPU only).

The strict ``support_patterns`` are long gold-wording regexes, so they miss most paraphrases. The lenient test
normalizes text (lowercase, punctuation stripped, stopwords removed, light suffix stemming). For each decisive
phrase in ``decisive_fact.keyphrases + [decisive_fact.fact_value_or_polarity]`` it keeps only the tokens that are
NOT in the user query ("novel" memory content, so a reply that only echoes the text-cued query cannot hit). A reply
hits when, for some phrase, ``|novel ∩ answer| / |novel| >= THRESHOLD``. The test reads only the parsed
``<answer>``, never the reasoning trace that quotes the context. A row whose decisive phrases are entirely contained
in the query has no novel tokens; it is n/a (None).

THRESHOLD is frozen from the open TRAIN gold answers. The rule is the largest grid value whose gold hit rate is
>= 0.9 (see ``calibrate``). Control: the same test applied to the facts of the four non-decisive shown records in
the same row (the wrong-record false-positive rate), plus a query-echo control (0 by construction).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from eval_v1 import kret_data as KD  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402

logger = logging.getLogger(__name__)

LENIENT_VERSION = "kret_lenient_v2"  # v2: plural stemming fix (bottles -> bottle)
GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
GOLD_TARGET = 0.9
THRESHOLD = 0.5  # frozen 2026-10-03 by calibrate("train"): train gold 0.923, wrong-record control 0.016 (v2)
STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that "
    "these those your you yours their they them he she his her we our i my me not no do does did can could should "
    "would will just so than then there here into out up down over about before after while also any all some more "
    "most very too only own same such what which who whom when where why how prefers prefer usual usually user "
    "target needs need".split())
_WORD = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    if len(w) <= 4:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> frozenset[str]:
    """Normalized content tokens."""
    return frozenset(_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1)


def overlap(phrases: list[str], answer: str, query: str) -> float | None:
    """Max over phrases of the novel-token recall in ``answer``; None if no phrase has novel tokens."""
    ans, q = tokens(answer), tokens(query)
    vals = []
    for p in phrases:
        novel = tokens(p) - q
        if novel:
            vals.append(len(novel & ans) / len(novel))
    return max(vals) if vals else None


def _hit(score: float | None, threshold: float) -> bool | None:
    return None if score is None else score >= threshold


def decisive_phrases(view: dict) -> list[str]:
    fact = view["decisive_fact"]
    return [*fact["keyphrases"], fact["fact_value_or_polarity"]]


def wrong_record_phrases(view: dict) -> list[list[str]]:
    """Fact phrases of every shown non-decisive record (control)."""
    by_id = {r["record_id"]: r for r in view["records"]}
    return [[by_id[rid]["keyphrase"], by_id[rid]["fact_value_or_polarity"]]
            for rid in view["selected_record_ids"] if rid != view["decisive_record_id"]]


def fields(view: dict, answer: str, action_acc: bool, *, threshold: float = THRESHOLD) -> dict:
    """Lenient per-reply fields added to a K-retention record."""
    score = overlap(decisive_phrases(view), answer, view["query"])
    hit = _hit(score, threshold)
    wrong = [_hit(overlap(ph, answer, view["query"]), threshold) for ph in wrong_record_phrases(view)]
    return {"decisive_fact_overlap": None if score is None else round(score, 4), "decisive_fact_hit": hit,
            "memory_follow_lenient": None if hit is None else bool(action_acc and hit),
            "n_wrong_record_hits": sum(bool(w) for w in wrong), "wrong_record_hit": any(bool(w) for w in wrong)}


def calibrate(split: str) -> dict:
    """Gold / wrong-record / query-echo hit rates per grid threshold on an OPEN split (never holdout)."""
    rows, prov = KD.load_rows(split)
    rows = [r for r in rows if KD.exclusion_reason(r) is None]
    gold, ctrl, echo = [], [], []
    for r in rows:
        gold.append(overlap(decisive_phrases(r), r["gold_answer_draft"], r["query"]))
        echo.append(overlap(decisive_phrases(r), r["query"], r["query"]))
        ctrl += [overlap(ph, r["gold_answer_draft"], r["query"]) for ph in wrong_record_phrases(r)]
    def rate(vals: list, th: float) -> float:
        return round(sum((v or 0.0) >= th for v in vals) / len(vals), 4)  # n/a rows count as misses here
    grid = {str(th): {"gold_hit": rate(gold, th), "wrong_record_hit": rate(ctrl, th), "query_echo_hit": rate(echo, th)}
            for th in GRID}
    ok = [th for th in GRID if grid[str(th)]["gold_hit"] >= GOLD_TARGET]
    return {"version": LENIENT_VERSION, "split": split, "source_sha256": prov["sha256"], "n_rows": len(rows),
            "n_rows_no_novel_tokens": sum(v is None for v in gold), "n_wrong_records": len(ctrl), "grid": grid,
            "rule": f"largest grid threshold with gold_hit >= {GOLD_TARGET}",
            "selected": max(ok) if ok else None, "frozen_threshold": THRESHOLD}


def rescore(run: Path, out: Path) -> dict:
    """Re-score an existing run's texts.jsonl (no regeneration) into a new write-once dir."""
    from eval_v1 import kret_eval as KE
    P.refuse_holdout_path(run)
    rep = json.loads((run / "kret_report.json").read_text())
    meta, data = rep["meta"], rep["meta"]["data"]
    if data["split"] == "holdout":
        raise P.HoldoutRefusedError("refused: rescoring a holdout run is a final-step action")
    items, man = KD.load_kret(data["split"], variants=tuple(data["variants"]), limit=data.get("limit"))
    if man["sha256"] != data["sha256"]:
        raise ValueError("source rows changed since the run (sha256 mismatch)")
    by_key = {(it.row_id, it.variant): it for it in items}
    texts = [json.loads(x) for x in (run / "texts.jsonl").read_text().splitlines() if x.strip()]
    if {(t["row_id"], t["variant"]) for t in texts} != set(by_key) or len(texts) != len(by_key):
        raise ValueError("texts.jsonl does not match the run's items")
    recs = [KE.score(by_key[(t["row_id"], t["variant"])], t["completion"], kind=t["kind"],
                     sample_idx=t["sample_idx"]) for t in texts]
    new_meta = {**meta, "rescored_from": str(run), "lenient_version": LENIENT_VERSION,
                "lenient_threshold": THRESHOLD, "data": man}
    return KE.write(out, recs, texts, new_meta, draws=1000, seed=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="K-retention lenient metric: calibrate / rescore (CPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--splits", default="train,dev")
    c.add_argument("--out", required=True)
    r = sub.add_parser("rescore")
    r.add_argument("--run", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out = Path(args.out)
    P.refuse_holdout_path(out)
    if args.cmd == "rescore":
        rescore(Path(args.run), out)
        return 0
    if out.exists():
        raise FileExistsError(out)
    result = {s: calibrate(s) for s in args.splits.split(",")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps({s: {"grid": v["grid"], "selected": v["selected"]} for s, v in result.items()},
                                indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
