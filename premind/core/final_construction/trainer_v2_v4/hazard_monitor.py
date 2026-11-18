"""hazard/* training monitor (v4 training v2): per-step rates of specific / generic-only / none / contraindicated
among the ACUTE rollouts, from the backend that scored them (judge or cue) and from the cue path (always), plus the
judge-vs-cue state agreement, the judge fallback rate and the structure-ordered rate.

Online: ``Counter.add(info)`` per scored acute training episode (patch_h score hook), ``values`` -> all_sum over
ranks, ``rates`` -> the step record key ``hazard``; ``payload`` -> wandb ``hazard/*``.
Offline (CPU): ``python -I -m trainer_v2_v4.hazard_monitor RUN_DIR [RUN_DIR ...] --release REL [--key-actions F]``
classifies with the cue path every acute text found in
  RUN_DIR/rollout_texts.jsonl  (v2 runs with --log-rollout-texts; grouped in --bin-step bins), and
  RUN_DIR/dev_eval/step_*/texts.jsonl  (greedy in-loop dev / monitor-slice evals: a proxy, not rollouts).
``steps.jsonl`` is checked for rollout texts too (the v4 pilots log none: only reward / advantage / action_pred).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

logger = logging.getLogger(__name__)
STATES = ("specific", "generic_only", "none")
FIELDS = ("n_acute", "specific", "generic_only", "none", "contraindicated", "cue_specific", "cue_generic_only",
          "cue_none", "cue_contraindicated", "judged", "judge_fallback", "agree", "ordered", "reward_sum",
          "g1_injection", "g2_cue_clause", "g3_off_type", "g4_empty")


class Counter:
    def __init__(self) -> None:
        self.c = dict.fromkeys(FIELDS, 0.0)

    def add(self, info: Optional[dict], value: Optional[float] = None, guard: Optional[list] = None) -> None:
        if not info or not info.get("acute"):
            return
        c, cue = self.c, info["cue"]
        c["n_acute"] += 1
        c[info["state"]] += 1
        c["contraindicated"] += bool(info["contraindicated"])
        c[f"cue_{cue['state']}"] += 1
        c["cue_contraindicated"] += bool(cue["contraindicated"])
        c["judged"] += info["backend"] == "judge"
        c["judge_fallback"] += info["backend"] == "cue(judge_fallback)"
        c["agree"] += info["backend"] == "judge" and info["state"] == cue["state"]
        c["ordered"] += bool(info.get("ordered"))
        c["reward_sum"] += value or 0.0
        for g in guard or ():                            # v4wh_g guard notes (reward_v4h_guard)
            k = {"G1": "g1_injection", "G2": "g2_cue_clause", "G3": "g3_off_type", "G4": "g4_empty"}.get(g[:2])
            if k:
                c[k] += 1

    def values(self) -> list[float]:
        return [self.c[k] for k in FIELDS]

    @staticmethod
    def rates(vals: list[float]) -> dict:
        c = dict(zip(FIELDS, vals))
        n = c["n_acute"]
        if not n:
            return {"n_acute": 0}
        out = {"n_acute": int(n), **{f"{k}_rate": c[k] / n for k in STATES + ("contraindicated",)},
               **{f"cue_{k}_rate": c[f"cue_{k}"] / n for k in STATES + ("contraindicated",)},
               "judge_fallback_rate": c["judge_fallback"] / n, "ordered_rate": c["ordered"] / n,
               "term_mean": c["reward_sum"] / n,
               **{f"{k}_rate": c[k] / n for k in ("g1_injection", "g2_cue_clause", "g3_off_type", "g4_empty")}}
        if c["judged"]:
            out["judge_cue_agreement"] = c["agree"] / c["judged"]
        return out


VERBOSE_WORDS = 260          # = the v4wh_g judge clip (G5): longer answers are partly hidden from the judge


def verbosity(lens: list, gold_median: Optional[float] = None) -> dict:
    """Per-step verbosity of the acute rollouts: <answer> word-count median / mean / share > VERBOSE_WORDS."""
    out: dict = {} if gold_median is None else {"gold_len_median": gold_median}
    if not lens:
        return out
    xs = sorted(lens)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {**out, "acute_len_median": med, "acute_len_mean": sum(xs) / n,
            "acute_len_share_gt260": sum(x > VERBOSE_WORDS for x in xs) / n}


def gold_len_median(release: Path, mode: str = "gt_retrieved_drop", split: str = "train") -> Optional[float]:
    """Median <answer> word count of the release's acute gold answers (the verbosity reference)."""
    sub = "" if mode == "retrieved" else f"{mode}/"
    f = Path(release) / f"{sub}{split}.jsonl"
    if not f.is_file():
        return None
    lens = [len((r.get("gold_answer") or "").split()) for r in _jl(f) if _acute(r)]
    return verbosity(lens).get("acute_len_median") if lens else None


def payload(rec: dict) -> dict:
    return {f"hazard/{k}": v for k, v in (rec.get("hazard") or {}).items() if v is not None}


# ----------------------------------------------------------------------------- offline
def _jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def release_index(release: Path, mode: str) -> dict[str, dict]:
    sub = "" if mode == "retrieved" else f"{mode}/"
    out = {}
    for sp in ("train", "dev"):
        f = Path(release) / f"{sub}{sp}.jsonl"
        if f.is_file():
            out.update({r["sample_id"]: r for r in _jl(f)})
    return out


def _acute(row: dict) -> Optional[dict]:
    ev = row["evaluator"] if isinstance(row["evaluator"], dict) else json.loads(row["evaluator"])
    ah = ev.get("acute_visual_hazard") or {}
    return ah if (ah.get("image_verdict") or {}).get("severity") == "acute" else None


def classify_texts(texts: list[dict], rows: dict[str, dict], ka: dict) -> Counter:
    from trainer_v2_v4 import key_actions_v4h as KA
    from trainer_v2_v4 import reward_v4h as RH
    cnt = Counter()
    for t in texts:
        row = rows.get(t["row_id"])
        ah = _acute(row) if row else None
        if ah is None:
            continue
        ra = ka.get(t["row_id"]) or KA.type_default(t["row_id"], ah["image_verdict"].get("what") or "")
        stage2 = t.get("stage2") or f"<answer>{t.get('answer') or t.get('hazard_part') or ''}</answer>"
        info = RH.hazard_info({"acute": True, "ra": ra, "judge": None, "backend": "cue"}, stage2, ah.get("gold_pattern"))
        if t.get("hazard") and t["hazard"].get("state"):          # logged by a v2 run: the scored state wins
            info = {**info, "state": t["hazard"]["state"], "backend": t["hazard"].get("backend", "cue"),
                    "contraindicated": t["hazard"].get("contraindicated", info["contraindicated"])}
        cnt.add(info, RH.state_value(info["state"], info["contraindicated"], RH.PRESETS["v4wh"]))
    return cnt


def steps_have_texts(run: Path) -> bool:
    f = run / "steps.jsonl"
    if not f.is_file():
        return False
    with f.open(encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    eps = rec.get("episodes") or [{}]
    return any(k in eps[0] for k in ("stage2", "text", "completion", "answer"))


def offline(run: Path, rows: dict, ka: dict, bin_step: int = 25) -> dict:
    out: dict[str, Any] = {"run": str(run), "steps_jsonl_has_rollout_texts": steps_have_texts(run)}
    rt = run / "rollout_texts.jsonl"
    if rt.is_file():
        recs = _jl(rt)
        bins: dict[int, list] = {}
        for r in recs:
            bins.setdefault(int(r["step"]) // bin_step * bin_step, []).append(r)
        out["rollouts"] = {f"steps_{b}-{b + bin_step - 1}": Counter.rates(classify_texts(v, rows, ka).values())
                           for b, v in sorted(bins.items())}
    dev = {}
    for d in sorted((run / "dev_eval").glob("step_*")):
        if (d / "texts.jsonl").is_file():
            dev[d.name] = Counter.rates(classify_texts(_jl(d / "texts.jsonl"), rows, ka).values())
    out["dev_eval_greedy"] = dev
    return out


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import key_actions_v4h as KA
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--release", required=True)
    ap.add_argument("--mode", default="gt_retrieved_drop")
    ap.add_argument("--key-actions", default=None, help="KEY_ACTIONS jsonl (default: the train_diversify file if it "
                                                         "exists, else the v4h fallback file)")
    ap.add_argument("--bin-step", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    kpath = Path(a.key_actions) if a.key_actions else (KA.DIVERSIFY_FILE if KA.DIVERSIFY_FILE.is_file()
                                                       else KA.FALLBACK_FILE)
    ka = KA.load(kpath, Path(a.release), a.mode) if kpath.is_file() else {}
    rows = release_index(Path(a.release), a.mode)
    res = {"key_actions": str(kpath), "runs": [offline(Path(r), rows, ka, a.bin_step) for r in a.runs]}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
