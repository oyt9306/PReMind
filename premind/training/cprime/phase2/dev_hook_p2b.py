"""Phase II-triple-prime copy: + retention check (owner 2026-10-23): after each dev_eval the 90-row subset's
metrics_v9 overall correct_withhold (x100) is compared with REF_CORRECT_WITHHOLD (51.4, PReMind-4B dev) and written to
OUT/dev_eval_orig90/step_N/retention.json; a "RETENTION WARN" log line when it is more than WARN_DROP points lower.

Phase II' in-loop dev hook (owner amendment 2026-10-23). eval_v1.main_run._dev_eval is replaced in-process by a
copy of its own body (frozen file untouched; ddp_main calls ``MR._dev_eval`` at call time) that:
  1. generates and writes the FULL dev split (109 = 90 original hca1 dev rows + 19 conflict-dev rows) to
     OUT/dev_eval/step_N (unchanged writer -> eval_report.json etc.);
  2. writes the 90-row original subset to OUT/dev_eval_orig90/step_N (same writer, records / texts filtered), the
     like-for-like guardrail report vs the reference's 90-row dev_eval;
  3. returns the wandb / stop-rule payload computed on the 90 ORIGINAL rows only, so the confirmed stop rule and the
     val/* curves see the same composition as the original Phase II run;
  4. judges the 19 conflict-dev greedy memory_call answers (<answer> text only) with mie_v1.MieLunaBackend
     (own ledger / cache / USD cap DEV_MIE_USD_CAP) against INVENTORY_conflict_train_v1_cdev and writes
     OUT/dev_eval/step_N/mie.json: label counts, applied / misapplied rates, the training-reward mean, and a CUE-BASED
     proxy CR = MIE applied AND the record's hazard_mention_strict (the acute-adequacy judge is NOT run in-loop; the
     proper CR needs the RUNBOOK's acute adequacy + severity judges). A judge failure never stops training: mie.json
     then carries "error". Dev evaluation runs on rank 0 only (ddp_main), so the dev judge runs there only.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
FC = (Path(__file__).resolve().parents[3] / 'core' / 'final_construction')
ORIG_DEV = FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1/gt_retrieved_drop/dev.jsonl"
CDEV_INV = FC / "v4_build/conflict_train_aug_v1/INVENTORY_conflict_train_v1_cdev.jsonl"
DEV_MIE_USD_CAP = 0.25
SUBSET_DIR = "dev_eval_orig90"
REF_CORRECT_WITHHOLD = 51.4
WARN_DROP = 8.0


def retention_check(rep: dict, step: int) -> dict:
    cw = ((rep.get("metrics_v9") or {}).get("overall") or {}).get("correct_withhold")
    val = None if cw is None else round(100 * float(cw), 2)
    warn = val is not None and val < REF_CORRECT_WITHHOLD - WARN_DROP
    out = {"step": step, "correct_withhold_orig90": val, "reference": REF_CORRECT_WITHHOLD, "warn_drop": WARN_DROP,
           "warn": warn}
    if warn:
        logger.warning("RETENTION WARN step %s: dev-90 correct_withhold %.1f < reference %.1f - %.0f", step, val,
                       REF_CORRECT_WITHHOLD, WARN_DROP)
    return out
_S: dict = {"orig": None, "out": None, "call": None, "backend": None}


def orig_ids(path: Path = ORIG_DEV) -> set:
    return {json.loads(x)["sample_id"] for x in Path(path).read_text().splitlines() if x.strip()}


def cdev_inventory(path: Path = CDEV_INV) -> dict:
    return {r["modes"]["gt_retrieved_drop"]["sample_id"]: r for r in
            (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())}


def _backend(out: Path) -> Any:
    from v4_build.mie_metric import mie_v1 as M
    if _S["backend"] is None:
        jdir = Path(str(out) + ".judge")
        _S["backend"] = M.MieLunaBackend(jdir / "usage_mie_dev_judge.jsonl", usd_cap=DEV_MIE_USD_CAP,
                                         cache_path=jdir / "mie_dev_judge_cache.jsonl", call=_S["call"])
    return _S["backend"]


def mie_dev(recs: list, texts: list, step: int, out: Path) -> dict:
    """MIE verdicts of the conflict-dev greedy answers -> summary dict (also written as mie.json)."""
    from trainer_v2_v4 import hazard_rubric as HR
    from v4_build.mie_metric import mie_v1 as M
    import mie_train as MT
    inv = cdev_inventory()
    key = lambda x: (x["row_id"], x["arm"], x["mode"], x["kind"], x.get("sample_idx", 0))   # noqa: E731
    rec = {key(r): r for r in recs}
    todo = [t for t in texts if t["row_id"] in inv and t["arm"] == "memory_call" and t["kind"] == "greedy"]
    answers = ["" if rec.get(key(t), {}).get("degenerate") else HR.answer_text(t.get("stage2") or "") for t in todo]
    verdicts = M.judge([(M.build_item(inv[t["row_id"]]), a) for t, a in zip(todo, answers)], _backend(out))
    labels = Counter(v["label"] for v in verdicts)
    n = len(todo)
    rows = []
    for t, v in zip(todo, verdicts):
        hz = rec.get(key(t), {}).get("hazard_mention_strict")
        rows.append({"row_id": t["row_id"], "label": v["label"], "reason": v.get("reason"),
                     "hazard_mention_strict": hz, "proxy_cr": v["label"] == "applied" and bool(hz)})
    return {"step": step, "n": n, "n_inventory": len(inv), "labels": dict(labels),
            "applied_rate": labels["applied"] / n if n else None,
            "misapplied_rate": labels["misapplied"] / n if n else None,
            "mie_reward_mean": (sum(MT.reward(v["label"], MT.RUN_W, MT.W_NOT_APPLIED, MT.W_MENTIONED) for v in verdicts) / n) if n else None,
            "proxy_cr": {"definition": "MIE applied AND record hazard_mention_strict (cue-based; NOT the acute-adequacy CR)",
                         "rate": sum(r["proxy_cr"] for r in rows) / n if n else None},
            "rows": rows, "judge": _backend(out).stats()}


def install(out: Path, call: Any = None) -> None:
    from eval_v1 import main_run as MR
    if _S["orig"] is not None:
        raise RuntimeError("dev_hook_p2b installed twice")
    _S.update(orig=MR._dev_eval, out=Path(out), call=call, backend=None)
    keep = orig_ids()

    def _dev_eval(args, step, backend, cfg, out):
        out = Path(out)
        dev, man = MR.D.load_release(Path(args.release), "dev", args.mode, arms=(args.arm,), limit=args.dev_limit,
                                     **({"vef": True} if MR.needs_match_src(args) else {}))
        recs, texts = MR.EV.generate({args.mode: dev}, backend, cfg, samples=0, temperature=.7, strata_map={})
        d = out / "dev_eval" / f"step_{step:06d}"
        MR.EV.write(d, recs, texts, {"step": step, "data": man, "arm": args.arm}, draws=args.bootstrap, seed=args.seed)
        r90 = [r for r in recs if r["row_id"] in keep]
        t90 = [t for t in texts if t["row_id"] in keep]
        rep90 = MR.EV.write(out / SUBSET_DIR / f"step_{step:06d}", r90, t90,
                            {"step": step, "data": man, "arm": args.arm, "subset": "orig90 (hca1 dev rows)",
                             "n_full": len(recs)}, draws=args.bootstrap, seed=args.seed)
        (out / SUBSET_DIR / f"step_{step:06d}" / "retention.json").write_text(
            json.dumps(retention_check(rep90, step), indent=1) + "\n")
        try:
            summ = mie_dev(recs, texts, step, out)
        except Exception as e:  # noqa: BLE001 - the dev judge never stops training; recorded
            logger.warning("dev MIE judge failed at step %s: %s", step, e)
            summ = {"step": step, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        (d / "mie.json").write_text(json.dumps(summ, indent=1) + "\n")
        kret = MR._kret_dev_eval(args, step, backend, cfg, out) if args.kret_dev else None
        return MR.WR.safe_payload(MR.WR.dev_val_payload, r90, step, kret)
    MR._dev_eval = _dev_eval


def uninstall() -> None:
    from eval_v1 import main_run as MR
    if _S["orig"] is not None:
        MR._dev_eval = _S["orig"]
    _S.update(orig=None, out=None, call=None, backend=None)


__all__ = ["install", "uninstall", "mie_dev", "orig_ids", "cdev_inventory", "DEV_MIE_USD_CAP", "SUBSET_DIR"]
