"""Phase II-quadruple-prime copy (2026-10-24): outcome_labels reads the gated MIE outcome (mie_train gate);
p2d_mastery gains conflict_draws_this_step = [row, src, band, tier] per conflict draw (p2e curated bands).

Phase II-triple-prime runtime hooks for the mastery-aware conflict stream (stillpen mode "p2d") + retention monitor.

  * curriculum_v2_1.group_outcome (called by main_v4hcur21's _informative for every rolled-out group, the outcome is
    gathered over ranks and handed to sampler.observe on every rank) -> the outcome gains ``mie_labels`` (the MIE label
    of each rollout of the group, from the final ledger's ``mie`` block; [] for non-MIE rows).
  * algo_v2.dynamic_rollout (outermost; installed last): the ``rollout`` callable is wrapped so that the prompts of the
    round that are in stillpen.HOT (no_signal replays drawn this step, within the 5 % cap) are generated in a separate
    grpo_v2.rollout call with temperature = NO_SIGNAL_T (grpo_v2.rollout's own ``temperature`` argument -> backend
    generate); every other prompt is generated in one call at the backend temperature (1.0), exactly as before.
    Episodes are re-assembled in the original item order (G contiguous episodes per item). Per-prompt temperatures are
    logged. With no HOT prompt in a round the original call is made unchanged. (Sampling at 1.1 while the loss uses
    on-policy log-probs at 1.0 -- sampler log-probs are NaN, IS ratio 1 -- is a deliberate small off-policy bias.)
  * grpo_v2.train_step -> result keys ``p2d_mastery`` (sampler counts after the previous step's observe + this step's
    temperatures) and ``retention`` (this step's trained records: correct_withhold on gt_absent rows, cite_f1 on present
    rows, call rate on G0 safe rows); both added to run_v2.STEP_EXTRA_KEYS -> STEP line + steps.jsonl.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)
_S: dict = {"orig": {}, "temps": []}


def no_signal_t() -> float:
    import stillpen as SP
    return SP.MASTERY_CFG["no_signal_temperature"]


def outcome_labels(ledgers: list) -> list:
    """Per-rollout MIE label for mastery: the PAID outcome (Phase II-quadruple-prime: applied_ungated when the hazard
    gate withheld the reward), else the judge label (p2d ledgers have no outcome)."""
    return [(led.get("mie") or {}).get("outcome") or (led.get("mie") or {}).get("label")
            for led in ledgers if isinstance(led, dict) and "mie" in led]


def fact_labels(ledgers: list) -> Optional[list]:
    """Per rollout: hit (bonus fired) | miss (eligible, fact fraction < 1, cited, no other blocker) | other (eligible, blocked) | na.
    None when no ledger carries a fact_bonus block (bonus off) -> the outcome is byte-identical to the base package."""
    fb = [led.get("fact_bonus") for led in ledgers if isinstance(led, dict)]
    if not any(isinstance(x, dict) for x in fb):
        return None
    return [("na" if not isinstance(x, dict) or not x.get("eligible") else "hit" if x.get("fired") else "miss" if x.get("miss") else "other") for x in fb]


def split_rollout(rollout: Any) -> Any:
    import stillpen as SP

    def rollout_t(batch_items, batch_images, backend, cfg, **kw):
        hot = [i for i, it in enumerate(batch_items) if it.row_id in SP.HOT]
        if not hot:
            _S["temps"] += [[it.row_id, None] for it in batch_items]
            return rollout(batch_items, batch_images, backend, cfg, **kw)
        cold = [i for i in range(len(batch_items)) if i not in hot]
        g = cfg.num_generations
        got = {}
        if cold:
            eps = rollout([batch_items[i] for i in cold], [batch_images[i] for i in cold], backend, cfg, **kw)
            got.update({i: eps[j * g:(j + 1) * g] for j, i in enumerate(cold)})
        eps = rollout([batch_items[i] for i in hot], [batch_images[i] for i in hot], backend, cfg,
                      **{**kw, "temperature": no_signal_t()})
        got.update({i: eps[j * g:(j + 1) * g] for j, i in enumerate(hot)})
        _S["temps"] += [[it.row_id, no_signal_t() if i in hot else None] for i, it in enumerate(batch_items)]
        return [ep for i in range(len(batch_items)) for ep in got[i]]
    return rollout_t


def retention(records: list) -> dict:
    def mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return (round(sum(xs) / len(xs), 6), len(xs)) if xs else (None, 0)
    cw, n_cw = mean([r.get("correct_withhold") for r in records if r.get("condition") == "gt_absent"])
    cf, n_cf = mean([r.get("cite_f1") for r in records if r.get("condition") != "gt_absent"
                     and r.get("memory_required")])
    g0 = [r for r in records if str(r.get("row_type", "")).startswith("G0") and r.get("risk") == "safe"]
    return {"absent_correct_withhold": cw, "n_absent": n_cw, "present_cite_f1": cf, "n_present": n_cf,
            "g0_safe_call_rate": round(sum(r.get("decision") == "call_pdb" for r in g0) / len(g0), 6) if g0 else None,
            "n_g0_safe": len(g0)}


def install() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    from trainer_v2_v4 import curriculum_v2_1 as C21
    import stillpen as SP
    if _S["orig"]:
        raise RuntimeError("mastery_p2d installed twice")
    go, dyn, ts = C21.group_outcome, AL.dynamic_rollout, G.train_step
    _S["orig"] = {"go": go, "dyn": dyn, "ts": ts, "keys": RV.STEP_EXTRA_KEYS}

    def group_outcome(row_id, ledgers, acute, *a, **kw):
        out = go(row_id, ledgers, acute, *a, **kw)
        fl = fact_labels(ledgers)
        return {**out, "mie_labels": outcome_labels(ledgers), **({"fact_labels": fl} if fl is not None else {})}

    def dynamic_rollout(items, images, backend, cfg, refill, rollout, score, *a, **kw):
        return dyn(items, images, backend, cfg, refill, split_rollout(rollout), score, *a, **kw)

    def train_step(*a, **kw):
        _S["temps"] = []
        out = ts(*a, **kw)
        s = SP.ACTIVE["sampler"]
        dist = kw.get("dist")
        temps = _S["temps"] if dist is None else [x for p in dist.gather(_S["temps"]) for x in p]
        out["p2d_mastery"] = {**(s.mastery_counts() if s is not None else {}),
                              "conflict_draws_this_step": ([[r, v["src"], v.get("band"), v.get("tier"), v.get("difficulty")]
                                                            for r, v in sorted(s.cf_src.items())] if s is not None else []),
                              "hot_rows_this_step": sorted({r for r, t in temps if t is not None}),
                              "temperature_by_prompt": temps, "no_signal_temperature": no_signal_t()}
        out["retention"] = retention(out.get("records") or [])
        return out
    C21.group_outcome, AL.dynamic_rollout, G.train_step = group_outcome, dynamic_rollout, train_step
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("p2d_mastery", "retention")


def uninstall() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    from trainer_v2_v4 import curriculum_v2_1 as C21
    o = _S["orig"]
    if o:
        C21.group_outcome, AL.dynamic_rollout, G.train_step, RV.STEP_EXTRA_KEYS = o["go"], o["dyn"], o["ts"], o["keys"]
    _S.update(orig={}, temps=[])


__all__ = ["install", "uninstall", "split_rollout", "retention", "outcome_labels", "no_signal_t", "fact_labels"]
