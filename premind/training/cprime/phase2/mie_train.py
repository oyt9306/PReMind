"""Phase II-quadruple-prime copy (2026-10-24, PREREG_PHASE3_MIE.md Amendment 9; REWARD_BREAKEVEN_v2.md): + the
hazard GATE on the applied reward: an ``applied`` verdict is paid (+W) only when the rollout's hazard state (the final
ledger's ``v4h`` block = reward_v4h.hazard_info, the same state the hazard term and the hazard_first gate use) is
``specific`` OR the answer is flagged contraindicated (owner 2026-10-24: the judge over-flags contraindicated on
standard first aid; a contraindicated answer keeps its own -0.15 hazard penalty, no double penalty); it is zeroed
(``applied_ungated``) only for generic_only / none / no hazard verdict. W = 0.15 (owner 2026-10-24). misapplied -W, mentioned_only / not_applied -0.05 unchanged. The ledger's ``mie`` block gains ``outcome`` (the
label actually paid: applied / applied_ungated / ...) and ``gate``; mastery (mastery_p2d) reads ``outcome``, so an
ungated applied does not count towards mastery. GATE = False (module default) is the p2d path, bit-identical.
INVENTORY -> the p2e inventory (129 + 87 conflict_train_v2 rows).

Phase II'' copy (REWARD_BREAKEVEN_v1.md section 5): + mentioned_only -W_MENTIONED (module default 0 -> off path
unchanged); applied +0.2, misapplied -0.2, mentioned_only -0.05, not_applied -0.05, unjudged 0.

Phase II' copy of train_mie_p3_4b/mie_train.py (2026-10-23, PREREG_PHASE3_MIE.md Amendment 2 + owner amendment:
applied +0.2, misapplied -0.2, mentioned_only 0, not_applied -0.05, unjudged 0). Only the constants, the not_applied
weight (``W_NOT_APPLIED``; module default 0 -> the off path stays bit-identical) and this header changed.

MIE reward term: memory-informed emergency response on the MIE-inventory (conflict) training rows.

Frozen trainer modules are imported read-only and wrapped in-process (never edited):
  * trainer_v2.algo_v2.dynamic_rollout: the ``rollout`` callable it receives is wrapped so that, after every DAPO
    round's rollout (after patch_h's hazard prefetch, which lives inside grpo_v2.rollout), the round's ELIGIBLE
    episodes (row in the MIE train inventory, not CoViP aux) are judged ONCE (mie_v1.judge over MieLunaBackend, own
    ledger / cache / USD cap under OUT.judge); the ``score`` callable is wrapped to add the term to the ledger
    ``total`` and to record a top-level ``mie`` block. Installed AFTER stillpen.install(): this wrapper is the outer
    one, so stillpen's penalty is subtracted from the MIE-adjusted total (both are additive; order only matters for
    which wrapper sees which ledger).
  * trainer_v2.run_v2.run_steps: per-step counts (all ranks) -> rec["mie"]; per-rollout verdicts -> OUT/mie_rollouts.jsonl.
Judge input = the final <answer> text only (hazard_rubric.answer_text of the Stage-2 text; mie_v1 clips 260 words).
Reward: applied +W, misapplied -W, not_applied -W_NA, mentioned_only / unjudged 0. W = 0 (module default) leaves every
ledger object untouched and makes no judge call (bit-identical totals).
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
FC = (Path(__file__).resolve().parents[3] / 'core' / 'final_construction')
INVENTORY = FC / "v4_build/conflict_train_aug_v1/p2e_release/MIE_INVENTORY_TRAIN_P2E_v1.jsonl"
MODE = "gt_retrieved_drop"
DEFAULT_W = 0.0                 # module default: off (bit-identical)
DEFAULT_W_NA = 0.0              # module default not_applied weight
DEFAULT_W_MO = 0.0              # module default mentioned_only weight
RUN_W = 0.15                    # Phase II-quadruple-prime value (owner 2026-10-24; Phase II'-III 0.2)
W_NOT_APPLIED = 0.05            # Phase II' not_applied penalty (pinned; owner amendment 2026-10-23)
W_MENTIONED = 0.05              # Phase II'' mentioned_only penalty (pinned; REWARD_BREAKEVEN_v1 section 2)
DEFAULT_GATE = False            # module default: applied paid whatever the hazard state (p2d path)
RUN_GATE = True                 # Phase II-quadruple-prime (pinned): applied paid only if state specific OR contraindicated
MIE_USD_CAP = 0.5               # MIE judge total (split evenly over DDP ranks); owner total 3.0 = 0.80 + 0.25 + 1.70 + dev 0.25
CONCURRENCY = 8
SIGN = {"applied": 1.0, "misapplied": -1.0, "mentioned_only": 0.0, "not_applied": 0.0, "unjudged": 0.0}  # x W
ROLLOUT_LOG = "mie_rollouts.jsonl"
_S: dict = {"w": DEFAULT_W, "w_na": DEFAULT_W_NA, "w_mo": DEFAULT_W_MO, "gate": DEFAULT_GATE, "inv": {}, "backend": None, "out": None, "counts": Counter(), "recs": [], "orig": {}}


def load_inventory(path: Path = INVENTORY) -> dict[str, dict]:
    """{release sample_id: inventory row}; every row must be decisive with gold_applies true."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["fact_role"] != "decisive" or r["gold_applies"] is not True:
                raise ValueError(f"MIE inventory row not eligible: {r['canon_id']}")
            out[r["modes"][MODE]["sample_id"]] = r
    return out


def reward(label: str, w: float, w_na: float = DEFAULT_W_NA, w_mo: float = DEFAULT_W_MO) -> float:
    if label == "not_applied":
        return -w_na
    if label == "mentioned_only":
        return -w_mo
    return w * SIGN.get(label, 0.0)


def hazard_ok(led: dict) -> bool:
    """The rollout's hazard state (final ledger v4h block) is specific, or the answer is flagged contraindicated
    (owner 2026-10-24); False for generic_only / none / no hazard verdict."""
    v = led.get("v4h") or {}
    return bool(v.get("acute")) and (v.get("state") == "specific" or bool(v.get("contraindicated")))


def gated_label(label: str, led: dict, gate: bool) -> str:
    """The label that is paid: applied -> applied_ungated (reward 0) when the gate is on and the hazard is not ok."""
    return "applied_ungated" if gate and label == "applied" and not hazard_ok(led) else label


def _eligible(ep: Any) -> bool:
    from trainer_v2_v4 import covip_aux as CA
    return not CA.is_aux(ep.item) and ep.item.row_id in _S["inv"]


def answer(ep: Any) -> str:
    from trainer_v2_v4 import hazard_rubric as HR
    return HR.answer_text(ep.stage2_text)


def prefetch(eps: list) -> None:
    """One judge batch for a DAPO round's eligible episodes (verdict stored on the episode)."""
    from v4_build.mie_metric import mie_v1 as M
    todo = [ep for ep in eps if _eligible(ep) and not hasattr(ep, "_mie")]
    if not todo:
        return
    got = M.judge([(M.build_item(_S["inv"][ep.item.row_id]), answer(ep)) for ep in todo], _S["backend"])
    for ep, v in zip(todo, got, strict=True):
        ep._mie = v
    _S["counts"]["n_judge_batches"] += 1


def score_with_mie(score: Any, ep: Any, cfg: Any) -> dict:
    led = score(ep, cfg)
    w = _S["w"]
    if not w or not _eligible(ep):
        return led
    v = getattr(ep, "_mie", None) or {"label": "unjudged", "reason": "not prefetched", "source": "none"}
    out = gated_label(v["label"], led, _S["gate"])
    r = reward(out, w, _S["w_na"], _S["w_mo"]) if _S["reward_enabled"] else 0.0
    c = _S["counts"]
    c["n_eligible"] += 1
    c[v["label"]] += 1
    c["applied_ungated"] += out == "applied_ungated"
    c["reward_sum"] += r
    hz = led.get("v4h") or {}
    gate = {"on": _S["gate"], "state": hz.get("state"), "contraindicated": hz.get("contraindicated"), "ok": hazard_ok(led)}
    _S["recs"].append({"row_id": ep.item.row_id, "label": v["label"], "outcome": out, "reward": r, "gate": gate,
                       "source": v.get("source"), "quote_ok": v.get("quote_ok"), "reason": v.get("reason"),
                       "answer": answer(ep)})
    return {**led, "total": led["total"] + r, "mie": {"reward_enabled": _S["reward_enabled"], "w": w if _S["reward_enabled"] else 0.0, "w_na": _S["w_na"] if _S["reward_enabled"] else 0.0, "w_mo": _S["w_mo"] if _S["reward_enabled"] else 0.0, "label": v["label"],
                                                       "outcome": out, "gate": gate, "reward": r}}


def _step() -> Optional[int]:
    from trainer_v2_v4 import patch_h as PH
    return PH.H.step if PH.H is not None else None


def install(out: Path, *, w: float = DEFAULT_W, w_na: float = DEFAULT_W_NA, w_mo: float = DEFAULT_W_MO,
            gate: bool = DEFAULT_GATE, inventory: Path = INVENTORY,
            call: Any = None, usd_cap: float = MIE_USD_CAP, reward_enabled: bool = True) -> None:
    """``call`` = injected provider call (CPU tests: fake); None = the paid training-judge client on first use."""
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import run_v2 as RV
    if _S["orig"]:
        raise RuntimeError("mie_train hooks installed twice")
    if (w_na or w_mo) and not w:
        raise ValueError("W_NOT_APPLIED / W_MENTIONED need W > 0 (the off path is W = 0)")
    _S.update(reward_enabled=bool(reward_enabled), w=float(w), w_na=float(w_na), w_mo=float(w_mo), gate=bool(gate), inv=load_inventory(inventory), out=Path(out), counts=Counter(), recs=[])
    if _S["w"]:
        from v4_build.mie_metric import mie_v1 as M
        rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        sfx = "" if rank == 0 else f".rank{rank}"
        jdir = Path(str(out) + ".judge")
        _S["backend"] = M.MieLunaBackend(jdir / f"usage_mie_judge{sfx}.jsonl", usd_cap=usd_cap / world,
                                         concurrency=CONCURRENCY, cache_path=jdir / f"mie_judge_cache{sfx}.jsonl",
                                         call=call)
    dyn, run = AL.dynamic_rollout, RV.run_steps
    _S["orig"] = {"dyn": dyn, "run": run}

    def dynamic_rollout(items, images, backend, cfg, refill, rollout, score, *a, **kw):
        if not _S["w"]:
            return dyn(items, images, backend, cfg, refill, rollout, score, *a, **kw)
        _S["counts"]["n_dyn_calls"] += 1

        def rollout_mie(*ra, **rk):
            eps = rollout(*ra, **rk)
            prefetch(eps)
            return eps
        return dyn(items, images, backend, cfg, refill, rollout_mie, lambda ep, c: score_with_mie(score, ep, c),
                   *a, **kw)

    def run_steps(*a, **kw):
        _S["counts"], _S["recs"] = Counter(), []
        recs = run(*a, **kw)
        if not _S["w"]:
            return recs
        dist = kw.get("dist")
        step = _step()
        mine = [{"step": step, "rank": int(os.environ.get("RANK", "0")), **r} for r in _S["recs"]]
        parts = dist.gather(dict(_S["counts"])) if dist is not None else [dict(_S["counts"])]
        rows = [x for p in dist.gather(mine) for x in p] if dist is not None else mine
        tot = Counter()
        for p in parts:
            tot.update(p)
        if not tot.get("n_dyn_calls"):
            raise RuntimeError("mie_train: dynamic_rollout hook never ran this step (dynamic sampling off?)")
        st = _S["backend"].stats() if _S["backend"] is not None else {}
        recs[0]["mie"] = {"reward_enabled": _S["reward_enabled"], "w": _S["w"] if _S["reward_enabled"] else 0.0, "w_na": _S["w_na"] if _S["reward_enabled"] else 0.0, "w_mo": _S["w_mo"] if _S["reward_enabled"] else 0.0, "gate": _S["gate"], **{k: round(v, 6) for k, v in sorted(tot.items())},
                          "usd_spent_rank": st.get("usd_spent_total_ledger"), "capped_rank": st.get("capped")}
        if (dist is None or dist.is_main) and _S["out"] is not None and rows:
            with (_S["out"] / ROLLOUT_LOG).open("a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r, default=str) + "\n" for r in rows))
        return recs
    AL.dynamic_rollout, RV.run_steps = dynamic_rollout, run_steps
    logger.info("mie_train installed: w=%s inventory=%d rows", _S["w"], len(_S["inv"]))


def uninstall() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import run_v2 as RV
    if _S["orig"]:
        AL.dynamic_rollout, RV.run_steps = _S["orig"]["dyn"], _S["orig"]["run"]
    _S.update(w=DEFAULT_W, w_na=DEFAULT_W_NA, w_mo=DEFAULT_W_MO, gate=DEFAULT_GATE, inv={}, backend=None, out=None, counts=Counter(), recs=[], orig={})


__all__ = ["DEFAULT_GATE", "RUN_GATE", "hazard_ok", "gated_label", "DEFAULT_W", "DEFAULT_W_NA", "DEFAULT_W_MO", "RUN_W", "W_NOT_APPLIED", "W_MENTIONED", "MIE_USD_CAP", "SIGN", "install", "uninstall", "reward", "load_inventory", "prefetch",
           "score_with_mie"]
