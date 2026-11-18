"""Phase II'' hacking monitors (REWARD_BREAKEVEN_v1.md section 4), logging only (rewards untouched).

Per training step, over every scored rollout of every DAPO round (all ranks summed):
  g0_call_rate         share of call_pdb decisions on the non-acute G0 replay rows (labels ``g0_replay``)
  conflict_mentioned_only_share   mentioned_only / MIE-judged episodes on conflict rows (mie_train counts)
  acute_generic_only_share        hazard state generic_only / acute episodes with a hazard state (final ledger v4h)
  acute_stillpen_rate             acute episodes with a stillpen penalty < 0 / acute episodes
Hooks: algo_v2.dynamic_rollout (outermost score wrapper, installed after mie_train) records each final ledger;
grpo_v2.train_step is wrapped to put the step's ``p2c_monitor`` block into the train_step result, and
run_v2.STEP_EXTRA_KEYS gains "p2c_monitor", so the block is in the STEP log line and in steps.jsonl.
"""
from __future__ import annotations

from typing import Any

KEYS = ("g0_n", "g0_call", "conf_judged", "conf_mentioned_only", "acute_n", "acute_state_n", "acute_generic_only",
        "acute_stillpen")
_S: dict = {"orig": {}, "c": dict.fromkeys(KEYS, 0), "labels": None}


def record(ep: Any, led: dict) -> None:
    lab = (_S["labels"] or {}).get(ep.item.row_id) or {}
    c = _S["c"]
    if lab.get("g0_replay"):
        c["g0_n"] += 1
        c["g0_call"] += led.get("decision") == "call_pdb"
    v = led.get("v4h") or {}
    if lab.get("acute"):
        c["acute_n"] += 1
        c["acute_stillpen"] += (led.get("stillpen") or {}).get("penalty", 0) < 0
        if v.get("acute") and "state" in v:
            c["acute_state_n"] += 1
            c["acute_generic_only"] += v["state"] == "generic_only"


def block(c: dict) -> dict:
    def rate(a, b):
        return round(c[a] / c[b], 6) if c[b] else None
    return {"g0_call_rate": rate("g0_call", "g0_n"),
            "conflict_mentioned_only_share": rate("conf_mentioned_only", "conf_judged"),
            "acute_generic_only_share": rate("acute_generic_only", "acute_state_n"),
            "acute_stillpen_rate": rate("acute_stillpen", "acute_n"), **{k: int(c[k]) for k in KEYS}}


def install(labels: dict) -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    import mie_train as MT
    if _S["orig"]:
        raise RuntimeError("monitor_p2c installed twice")
    _S["labels"] = labels
    dyn, ts = AL.dynamic_rollout, G.train_step
    _S["orig"] = {"dyn": dyn, "ts": ts, "keys": RV.STEP_EXTRA_KEYS}

    def dynamic_rollout(items, images, backend, cfg, refill, rollout, score, *a, **kw):
        def scored(ep, c):
            led = score(ep, c)
            record(ep, led)
            return led
        return dyn(items, images, backend, cfg, refill, rollout, scored, *a, **kw)

    def train_step(*a, **kw):
        _S["c"] = dict.fromkeys(KEYS, 0)
        out = ts(*a, **kw)
        mc = MT._S["counts"]
        _S["c"]["conf_judged"] = int(mc.get("n_eligible", 0))
        _S["c"]["conf_mentioned_only"] = int(mc.get("mentioned_only", 0))
        vals = [_S["c"][k] for k in KEYS]
        dist = kw.get("dist")
        if dist is not None:
            vals = dist.all_sum(vals)
        out["p2c_monitor"] = block(dict(zip(KEYS, vals)))
        return out
    AL.dynamic_rollout, G.train_step = dynamic_rollout, train_step
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("p2c_monitor",)


def uninstall() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    o = _S["orig"]
    if o:
        AL.dynamic_rollout, G.train_step, RV.STEP_EXTRA_KEYS = o["dyn"], o["ts"], o["keys"]
    _S.update(orig={}, c=dict.fromkeys(KEYS, 0), labels=None)


__all__ = ["install", "uninstall", "record", "block", "KEYS"]
