"""Warning credit in the training reward (pilot F, owner decision 2026-10-14; NEW file). Wires
trainer_v2_v4/warning_reward.py (rule, judge, cue fallback; v4_build/warning_judge/INTEGRATION.md) into main_v4haa
behind ``--warning-credit judge`` (default ``off`` = nothing is installed).

Rule: effective risk = "unsafe" if <risk> is unsafe OR the answer warns specifically, else the tag; so on a gold-unsafe
row risk_correct = (tag == gold) OR warns_specific, and on a gold-safe row the false alarm = tag unsafe OR
warns_specific. reward_v5 / reward_v4hg are untouched: the episode is scored on a copy of its Stage-2 text whose <risk>
tag is the effective one (warning_reward.rewrite_risk_tag); masking stays reward_v5's.
In-process hooks (training only, i.e. inside patch_h's train_step; the in-loop dev eval is unchanged and never judged):
  grpo_v2.rollout            after the rollout, one batched warning judgment for the step's PReMind episodes; only
                             episodes the warning could flip are sent (warning_reward.needs_judge), duplicate prompts
                             once, <= 32 unique per rank and step, 60 s budget; failed / capped / unsent -> cue fallback
  reward_v4hg.score_episode  scores the rewritten copy; ledger key ``warning_credit``
  grpo_v2.train_step         per-step record key ``warning`` (all_sum over ranks): needed, judged, fallback,
                             fallback_rate, credit / false-alarm flips, judge spend
Backend: gpt-6-luna@medium via wd_judge.WarningJudge (= hazard_judge_api.ApiHazardJudge client) with its OWN ledger
OUT.judge/usage_warning_judge[.rank<r>].jsonl, reply cache and USD cap (default $3 per run, split evenly over ranks),
independent of the hazard judge's cap.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from trainer_v2_v4 import warning_reward as WR

logger = logging.getLogger(__name__)
MODES = ("off", "judge")
USD_CAP = 3.0
MODEL = "gpt-6-luna@medium"
COUNTERS = ("episodes", "needed", "judged", "fallback", "credit", "false_alarm")


@dataclass
class WState:
    hook: Any
    judge: Any
    step: dict = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
    originals: dict = field(default_factory=dict)


W: Optional[WState] = None


def judge_paths(out: Path) -> tuple[Path, Path]:
    rank = int(os.environ.get("RANK", "0"))
    sfx = "" if rank == 0 else f".rank{rank}"
    jdir = out.parent / (out.name + ".judge")          # outside OUT (main_run refuses a non-empty fresh OUT)
    return jdir / f"usage_warning_judge{sfx}.jsonl", jdir / f"warning_judge_cache{sfx}.jsonl"


def _in_train() -> bool:
    from trainer_v2_v4 import patch_h as PH
    return PH.H is not None and PH.H.in_train


def _rollout(items: list, images: list, backend: Any, cfg: Any, **kw: Any) -> list:
    from trainer_v2_v4 import covip_aux as CA
    eps = W.originals["G.rollout"](items, images, backend, cfg, **kw)
    if not _in_train():
        return eps
    mine = [ep for ep in eps if not CA.is_aux(ep.item) and ep.item.row_id in W.hook.refs]
    res = W.hook.score_step([(ep.item.row_id, ep.stage2_text) for ep in mine])
    for ep, r in zip(mine, res):
        ep._warning = r
        ref = W.hook.refs[ep.item.row_id]
        need = WR.needs_judge(ref.gold_risk, r["risk_pred"])
        s = W.step
        s["episodes"] += 1
        s["needed"] += need
        s["judged"] += need and r["backend"] == "judge"
        s["fallback"] += need and r["backend"] != "judge"
        s["credit"] += r["changed"] and ref.gold_risk == "unsafe"
        s["false_alarm"] += r["changed"] and ref.gold_risk != "unsafe"
    return eps


def score_with(orig: Any, ep: Any, cfg: Any, hz: Any) -> dict:
    r = getattr(ep, "_warning", None)
    if r is None:
        return orig(ep, cfg, hz)
    sc = ep
    if r["changed"]:
        sc = copy.copy(ep)
        sc.stage2_text = WR.rewrite_risk_tag(ep.stage2_text, r["effective_risk"])
    led = orig(sc, cfg, hz)
    led = {**led, "warning_credit": {k: r[k] for k in ("risk_pred", "warns_specific", "effective_risk", "changed",
                                                        "backend")}}
    return led


def step_record(dist: Any = None) -> dict:
    vals = [W.step[k] for k in COUNTERS]
    st = W.judge.stats() if W.judge is not None and hasattr(W.judge, "stats") else {}
    api = [st.get("usd_spent", 0.0), float(bool(st.get("capped"))), st.get("api_failed", 0)]
    if dist is not None:
        vals, api = dist.all_sum(vals), dist.all_sum(api)
    rec = dict(zip(COUNTERS, (int(v) for v in vals)))
    rec["fallback_rate"] = round(rec["fallback"] / rec["needed"], 6) if rec["needed"] else 0.0
    rec.update(judge_usd_total=round(api[0], 6), judge_capped_ranks=int(api[1]), judge_api_failed_total=int(api[2]))
    return rec


def _train_step(*a: Any, **kw: Any) -> dict:
    W.step = dict.fromkeys(COUNTERS, 0)
    out = W.originals["G.train_step"](*a, **kw)
    out["warning"] = step_record(kw.get("dist"))
    logger.info("warning credit: %s", out["warning"])
    return out


def install(mode: str, *, release: Path, out: Path, usd_cap: float = USD_CAP, model: str = MODEL,
            judge: Any = None) -> Optional[WState]:
    """``off``: nothing. ``judge``: hooks above; ``judge`` = an injected judge (tests) or the paid WarningJudge."""
    global W
    if mode not in MODES:
        raise ValueError(f"warning_credit must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return None
    if W is not None:
        raise RuntimeError("trainer_v2_v4.warning_credit already installed")
    if judge is None:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        led, cache = judge_paths(Path(out))
        led.parent.mkdir(parents=True, exist_ok=True)
        judge = WR.make_judge(led, usd_cap=usd_cap / world, cache_path=cache, model=model)
        judge.health_check()                              # no paid call (records the served spec)
    hook = WR.WarningRewardHook(WR.load_refs(Path(release)), backend="judge", judge=judge)
    W = WState(hook=hook, judge=judge)
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    from trainer_v2_v4 import reward_v4hg as RHG
    W.originals = {"G.rollout": G.rollout, "G.train_step": G.train_step, "RHG.score_episode": RHG.score_episode,
                   "RV.STEP_EXTRA_KEYS": RV.STEP_EXTRA_KEYS}
    orig_score = RHG.score_episode
    G.rollout, G.train_step = _rollout, _train_step
    RHG.score_episode = lambda ep, cfg, hz: score_with(orig_score, ep, cfg, hz)
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("warning",)
    logger.info("warning credit installed: judge=%s usd_cap=%s refs=%d", model, usd_cap, len(hook.refs))
    return W


def uninstall() -> None:
    global W
    if W is None:
        return
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    from trainer_v2_v4 import reward_v4hg as RHG
    G.rollout, G.train_step = W.originals["G.rollout"], W.originals["G.train_step"]
    RHG.score_episode, RV.STEP_EXTRA_KEYS = W.originals["RHG.score_episode"], W.originals["RV.STEP_EXTRA_KEYS"]
    W = None


__all__ = ["MODES", "USD_CAP", "MODEL", "COUNTERS", "install", "uninstall", "score_with", "step_record",
           "judge_paths"]
