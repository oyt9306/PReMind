"""Batch-level CoViP mixing + leak monitors (P1, owner 2026-10-13), installed ON TOP of ``patch.install`` (called with
aux_ratio 0) by ``main_v4b`` only. New file: patch.py / kl_ref.py / reward_v4.py / main_v4.py / pins_v4.py (read and
pinned by the running P3 job) are not modified.

Extra in-process hooks (all restored by ``uninstall``):
  eval_v1.main_run.train_one_step / eval_v1.ddp_main.ddp_train_one_step   record (step, rank, dist) for the step;
  trainer_v2.run_v2.run_steps     appends this rank's CoViP prompts of the step (batch_mix.AuxBatchPlan) after the
                                  step's PReMind prompts; CoViP images via covip_aux.item_images; the PReMind cursor
                                  (n_prompts_drawn) is corrected so the sampler stream is exactly P3's;
  trainer_v2.algo_v2.dynamic_rollout / rl_weights   per-source DAPO filter + refill and per-source token-level loss;
  trainer_v2.grpo_v2.rollout      leak / format counters on training rollouts (leak.py);
  trainer_v2.grpo_v2.train_step   splits the step result by source: every PReMind key (records, episodes, rewards,
                                  reward_mean, dapo_* counts, ref_kl_mean, used_items) holds the PReMind subset only,
                                  so metrics, the over-abstain monitor and the stop rule read exactly what P3's do;
                                  CoViP stats go to ``aux_batch``, monitors to ``leak``;
  eval_v1.wandb_layout.train_payload   + aux/* and leak/* scalars.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from trainer_v2 import algo_v2 as AL
from trainer_v2 import grpo_v2 as G
from trainer_v2 import run_v2 as RV
from trainer_v2_v4 import batch_mix as BM
from trainer_v2_v4 import covip_aux as CA
from trainer_v2_v4 import leak as LK

logger = logging.getLogger(__name__)


@dataclass
class BState:
    plan: BM.AuxBatchPlan
    pool: list                               # permuted, overlap-excluded CoViP items
    ctx: Optional[dict] = None               # {"step", "rank", "dist"} of the running step
    refill_taken: int = 0
    in_train: bool = False
    leak: LK.Counter = field(default_factory=LK.Counter)
    last_info: Optional[dict] = None
    originals: dict = field(default_factory=dict)


B: Optional[BState] = None


def _aux_for(step: int, rank: int) -> list:
    return [B.pool[p] for p in B.plan.positions(step, rank)]


def _aux_refill(k: int) -> tuple[list, list]:
    c = B.ctx or {"step": 0, "rank": 0}
    pos = B.plan.refill_positions(c["step"], c["rank"], B.refill_taken, k)
    B.refill_taken += k
    got = [B.pool[p] for p in pos]
    return got, [CA.item_images(it) for it in got]


# ----------------------------------------------------------------------------- hooks
def _train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, sampler=None,
                    cursor=None, vef=None):
    B.ctx = {"step": step, "rank": 0, "dist": None}
    try:
        return B.originals["MR.train_one_step"](items, step, args, backend, optimizer, params, cfg, opsd, monitor, out,
                                                sampler, cursor, vef=vef)
    finally:
        B.ctx = None


def _ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler,
                        cursor, loader=None, cuda=True, vef=None):
    B.ctx = {"step": step, "rank": dist.rank, "dist": dist}
    try:
        kw = {} if loader is None else {"loader": loader}
        return B.originals["DM.ddp_train_one_step"](items, step, args, backend, optimizer, params, cfg, opsd, monitor,
                                                    out, dist, sampler, cursor, cuda=cuda, vef=vef, **kw)
    finally:
        B.ctx = None


def _run_steps(items, loader, backend, optimizer, params, cfg, *, steps, prompts_per_step, **kw):
    orig = B.originals["RV.run_steps"]
    if B.ctx is None or steps != 1:
        return orig(items, loader, backend, optimizer, params, cfg, steps=steps, prompts_per_step=prompts_per_step, **kw)
    aux = _aux_for(B.ctx["step"], B.ctx["rank"])
    B.refill_taken = 0
    p = prompts_per_step
    mixed = list(items[:p]) + aux + list(items[p:])

    def load(it):
        return CA.item_images(it) if CA.is_aux(it) else loader(it)
    recs = orig(mixed, load, backend, optimizer, params, cfg, steps=1, prompts_per_step=p + len(aux), **kw)
    for r in recs:
        if "n_prompts_drawn" in r:
            r["n_prompts_drawn"] -= len(aux)                 # the PReMind sampler cursor sees only PReMind draws
    return recs


def _dynamic_rollout(items, images, backend, cfg, refill, rollout, score, prefetch=None):
    if prefetch is not None:
        raise RuntimeError("batch-level CoViP mixing does not support VEF presets")
    eps, its, ims, info = BM.dynamic_rollout_by_source(items, images, backend, cfg, {BM.PM: refill, BM.AUX: _aux_refill},
                                                       rollout, score, AL.informative, AL.shaped_rewards)
    B.last_info = info["by_source"]
    return eps, its, ims, info


def _rl_weights(eps, cfg, totals=None):
    return BM.rl_weights_by_source(eps, cfg, totals, dist=(B.ctx or {}).get("dist"))


def _rollout(items, images, backend, cfg, **kw):
    eps = B.originals["G.rollout"](items, images, backend, cfg, **kw)
    if B.in_train:
        for ep in eps:
            B.leak.add(aux=CA.is_aux(ep.item), stage1=ep.stage1_text, stage2=ep.stage2_text)
    return eps


def _split(out: dict, dist: Any) -> dict:
    eps = out["episodes"]
    is_aux = [str(e["row_id"]).startswith("covip:") for e in eps]
    keep = lambda xs: [x for x, a in zip(xs, is_aux) if not a]  # noqa: E731
    take = lambda xs: [x for x, a in zip(xs, is_aux) if a]      # noqa: E731
    a_eps, a_recs, a_rew = take(eps), take(out["records"]), take(out["rewards"])
    out["episodes"], out["records"], out["rewards"] = keep(eps), keep(out["records"]), keep(out["rewards"])
    n = len(out["rewards"])
    out["n_episodes"] = n
    out["reward_mean"] = sum(out["rewards"]) / n if n else None
    if "used_items" in out:
        out["used_items"] = [it for it in out["used_items"] if not CA.is_aux(it)]
    kls = [e["ref_kl"] for e in out["episodes"] if e.get("ref_kl") is not None]
    if "ref_kl_mean" in out:
        out["ref_kl_mean"] = sum(kls) / len(kls) if kls else None
    src = B.last_info or {}
    vals = [src.get(s, {}).get(k, 0) for s in (BM.PM, BM.AUX) for k in ("wanted", "drawn", "filtered", "kept")]
    if dist is not None:
        vals = [int(v) for v in dist.all_sum(vals)]
    pm, ax = dict(zip(("wanted", "drawn", "filtered", "kept"), vals[:4])), dict(zip(("wanted", "drawn", "filtered", "kept"), vals[4:]))
    if "dapo_filtered_groups" in out and B.last_info is not None:
        out.update(dapo_filtered_groups=pm["filtered"], dapo_prompts_drawn=pm["drawn"], dapo_kept_groups=pm["kept"])
    akl = [e["ref_kl"] for e in a_eps if e.get("ref_kl") is not None]
    out["aux_batch"] = {"n_prompts_planned": ax["wanted"], "n_groups_kept": ax["kept"], "n_episodes": len(a_eps),
                        "reward_mean": sum(a_rew) / len(a_rew) if a_rew else None,
                        "ref_kl_mean": sum(akl) / len(akl) if akl else None,
                        "degenerate_rate": (sum(bool(r["degenerate"]) for r in a_recs) / len(a_recs)) if a_recs else None,
                        "completion_len_mean": (sum(e["n_completion"] for e in a_eps) / len(a_eps)) if a_eps else None,
                        "filtered_group_rate": ax["filtered"] / ax["drawn"] if ax["drawn"] else None,
                        "premind_filtered_group_rate": pm["filtered"] / pm["drawn"] if pm["drawn"] else None}
    return out


def _train_step(*a, **kw):
    B.leak = LK.Counter()
    B.last_info = None
    B.in_train = True
    try:
        out = B.originals["G.train_step"](*a, **kw)
    finally:
        B.in_train = False
    dist = kw.get("dist")
    vals = B.leak.values() if dist is None else [int(v) for v in dist.all_sum(B.leak.values())]
    out = _split(out, dist)
    out["leak"] = LK.Counter.rates(vals)
    return out


def _train_payload(rec: dict, *, step: int, **kw: Any) -> dict:
    out = B.originals["WL.train_payload"](rec, step=step, **kw)
    for k, v in (rec.get("aux_batch") or {}).items():
        if v is not None:
            out[f"aux/{k}"] = v
    for k, v in (rec.get("leak") or {}).items():
        if v is not None:
            out[f"leak/{k}"] = v
    return out


# ----------------------------------------------------------------------------- install / uninstall
def install(plan: BM.AuxBatchPlan, pool: list) -> BState:
    """Call AFTER patch.install(..., aux_ratio=0). ``pool`` = the overlap-excluded CoViP items (permuted here)."""
    global B
    from trainer_v2_v4 import mix as MX
    from trainer_v2_v4 import patch as PT
    if B is not None:
        raise RuntimeError("patch_b already installed")
    if PT.STATE is None or PT.STATE.schedule is not None:
        raise RuntimeError("install patch.install(aux_ratio=0) first (batch mixing replaces the step schedule)")
    if len(pool) != plan.n_pool:
        raise ValueError("plan.n_pool must equal the pool size")
    B = BState(plan=plan, pool=MX.permuted(pool, plan.seed))
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import wandb_layout as WL
    targets = {"MR.train_one_step": (MR, "train_one_step", _train_one_step),
               "DM.ddp_train_one_step": (DM, "ddp_train_one_step", _ddp_train_one_step),
               "RV.run_steps": (RV, "run_steps", _run_steps),
               "AL.dynamic_rollout": (AL, "dynamic_rollout", _dynamic_rollout),
               "AL.rl_weights": (AL, "rl_weights", _rl_weights),
               "G.rollout": (G, "rollout", _rollout), "G.train_step": (G, "train_step", _train_step),
               "WL.train_payload": (WL, "train_payload", _train_payload)}
    for key, (mod, name, fn) in targets.items():
        B.originals[key] = getattr(mod, name)
        setattr(mod, name, fn)
    B.originals["RV.STEP_EXTRA_KEYS"] = RV.STEP_EXTRA_KEYS
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("aux_batch", "leak")
    logger.info("patch_b installed: r=%s total=%d over %d steps, pool %d", plan.ratio, plan.total, plan.steps,
                plan.n_pool)
    return B


def uninstall() -> None:
    global B
    if B is None:
        return
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import wandb_layout as WL
    mods = {"G": G, "MR": MR, "DM": DM, "WL": WL, "RV": RV, "AL": AL}
    for key, fn in B.originals.items():
        m, name = key.split(".", 1)
        setattr(mods[m], name, fn)
    B = None
