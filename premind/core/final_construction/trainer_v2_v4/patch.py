"""In-process hooks that add the v4 recipe pieces to the frozen trainer_v2 / eval_v1 loop WITHOUT editing them.

Why hooks and not edits: every existing PREREG pins the sha256 of trainer_v2/{reward_v5,grpo_v2,train_mb,...}.py and
live eval / resume jobs verify those pins at start-up, so any edit there would break them. ``install`` rebinds module
attributes in the CURRENT process only (the v4 entry point ``main_v4`` calls it; nothing else imports this module):

  grpo_v2.score             CoViP items -> covip_aux.score_aux; v4 presets -> reward_v4; else the original.
  grpo_v2.rollout           training only (inside train_step) and kl_beta > 0: attach ref log-probs (adapter off).
  grpo_v2.loss_from_logits  kl_beta > 0: + beta * mean k3 (weighted like the RL term); stats["ref_kl"].
  grpo_v2.train_step        marks "in training" for the rollout hook; adds ref_kl_mean to the step result.
  run_v2.EPISODE_KEYS / STEP_EXTRA_KEYS   + ref_kl / ref_kl_mean (logging).
  eval_v1.main_run.train_one_step, eval_v1.ddp_main.ddp_train_one_step   the aux steps due before PReMind step p
                            (mix.AuxSchedule.due) run the SAME function on CoViP prompts first (no OPSD, no monitor,
                            records in out/aux/steps.jsonl keyed by aux index); then PReMind step p runs unchanged and
                            its record carries the aux summaries. The step counter counts PReMind steps (aux-only: aux).
  eval_v1.wandb_layout.train_payload      aux summaries -> aux/*; PReMind steps also get train/policy/ref_kl.
  eval_v1.release_data.load_release       monitor slice on (owner 2026-10-13, no dev split): split "train" drops the
                            held-out monitor slice (monitor_slice.py); split "dev" (the in-loop dev eval, i.e. the
                            over-abstention monitor + confirmed stop rule) returns that slice from the TRAIN file.
With kl_beta = 0, no aux and a reward_v5 preset every hook returns the original result unchanged (tests prove the
updated weights are bit-identical to the unhooked loop). ``uninstall`` restores the originals.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from trainer_v2 import grpo_v2 as G
from trainer_v2 import run_v2 as RV
from trainer_v2_v4 import covip_aux as CA
from trainer_v2_v4 import kl_ref as KL
from trainer_v2_v4 import mix as MX
from trainer_v2_v4 import monitor_slice as MS
from trainer_v2_v4 import reward_v4 as RW

logger = logging.getLogger(__name__)


@dataclass
class V4State:
    kl_beta: float = 0.0
    aux_items: list = field(default_factory=list)       # permuted CoViP items
    schedule: Optional[MX.AuxSchedule] = None
    p_aux: int = 0                                      # per rank
    in_train: bool = False
    monitor: Optional[tuple[float, int]] = None        # (frac, seed) of the held-out monitor slice
    originals: dict = field(default_factory=dict)


STATE: Optional[V4State] = None


# ----------------------------------------------------------------------------- trainer hooks
def _score(ep: Any, cfg: Any) -> dict:
    if CA.is_aux(ep.item):
        return CA.score_aux(ep.item, ep.stage2_text, ep.n_stage2)
    if cfg.reward_preset in RW.PRESETS:
        return RW.score_episode(ep, cfg)
    return STATE.originals["G.score"](ep, cfg)


def _rollout(items: list, images: list, backend: Any, cfg: Any, **kw: Any) -> list:
    eps = STATE.originals["G.rollout"](items, images, backend, cfg, **kw)
    if STATE.kl_beta > 0 and STATE.in_train and not kw.get("greedy") and not kw.get("teacher"):
        by_item = {id(it): im for it, im in zip(items, images)}
        for ep in eps:
            ep.ref_logp = KL.ref_logp(ep, backend, by_item[id(ep.item)])
    return eps


def _loss_from_logits(ep: Any, adv: float, logits: Any, cfg: Any, teacher: Any = None, old_logp: Any = None,
                      rl_weight: Optional[float] = None) -> tuple[Any, dict, Any]:
    loss, stats, old = STATE.originals["G.loss_from_logits"](ep, adv, logits, cfg, teacher, old_logp, rl_weight)
    if STATE.kl_beta > 0:
        ref = getattr(ep, "ref_logp", None)
        if ref is None:
            raise RuntimeError("kl_beta > 0 but the episode has no reference log-probs (rollout hook not active)")
        kl, val = KL.kl_term(ep, logits, ref)
        loss = loss + STATE.kl_beta * kl * KL.weighted(rl_weight)
        stats = {**stats, "ref_kl": val}
    return loss, stats, old


def _train_step(*a: Any, **kw: Any) -> dict:
    STATE.in_train = True
    try:
        out = STATE.originals["G.train_step"](*a, **kw)
    finally:
        STATE.in_train = False
    if STATE.kl_beta > 0:
        vals = [e["ref_kl"] for e in out["episodes"] if e.get("ref_kl") is not None]
        out["ref_kl_mean"] = sum(vals) / len(vals) if vals else None
    return out


# ----------------------------------------------------------------------------- step hooks (aux mixing)
AUX_KEYS = ("reward_mean", "grad_norm", "t_rollout_s", "t_train_s", "dapo_filtered_groups", "dapo_kept_groups",
            "ref_kl_mean")


def _aux_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), "prompts_per_step": STATE.p_aux})


def _aux_items(aux_index: int) -> list:
    return MX.rotated(STATE.aux_items, STATE.schedule.offset(aux_index, len(STATE.aux_items)))


def _summary(rec: dict, aux_index: int) -> dict:
    srcs = sorted({CA.SOURCE[m] for m in {r.split(":")[1] for r in rec.get("rows") or []}})
    eps = rec.get("episodes") or []
    return {"aux_index": aux_index, "aux_source": ",".join(srcs) or "covip",
            **{k: rec.get(k) for k in AUX_KEYS if rec.get(k) is not None},
            **({"success_rate": sum(bool(e.get("success")) for e in eps) / len(eps)} if eps else {})}


def _aux_out(out: Path, main: bool = True) -> Path:
    p = Path(out) / "aux"
    if main:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _run_aux_single(i, args, backend, optimizer, params, cfg, out) -> dict:
    """One aux optimizer step (single GPU): the frozen train_one_step on the CoViP window of aux index ``i`` (no OPSD,
    no monitor, records in OUT/aux/steps.jsonl keyed by the aux index)."""
    from trainer_v2 import vcm_data as VD
    real = VD.item_images
    try:
        VD.item_images = CA.item_images                       # train_one_step passes VD.item_images to run_steps
        rec = STATE.originals["MR.train_one_step"](_aux_items(i), i, _aux_args(args), backend, optimizer, params, cfg,
                                                   None, None, _aux_out(out), None, 0, vef=None)
    finally:
        VD.item_images = real
    return _summary(rec, i)


def _train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, sampler=None,
                    cursor=None, vef=None):
    orig = STATE.originals["MR.train_one_step"]
    sched = STATE.schedule
    due = [] if sched is None else sched.due(step)
    aux = [_run_aux_single(i, args, backend, optimizer, params, cfg, out) for i in due]
    if sched is not None and sched.aux_only:                  # ablation (c): this step IS the aux step
        return {**aux[0], "step": step, "aux_only": True, "n_prompts_drawn": 0}
    rec = orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, sampler, cursor, vef=vef)
    if aux:
        rec["aux"] = aux
    return rec


def _ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler,
                        cursor, loader=None, cuda=True, vef=None):
    orig = STATE.originals["DM.ddp_train_one_step"]
    from trainer_v2 import vcm_data as VD
    sched = STATE.schedule
    aux = []
    for i in ([] if sched is None else sched.due(step)):
        r, _ = orig(_aux_items(i), i, _aux_args(args), backend, optimizer, params, cfg, None, None,
                    _aux_out(out, dist.is_main), dist, None, 0, loader=CA.item_images, cuda=cuda, vef=None)
        aux.append(_summary(r, i))
    if sched is not None and sched.aux_only:
        return {**aux[0], "step": step, "aux_only": True}, 0
    rec, used = orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler, cursor,
                     loader=loader or VD.item_images, cuda=cuda, vef=vef)
    if aux:
        rec["aux"] = aux
    return rec, used


def _train_payload(rec: dict, *, step: int, **kw: Any) -> dict:
    if rec.get("aux_only"):
        return {"train/step": step, **{f"aux/{k}": v for k, v in rec.items() if k in AUX_KEYS + ("success_rate",)}}
    out = STATE.originals["WL.train_payload"](rec, step=step, **kw)
    if rec.get("ref_kl_mean") is not None:
        out["train/policy/ref_kl"] = rec["ref_kl_mean"]
    for a in rec.get("aux") or []:
        out.update({f"aux/{k}": v for k, v in a.items() if k in AUX_KEYS + ("success_rate", "aux_index")})
    return out


def _load_release(release, split, mode, **kw):
    orig = STATE.originals["D.load_release"]
    if STATE.monitor is None or split not in ("train", "dev"):
        return orig(release, split, mode, **kw)
    ids, info = MS.slice_ids(Path(release), mode, *STATE.monitor)
    if split == "train":
        items, man = orig(release, "train", mode, **kw)
        kept = [it for it in items if it.row_id not in ids]
        return kept, {**man, "n_items": len(kept), "monitor_slice": {**info, "n_items_held_out": len(items) - len(kept)}}
    limit = kw.pop("limit", None)
    arms = kw.get("arms") or ("passive", "memory_call")
    items, man = orig(release, "train", mode, **kw)
    mon = [it for it in items if it.row_id in ids]
    if limit is not None:
        mon = mon[:limit * len(arms)]
    return mon, {**man, "split": "monitor_slice(train)", "n_items": len(mon), "monitor_slice": info}


# ----------------------------------------------------------------------------- install / uninstall
def install(*, kl_beta: float = 0.0, aux_items: Optional[list] = None, aux_ratio: float = 0.0, p_aux: int = 0,
            world: int = 1, dapo_factor: float = 3.0, seed: int = 0, register_presets: bool = True,
            monitor: Optional[tuple[float, int]] = None) -> V4State:
    """Install the hooks. ``register_presets`` adds the v4 preset names to reward_v5.PRESETS (needed by main_run's
    CLI and train_step's preset lookup in a v4 run; the hooks-off legacy test plugin passes False)."""
    global STATE
    if STATE is not None:
        raise RuntimeError("trainer_v2_v4.patch already installed")
    KL.validate(kl_beta)
    if register_presets:
        RW.register()
    sched = None
    if aux_ratio > 0:
        if not aux_items or p_aux < 1:
            raise ValueError("aux_ratio > 0 needs aux items and --aux-prompts-per-step >= 1")
        sched = MX.make_schedule(aux_ratio, p_aux, world, dapo_factor)
    STATE = V4State(kl_beta=kl_beta, aux_items=MX.permuted(aux_items or [], seed), schedule=sched, p_aux=p_aux,
                    monitor=monitor)
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import release_data as D
    from eval_v1 import wandb_layout as WL
    targets = {"G.score": (G, "score", _score), "G.rollout": (G, "rollout", _rollout),
               "G.loss_from_logits": (G, "loss_from_logits", _loss_from_logits),
               "G.train_step": (G, "train_step", _train_step),
               "MR.train_one_step": (MR, "train_one_step", _train_one_step),
               "DM.ddp_train_one_step": (DM, "ddp_train_one_step", _ddp_train_one_step),
               "WL.train_payload": (WL, "train_payload", _train_payload)}
    if monitor is not None:
        targets["D.load_release"] = (D, "load_release", _load_release)
    for key, (mod, name, fn) in targets.items():
        STATE.originals[key] = getattr(mod, name)
        setattr(mod, name, fn)
    if kl_beta > 0:                     # logging keys only when the KL term is on (records unchanged otherwise)
        STATE.originals["RV.EPISODE_KEYS"], STATE.originals["RV.STEP_EXTRA_KEYS"] = RV.EPISODE_KEYS, RV.STEP_EXTRA_KEYS
        RV.EPISODE_KEYS = RV.EPISODE_KEYS + ("ref_kl",)
        RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("ref_kl_mean",)
    logger.info("trainer_v2_v4 installed: kl_beta=%s aux_ratio=%s p_aux=%s n_aux=%d monitor=%s", kl_beta, aux_ratio,
                p_aux, len(STATE.aux_items), monitor)
    return STATE


def uninstall() -> None:
    global STATE
    if STATE is None:
        return
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import release_data as D
    from eval_v1 import wandb_layout as WL
    mods = {"G": G, "MR": MR, "DM": DM, "WL": WL, "RV": RV, "D": D}
    for key, fn in STATE.originals.items():
        m, name = key.split(".", 1)
        setattr(mods[m], name, fn)
    STATE = None
