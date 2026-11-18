"""In-process hooks of the v4h options (training v2), installed by ``main_v4h`` BEFORE main_v4 / main_v4b install
their own hooks on top (patch.py's score hook delegates non-reward_v4 presets, i.e. v4wh, to the function below it).
New file: patch.py / patch_b.py / main_v4*.py and every frozen trainer_v2 / eval_v1 file are untouched.

  grpo_v2.score        v4wh presets -> reward_v4h (hazard term on acute rows); any other preset -> the original, and
                       with ``monitor`` on the cue classification is recorded (reward unchanged).
  grpo_v2.rollout      training only, backend "judge", v4wh preset: one async judge batch for the round's acute
                       episodes (hazard_judge.prefetch) before they are scored; failures -> cue path.
  grpo_v2.train_step   per-step hazard counters (all_sum over ranks) -> record key ``hazard``; rollout text log.
  main_run.train_one_step / ddp_main.ddp_train_one_step   record the global step and OUT dir of the step.
  wandb_layout.train_payload   + hazard/* scalars.   run_v2.STEP_EXTRA_KEYS   + "hazard" (only when active).
Rollout text log (``log_texts``; ON in the v2 launcher, OFF elsewhere): OUT/rollout_texts.jsonl, every acute-row
training rollout + a deterministic sha256-keyed ``text_frac`` (5 %) of the others (no RNG is consumed): step,
row_id, arm, sample_idx, acute, answer, total, components, hazard (state, backend, cue, judge reply).
With every option off the hooks return the original results unchanged (tests: bit-identical weights / records).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from trainer_v2 import grpo_v2 as G
from trainer_v2 import run_v2 as RV
from trainer_v2_v4 import covip_aux as CA
from trainer_v2_v4 import hazard_judge as HJ
from trainer_v2_v4 import hazard_monitor as HMN
from trainer_v2_v4 import hazard_rubric as HR
from trainer_v2_v4 import key_actions_v4h as KA
from trainer_v2_v4 import reward_v4h as RH
from trainer_v2_v4 import reward_v4hg as RHG

logger = logging.getLogger(__name__)
TEXT_LOG = "rollout_texts.jsonl"


@dataclass
class HState:
    key_actions: dict = field(default_factory=dict)        # row_id -> RowActions
    backend: str = "cue"                                   # "cue" | "judge"
    judge: Any = None                                      # HazardJudge (backend "judge")
    monitor: bool = False
    log_texts: bool = False
    text_frac: float = 0.05
    in_train: bool = False
    step: Optional[int] = None
    out_dir: Optional[Path] = None
    counter: HMN.Counter = field(default_factory=HMN.Counter)
    judge_stats: dict = field(default_factory=dict)
    texts: list = field(default_factory=list)
    group_idx: dict = field(default_factory=dict)
    missing: set = field(default_factory=set)
    acute_lens: list = field(default_factory=list)         # <answer> word counts of this step's acute rollouts
    gold_len_median: Optional[float] = None                 # train-gold acute <answer> median (verbosity reference)
    originals: dict = field(default_factory=dict)


H: Optional[HState] = None


def _acute_row(row: dict) -> Optional[dict]:
    from eval_v1 import acute_rule_v4 as AR                # read-only
    if not AR.is_acute_row(row):
        return None
    a = row.get("acute_visual_hazard")
    if a is None and isinstance(row.get("evaluator"), dict):
        a = row["evaluator"].get("acute_visual_hazard")
    return a


def hz_input(item: Any, judge: Optional[dict], backend: str) -> Optional[dict]:
    """reward_v4h ``hz`` of one training item (None for aux / non-acute rows)."""
    if CA.is_aux(item):
        return None
    a = _acute_row(item.row)
    if a is None:
        return None
    what = (a.get("image_verdict") or {}).get("what") or ""
    ra = H.key_actions.get(item.row_id) if H is not None else None
    if ra is None:
        if H is not None:
            H.missing.add(item.row_id)
        ra = KA.type_default(item.row_id, what)
    if what and ra.what != what:                       # the eval's visible hazard (image_verdict.what) wins
        ra = dataclasses.replace(ra, what=what)
    return {"acute": True, "ra": ra, "judge": judge, "backend": backend}


def _active(cfg: Any) -> bool:
    return RHG.is_hazard_preset(cfg.reward_preset)


def _sample(key: str, frac: float) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 16 ** 8 < frac


def _record(ep: Any, led: dict, info: Optional[dict]) -> None:
    if not H.in_train or getattr(ep, "_v4h_seen", False) or CA.is_aux(ep.item):
        return
    ep._v4h_seen = True
    comp = led["components"].get("hazard_appropriate") or {}
    val = comp.get("value") if comp.get("applicable") else (
        None if info is None else RH.state_value(info["state"], info["contraindicated"], RH.PRESETS["v4wh"]))
    H.counter.add(info, val, led.get("guard"))
    if info is not None:
        H.acute_lens.append(len(HR.answer_text(ep.stage2_text).split()))
    k = H.group_idx.get(id(ep.item), 0)
    H.group_idx[id(ep.item)] = k + 1
    if H.log_texts and (info is not None or _sample(f"{H.step}|{ep.item.row_id}|{k}", H.text_frac)):
        H.texts.append({"step": H.step, "row_id": ep.item.row_id, "arm": ep.item.arm, "sample_idx": k,
                        "acute": info is not None, "answer": HR.answer_text(ep.stage2_text), "total": led["total"],
                        "components": {n: c["value"] for n, c in led["components"].items() if c["applicable"]},
                        "hazard": None if info is None else {
                            **{x: info[x] for x in ("state", "backend", "contraindicated", "ordered", "cue", "judge")},
                            "judge_raw": getattr(ep, "hazard_judge_raw", None)}})


# ----------------------------------------------------------------------------- hooks
def _score(ep: Any, cfg: Any) -> dict:
    if CA.is_aux(ep.item) or not (_active(cfg) or H.monitor or H.log_texts):
        return H.originals["G.score"](ep, cfg)
    if _active(cfg):
        hz = hz_input(ep.item, getattr(ep, "hazard_judge", None), H.backend)
        led = RHG.score_episode(ep, cfg, hz)
        info = led["v4h"] if led["v4h"].get("acute") and "state" in led["v4h"] else None
    else:
        led = H.originals["G.score"](ep, cfg)
        info = RH.hazard_info(hz_input(ep.item, None, "cue"), ep.stage2_text, ep.item.target.get("hazard_pattern"))
    _record(ep, led, info)
    return led


def _rollout(items: list, images: list, backend: Any, cfg: Any, **kw: Any) -> list:
    eps = H.originals["G.rollout"](items, images, backend, cfg, **kw)
    if H.in_train and H.backend == "judge" and H.judge is not None and _active(cfg):
        jobs = []
        for ep in eps:
            hz = hz_input(ep.item, None, "judge")
            if hz is not None:
                jobs.append((ep, RHG.judge_prompt(cfg.reward_preset, hz["ra"].what, hz["ra"].reference(),
                                                  HR.answer_text(ep.stage2_text),
                                                 types=list(hz["ra"].types), request=ep.item.row.get("query"))))
        st = HJ.prefetch(jobs, H.judge)
        for k, v in st.items():
            H.judge_stats[k] = H.judge_stats.get(k, 0) + v
    return eps


def _train_step(*a: Any, **kw: Any) -> dict:
    H.in_train, H.counter, H.texts, H.group_idx, H.judge_stats = True, HMN.Counter(), [], {}, {}
    H.acute_lens = []
    try:
        out = H.originals["G.train_step"](*a, **kw)
    finally:
        H.in_train = False
    cfg = a[5] if len(a) > 5 else kw["cfg"]
    if not (_active(cfg) or H.monitor or H.log_texts):
        return out
    dist = kw.get("dist")
    vals = H.counter.values()
    js = [H.judge_stats.get(k, 0) for k in ("calls", "failed", "seconds")]
    if dist is not None:
        vals, js = dist.all_sum(vals), dist.all_sum(js)
    out["hazard"] = {**HMN.Counter.rates(vals), **({"judge_calls": int(js[0]), "judge_failed": int(js[1]),
                                                   "judge_seconds": js[2]} if H.backend == "judge" else {})}
    if hasattr(H.judge, "stats"):                       # API backend: run spend (all ranks) and cap state
        st = H.judge.stats()
        api = [st["usd_spent"], st["tokens_spent"], float(st["capped"]), st["api_failed"]]
        if dist is not None:
            api = dist.all_sum(api)
        out["hazard"].update(judge_usd_total=round(api[0], 6), judge_tokens_total=int(api[1]),
                             judge_capped_ranks=int(api[2]), judge_api_failed_total=int(api[3]))
    lens = H.acute_lens if dist is None else [x for part in dist.gather(H.acute_lens) for x in part]
    out["hazard"].update(HMN.verbosity(lens, H.gold_len_median))
    if H.missing:
        out["hazard"]["missing_key_actions"] = len(H.missing)
    if H.log_texts:
        rows = H.texts if dist is None else [r for part in dist.gather(H.texts) for r in part]
        if (dist is None or dist.is_main) and H.out_dir is not None and rows:
            with (Path(H.out_dir) / TEXT_LOG).open("a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return out


def _train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, *rest, **kw):
    H.step, H.out_dir = step, Path(out)
    return H.originals["MR.train_one_step"](items, step, args, backend, optimizer, params, cfg, opsd, monitor, out,
                                            *rest, **kw)


def _ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, *rest, **kw):
    H.step, H.out_dir = step, Path(out)
    return H.originals["DM.ddp_train_one_step"](items, step, args, backend, optimizer, params, cfg, opsd, monitor,
                                                out, *rest, **kw)


def _train_payload(rec: dict, *, step: int, **kw: Any) -> dict:
    out = H.originals["WL.train_payload"](rec, step=step, **kw)
    out.update(HMN.payload(rec))
    return out


# ----------------------------------------------------------------------------- install / uninstall
def install(*, key_actions: Optional[dict] = None, backend: str = "cue", judge: Any = None, monitor: bool = False,
            log_texts: bool = False, text_frac: float = 0.05, register_presets: bool = True,
            gold_len_median: Optional[float] = None) -> HState:
    global H
    if H is not None:
        raise RuntimeError("trainer_v2_v4.patch_h already installed")
    if backend not in ("cue", "judge"):
        raise ValueError("backend must be 'cue' or 'judge'")
    if backend == "judge" and judge is None:
        raise ValueError("backend 'judge' needs a HazardJudge")
    if register_presets:
        RHG.register()
    H = HState(key_actions=dict(key_actions or {}), backend=backend, judge=judge, monitor=monitor,
               log_texts=log_texts, text_frac=text_frac, gold_len_median=gold_len_median)
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import wandb_layout as WL
    targets = {"G.score": (G, "score", _score), "G.rollout": (G, "rollout", _rollout),
               "G.train_step": (G, "train_step", _train_step),
               "MR.train_one_step": (MR, "train_one_step", _train_one_step),
               "DM.ddp_train_one_step": (DM, "ddp_train_one_step", _ddp_train_one_step),
               "WL.train_payload": (WL, "train_payload", _train_payload)}
    for key, (mod, name, fn) in targets.items():
        H.originals[key] = getattr(mod, name)
        setattr(mod, name, fn)
    H.originals["RV.STEP_EXTRA_KEYS"] = RV.STEP_EXTRA_KEYS
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("hazard",)  # absent from records unless an option is on
    logger.info("trainer_v2_v4.patch_h installed: backend=%s monitor=%s log_texts=%s key_actions=%d", backend,
                monitor, log_texts, len(H.key_actions))
    return H


def uninstall() -> None:
    global H
    if H is None:
        return
    from eval_v1 import ddp_main as DM
    from eval_v1 import main_run as MR
    from eval_v1 import wandb_layout as WL
    mods = {"G": G, "MR": MR, "DM": DM, "WL": WL, "RV": RV}
    for key, fn in H.originals.items():
        m, name = key.split(".", 1)
        setattr(mods[m], name, fn)
    H = None


__all__ = ["install", "uninstall", "hz_input", "TEXT_LOG", "HState"]
