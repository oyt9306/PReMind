"""Phase II-quadruple-prime per-row advantage weight (owner 2026-10-24): must-learn conflict rows get a larger learning
signal on replay, late rows a smaller one. Reward magnitudes are untouched (REWARD_BREAKEVEN_v2 stays valid): only the
group-normalised advantage is scaled. Same mechanism as Phase III's replay down-weighting (train_mie_p3_4b/replay_p3.py
install_weight): ``AL.shaped_rewards`` is wrapped to remember the list it returned and the episodes, and
``G.grouped_advantages`` (called by grpo_v2.train_step on exactly that list; identity checked) multiplies each episode's
advantage by its row's weight. The DAPO clipped surrogate is positively homogeneous in the advantage, so this is a
per-group weight on the policy-gradient term (KL untouched).

Weights (row tier from TIERS = curated tiers: v1 CURATION_conflict_train_v1_claude.json + the owner's hca1 curation):
  core  1.0 on the row's first (fresh) visit, 1.5 on a replay visit (stillpen sampler cf_src src == "replay")
  late  0.5 on every visit
  every other row (non-conflict, uncurated) 1.0 (unchanged)
Cap: per rank and step, if sum|A w| > CAP x sum|A| the up-weights (w > 1) are shrunk towards 1 by the factor k that
makes the ratio exactly CAP (down-weights are kept). Logged per step (all ranks): ``p2e_advweight`` = groups
[row_id, tier, visit, w_nominal, w_applied], the realised factor sum|A w| / sum|A| and k.
"""
from __future__ import annotations

from typing import Any, Optional

CAP = 1.25
W_CORE_FIRST, W_CORE_REPLAY, W_LATE = 1.0, 1.5, 0.5
_W: dict = {"orig": {}, "last": None, "tiers": {}, "log": [], "stats": []}


def nominal(tier: Optional[str], visit: Optional[str]) -> float:
    if tier == "core":
        return W_CORE_REPLAY if visit == "replay" else W_CORE_FIRST
    if tier in ("late", "learned"):                 # learned: 8/8 rows kept only if < 40 rows remain (owner)
        return W_LATE
    return 1.0


def capped(absadv: list, w: list, cap: float = CAP) -> tuple[list, float, float]:
    """(applied weights, realised factor, k): up-weights shrunk by k so that sum|A| w <= cap sum|A|."""
    s = sum(absadv)
    if s <= 0:
        return list(w), 1.0, 1.0
    base = sum(a * min(x, 1.0) for a, x in zip(absadv, w))
    up = sum(a * (x - 1.0) for a, x in zip(absadv, w) if x > 1.0)
    k = 1.0 if up <= 0 or base + up <= cap * s else max(0.0, (cap * s - base) / up)
    out = [x if x <= 1.0 else 1.0 + k * (x - 1.0) for x in w]
    return out, sum(a * x for a, x in zip(absadv, out)) / s, k


def _visit(row_id: str) -> Optional[str]:
    import stillpen as SP
    s = SP.ACTIVE["sampler"]
    v = (getattr(s, "cf_src", {}) or {}).get(row_id) if s is not None else None
    return v["src"] if v else None


def install(tiers: dict) -> None:
    import torch

    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    if _W["orig"]:
        raise RuntimeError("advweight_p2e installed twice")
    sr, ga, ts = AL.shaped_rewards, G.grouped_advantages, G.train_step
    _W["orig"] = {"sr": sr, "ga": ga, "ts": ts, "keys": RV.STEP_EXTRA_KEYS}
    _W["tiers"] = dict(tiers)

    def shaped_rewards(eps: list, cfg: Any) -> list:
        out = sr(eps, cfg)
        _W["last"] = (out, list(eps))
        return out

    def grouped_advantages(rewards: Any, g: int, scale_rewards: str = "group") -> Any:
        adv = ga(rewards, g, scale_rewards)
        last = _W["last"]
        if last is None or rewards is not last[0]:
            raise RuntimeError("advweight_p2e: grouped_advantages called without the matching shaped_rewards list")
        eps = last[1]
        if len(eps) != adv.numel():
            raise RuntimeError("advweight_p2e: advantage / episode count mismatch")
        rows = [ep.item.row_id for ep in eps]
        tv = [(_W["tiers"].get(r), _visit(r) if _W["tiers"].get(r) else None) for r in rows]
        w = [nominal(t, v) for t, v in tv]
        if all(x == 1.0 for x in w):
            _W["stats"].append({"factor": 1.0, "k": 1.0})
            return adv
        absadv = [abs(float(a)) for a in adv.reshape(-1).tolist()]
        wa, factor, k = capped(absadv, w)
        for i in range(0, len(rows), g):
            if w[i] != 1.0:
                _W["log"].append([rows[i], tv[i][0], tv[i][1], w[i], round(wa[i], 6)])
        _W["stats"].append({"factor": round(factor, 6), "k": round(k, 6)})
        return adv * torch.tensor(wa, dtype=adv.dtype).reshape(adv.shape)

    def train_step(*a, **kw):
        _W["log"], _W["stats"] = [], []
        out = ts(*a, **kw)
        dist = kw.get("dist")
        logs = _W["log"] if dist is None else [x for p in dist.gather(_W["log"]) for x in p]
        stats = _W["stats"] if dist is None else [x for p in dist.gather(_W["stats"]) for x in p]
        out["p2e_advweight"] = {"groups": logs, "per_rank_call": stats, "cap": CAP,
                                "weights": {"core_first": W_CORE_FIRST, "core_replay": W_CORE_REPLAY, "late": W_LATE}}
        return out
    AL.shaped_rewards, G.grouped_advantages, G.train_step = shaped_rewards, grouped_advantages, train_step
    RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("p2e_advweight",)


def uninstall() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import run_v2 as RV
    o = _W["orig"]
    if o:
        AL.shaped_rewards, G.grouped_advantages, G.train_step, RV.STEP_EXTRA_KEYS = o["sr"], o["ga"], o["ts"], o["keys"]
    _W.update(orig={}, last=None, tiers={}, log=[], stats=[])


__all__ = ["CAP", "W_CORE_FIRST", "W_CORE_REPLAY", "W_LATE", "nominal", "capped", "install", "uninstall"]
