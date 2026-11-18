"""Batch-level CoViP mixing (owner 2026-10-13, default for P1): CoViP prompts are added INSIDE each optimizer step on top
of the unchanged PReMind prompts of that step (same absent-share sampler, seed, step numbers, dev evals and checkpoints
as the PReMind-only run P3). One prompt = one GRPO group (G rollouts), so advantages stay per prompt.

Allocation (``AuxBatchPlan``): q = r / (1 - r) CoViP prompts per PReMind prompt (r = share of ALL prompts; r = 0.25 ->
q = 1/3). Total over the run T = round(q * S * P_global) (S = PReMind steps, P_global = PReMind prompts per step), so the
realised share T / (T + S P_global) = r within one prompt. Per-step counts: T distinct slots drawn without replacement
(seeded) out of S x M (M = ceil(2 q P_global) slots per step) -> count(s) = chosen slots in step s (random, reproducible
on resume, no state). Prompts: positions off(s) .. off(s) + count(s) - 1 (off = prefix sum) of a seeded permutation of
the overlap-excluded pool (no repeat until the pool is exhausted; T = 333 < 555 for the P1 recipe). Rank split: global
position p goes to rank p mod W, so every rank gets floor/ceil of count / W per step and the run totals differ by <= 1. Refills (DAPO dynamic sampling) of a dropped CoViP group come from the same permutation after the
reserved region, at positions T + (s W + rank) R + j (R = ceil(F M)); they may repeat pool items once T + ... > pool.

Loss: per-source token-level mean, combined by prompt share. With N = global episodes of the step, N_src the episodes
of a source and T_src its global model tokens, episode k of source src gets the RL weight w_k = N_src * T_k / T_src, so
that  L = (1/N) sum_k w_k l_k = sum_src (N_src / N) * (sum_{k in src} T_k l_k / T_src)
(l_k = the episode's token-mean surrogate [+ beta k3]; with one source this is exactly algo_v2.rl_weights: N T_k / T).
DAPO dynamic sampling: the all-equal-reward filter and the refill run per source (a CoViP group is replaced by a CoViP
prompt, a PReMind group by a PReMind prompt from the step's sampler stream), each with its own budget ceil(F p_src).
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
AUX, PM = "covip", "premind"


def source(item: Any) -> str:
    return AUX if bool(getattr(item, "row", {}).get("aux_source")) else PM


@dataclass(frozen=True)
class AuxBatchPlan:
    ratio: float
    steps: int
    p_global: int
    world: int
    seed: int
    n_pool: int
    factor: float = 3.0
    counts: tuple = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0 or self.steps < 1 or self.p_global < 1 or self.world < 1 or self.n_pool < 1:
            raise ValueError("batch mixing needs 0 < r < 1, steps, prompts, world and pool >= 1")
        m = self.slots_per_step
        rng = random.Random(f"covip_batch_slots:{self.seed}")
        chosen = rng.sample(range(self.steps * m), self.total)
        c = [0] * self.steps
        for x in chosen:
            c[x // m] += 1
        object.__setattr__(self, "counts", tuple(c))

    @property
    def q(self) -> Fraction:
        r = Fraction(self.ratio).limit_denominator(1000)
        return r / (1 - r)

    @property
    def total(self) -> int:
        return int(round(self.q * self.steps * self.p_global))

    @property
    def slots_per_step(self) -> int:
        return max(1, math.ceil(2 * self.q * self.p_global))

    def count(self, step: int) -> int:
        return self.counts[step] if 0 <= step < self.steps else 0

    def offset(self, step: int) -> int:
        return sum(self.counts[:step])

    def positions(self, step: int, rank: int) -> list[int]:
        """Pool-permutation positions of this rank's CoViP prompts at ``step``: global position p goes to rank p mod W,
        so per step and over the run the ranks differ by at most one prompt."""
        off, c = self.offset(step), self.count(step)
        mine = [off + j for j in range(c) if (off + j) % self.world == rank]
        return [p % self.n_pool for p in mine]

    def refill_positions(self, step: int, rank: int, start: int, k: int) -> list[int]:
        r = math.ceil(self.factor * self.slots_per_step)
        base = self.total + (step * self.world + rank) * r
        return [(base + start + j) % self.n_pool for j in range(k)]


# ----------------------------------------------------------------------------- per-source DAPO pieces
def rl_weights_by_source(eps: list, cfg: Any, totals: Optional[tuple[int, int]] = None,
                         dist: Any = None) -> Optional[list[float]]:
    """DAPO token-level weights per source, combined by prompt share (module docstring); None for GRPO / GSPO.
    ``dist`` (ddp_v2.DistCtx): per-source global sums (every rank calls this at the same point of train_step)."""
    if cfg.algo != "dapo":
        return None
    toks = [sum(ep.env_mask) for ep in eps]
    srcs = [source(ep.item) for ep in eps]
    loc = [sum(1 for s in srcs if s == PM), sum(t for t, s in zip(toks, srcs) if s == PM),
           sum(1 for s in srcs if s == AUX), sum(t for t, s in zip(toks, srcs) if s == AUX)]
    glob = loc if dist is None else [int(v) for v in dist.all_sum(loc)]
    n_t = {PM: (glob[0], glob[1]), AUX: (glob[2], glob[3])}
    return [n_t[s][0] * t / n_t[s][1] if n_t[s][1] else 0.0 for t, s in zip(toks, srcs)]


def dynamic_rollout_by_source(items: list, images: list, backend: Any, cfg: Any,
                              refills: dict[str, Optional[Callable[[int], tuple[list, list]]]],
                              rollout: Callable[..., list], score: Callable[[Any, Any], dict],
                              informative: Callable[[list], bool], shaped: Callable[[list, Any], list],
                              min_rate: float = 0.25) -> tuple[list, list, list, dict]:
    """algo_v2.dynamic_rollout with the filter, the budget and the refill per source. Returns (kept episodes, items,
    images, info) with the kept groups in draw order; a source with no informative group contributes nothing unless
    no source has one (then the first round is used, as in algo_v2)."""
    g = cfg.num_generations
    want = {s: sum(1 for it in items if source(it) == s) for s in (PM, AUX)}
    budget = {s: math.ceil(cfg.dapo_max_prompts_factor * want[s]) for s in want}
    drawn, seen, filtered = dict(want), {PM: 0, AUX: 0}, {PM: 0, AUX: 0}
    kept: dict[str, list] = {PM: [], AUX: []}
    first: list = []
    rounds = 0
    b_items, b_images = items, images
    while True:
        eps = rollout(b_items, b_images, backend, cfg, batched=cfg.batch_rollout)
        for ep in eps:
            ep.ledger = score(ep, cfg)
        sh = shaped(eps, cfg)
        for j, (it, im) in enumerate(zip(b_items, b_images, strict=True)):
            grp, s = (it, im, eps[j * g:(j + 1) * g]), source(it)
            if rounds == 0:
                first.append(grp)
            seen[s] += 1
            if informative(sh[j * g:(j + 1) * g]):
                kept[s].append(grp)
            else:
                filtered[s] += 1
        more_items: list = []
        more_images: list = []
        for s in (PM, AUX):
            need = want[s] - len(kept[s])
            if need <= 0 or refills.get(s) is None or drawn[s] >= budget[s]:
                continue
            rate = max(len(kept[s]) / seen[s], min_rate) if seen[s] else min_rate
            k = min(budget[s] - drawn[s], math.ceil(need / rate))
            got_i, got_m = refills[s](k)
            more_items += list(got_i)
            more_images += list(got_m)
            drawn[s] += k
        if not more_items:
            break
        b_items, b_images = more_items, more_images
        rounds += 1
    use = [grp for s in (PM, AUX) for grp in kept[s][:want[s]]]
    if not use:
        use = first
    info = {"dapo_filtered_groups": filtered[PM] + filtered[AUX], "dapo_resample_rounds": rounds,
            "dapo_prompts_drawn": drawn[PM] + drawn[AUX], "dapo_kept_groups": sum(min(len(kept[s]), want[s]) for s in want),
            "dapo_no_informative": not (kept[PM] or kept[AUX]),
            "by_source": {s: {"wanted": want[s], "drawn": drawn[s], "filtered": filtered[s],
                              "kept": min(len(kept[s]), want[s]), "seen": seen[s]} for s in (PM, AUX)}}
    logger.info("DAPO dynamic sampling (per source): %s", info)
    return [ep for _, _, grp in use for ep in grp], [it for it, _, _ in use], [im for _, im, _ in use], info
