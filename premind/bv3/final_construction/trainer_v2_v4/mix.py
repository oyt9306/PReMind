"""Step-level mixing of the CoViP auxiliary task (WP-B item 3).

Ratio ``r`` = the fraction of OPTIMIZER STEPS that are auxiliary steps. Every optimizer step trains on ONE source:
either the PReMind prompts of the absent-share sampler (unchanged) or ``P_aux`` CoViP prompts, G samples each. Whole
steps (not prompts inside a step) are mixed, so every PReMind step record, monitor window, stop rule, DAPO token
normalisation and wandb curve is exactly that of a PReMind-only run; advantages are group-normalised either way (one
prompt = one group), so the two reward scales never meet.

Interleaved schedule (0 < r < 1): the trainer's step counter counts PReMind steps only (``--steps`` = PReMind steps, so
dev evals / checkpoints / the stop rule land on PReMind step numbers, e.g. 150 / 200 / 250 for every r). With
q = r / (1 - r) aux steps per PReMind step (exact fraction: r = 0.25 -> q = 1/3), the aux steps due right BEFORE
PReMind step p are k = floor(q (p + 1)) - floor(q p), with global aux indices floor(q p) .. floor(q (p + 1)) - 1.
Deterministic, no RNG state: a resume reproduces it. Total optimizer steps = P + floor(q P).
Aux-only (r = 1, ablation c): every step is an aux step and the step counter counts aux steps.
Aux prompts: a seeded permutation of the aux pool; aux step i reads a window starting at i * ceil(F * P_aux_global)
(F = DAPO max prompts factor, so refills never overlap the next window), wrapping around (epochs).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


def _frac(ratio: float) -> Fraction:
    return Fraction(ratio).limit_denominator(1000)


@dataclass(frozen=True)
class AuxSchedule:
    ratio: float
    prompts_global: int          # aux prompts per aux step over all ranks
    window: int                  # stream positions reserved per aux step (>= prompts_global)

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("aux ratio must be in [0, 1] (1 = aux-only run, ablation (c))")
        if self.prompts_global < 1 or self.window < self.prompts_global:
            raise ValueError("aux prompts per step >= 1 and window >= prompts")

    @property
    def aux_only(self) -> bool:
        return self.ratio >= 1.0

    @property
    def q(self) -> Fraction:
        """Aux steps per PReMind step (interleaved mode)."""
        r = _frac(self.ratio)
        return r / (1 - r)

    def due(self, step: int) -> list[int]:
        """Global aux indices to run right before PReMind step ``step`` (interleaved); [step] when aux-only."""
        if self.ratio <= 0:
            return []
        if self.aux_only:
            return [step]
        lo, hi = math.floor(self.q * step), math.floor(self.q * (step + 1))
        return list(range(lo, hi))

    def n_aux(self, premind_steps: int) -> int:
        """Aux optimizer steps that accompany ``premind_steps`` PReMind steps."""
        return 0 if self.ratio <= 0 else (premind_steps if self.aux_only else math.floor(self.q * premind_steps))

    def offset(self, aux_index: int, n_items: int) -> int:
        return (aux_index * self.window) % n_items


def make_schedule(ratio: float, p_aux: int, world: int, dapo_factor: float) -> AuxSchedule:
    pg = p_aux * world
    return AuxSchedule(ratio, pg, max(pg, math.ceil(dapo_factor * pg)))


def permuted(items: list, seed: int) -> list:
    out = list(items)
    random.Random(f"covip_aux_order:{seed}").shuffle(out)
    return out


def rotated(items: list, offset: int) -> list[Any]:
    return items[offset:] + items[:offset]
