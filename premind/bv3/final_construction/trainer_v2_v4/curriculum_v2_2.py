"""Curriculum v2.2 = curriculum v2.1 + SAFE_MIN_FRAC (v14 fallback, owner-approved 2026-10-15). NEW file:
curriculum_v2_1.py (pinned by the live CUR21B run) is imported read-only; the sampler is CurriculumSamplerV21.

``compute_quotas22`` = compute_quotas21, then, when ``safe_min_frac`` is set (default None = off; fallback 0.30):
  need = ceil(safe_min_frac x n_slots) - (safe present + safe absent slots)
  1. take min(need, replay budget) replay slots -> safe present (the sampler's acute repeat cap = the reduced budget)
  2. still short and unsafe present is x2: unsafe present -> x1 (whole pass; coverage stays exact) -> safe present
  3. still short -> CurriculumConflict (never cuts acute first pass, unsafe x1 or the absent slots).
Safe absent stays absent_share x n_slots - unsafe absent (the non-acute absent share is untouched), so the extra safe
slots are safe present (without replacement, <= the safe present rows).
"""
from __future__ import annotations

import math
from typing import Optional

from trainer_v2_v4 import curriculum_v1 as CV
from trainer_v2_v4 import curriculum_v2_1 as C21

SAFE_MIN_FRAC_FALLBACK = 0.30


def compute_quotas22(counts: dict[str, int], n_slots: int, absent_share: float, replay_budget: int,
                     safe_min_frac: Optional[float] = None) -> dict:
    q = C21.compute_quotas21(counts, n_slots, absent_share, replay_budget)
    q = {**q, "slots": dict(q["slots"]), "passes": dict(q["passes"]), "safe_min_frac": safe_min_frac,
         "replay_budget_requested": replay_budget, "safe_min_moves": {"from_replay": 0, "from_unsafe_present": 0}}
    if safe_min_frac is None:
        return q
    if not 0.0 < safe_min_frac < 1.0:
        raise CV.CurriculumConflict(f"REFUSED: safe_min_frac {safe_min_frac} not in (0, 1)")
    s = q["slots"]
    need = math.ceil(safe_min_frac * n_slots - 1e-9) - (s["safe_present"] + s["safe_absent"])
    if need > 0:
        take = min(need, s["replay"])
        s["replay"] -= take
        s["safe_present"] += take
        q["safe_min_moves"]["from_replay"] = take
        need -= take
    if need > 0 and q["passes"]["unsafe_present"] == 2:
        free = counts["unsafe_present"]
        s["unsafe_present"] -= free
        s["safe_present"] += free
        q["passes"]["unsafe_present"] = q["unsafe_present_passes"] = 1
        q["safe_min_moves"]["from_unsafe_present"] = free
        need -= free
    if need > 0:
        raise CV.CurriculumConflict(f"REFUSED: safe_min_frac {safe_min_frac}: {need} safe slots short after the "
                                    f"replay budget and unsafe present x1 (counts {counts})")
    if s["safe_present"] > counts["safe_present"]:
        raise CV.CurriculumConflict(f"REFUSED: safe present {s['safe_present']} > {counts['safe_present']} rows "
                                    f"(without replacement)")
    q["replay_budget"] = s["replay"]
    return q


class Plan22(C21.Plan21):
    """Plan21 with compute_quotas22 (same schedule builder and checks)."""

    def __init__(self, items: list, acute_ids: set, steps: int, p_global: int, absent_share: float, seed: int,
                 replay_budget: int = C21.REPLAY_BUDGET, safe_min_frac: Optional[float] = None):
        self.strata = CV.strata_of(items, acute_ids)
        self.counts = {s: len(v) for s, v in self.strata.items()}
        self.steps, self.p_global, self.seed = steps, p_global, seed
        self.quotas = compute_quotas22(self.counts, steps * p_global, absent_share, replay_budget, safe_min_frac)
        self.schedule = self.build(0)
        self.sha = CV.schedule_sha(self.schedule)

    def report(self) -> dict:
        return {**super().report(), "curriculum": "v2.2", "safe_min_frac": self.quotas["safe_min_frac"],
                "safe_slots_share": (self.quotas["slots"]["safe_present"] + self.quotas["slots"]["safe_absent"])
                / self.quotas["n_slots"]}


__all__ = ["SAFE_MIN_FRAC_FALLBACK", "compute_quotas22", "Plan22"]
