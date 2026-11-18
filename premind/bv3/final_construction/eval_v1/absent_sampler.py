"""Seeded training-prompt sampler with a target share of GT-absent contexts (owner rule 2026-10-03).

Each prompt first picks a side (``gt_absent`` vs the rest) by error diffusion with a seeded phase, so the realised
gt_absent share stays within 1/n of the target after n draws. Within a side it picks a (condition, row_type)
stratum with probability proportional to the stratum size (keeps the remaining mass proportional across
gt_present / partial and across row types), then takes the next item of that stratum's shuffled pool (without
replacement; reshuffled when exhausted). The state is a pure function of (items, target, seed, number of draws), so a
resume rebuilds it by replaying ``start * P`` draws.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

ABSENT = "gt_absent"
SEED_SALT = 0xAB5E          # own RNG stream: never touches the global python / torch RNG of the trainer


def condition(item: Any) -> str:
    return item.row["condition"]


class AbsentShareSampler:
    def __init__(self, items: list, target: float, seed: int) -> None:
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target absent share must be in [0, 1]: {target}")
        self.items, self.target = items, target
        self.rng = random.Random(seed * 1_000_003 + SEED_SALT)
        strata: dict[tuple[str, str], list[int]] = {}
        for i, it in enumerate(items):
            strata.setdefault((condition(it), it.row.get("row_type")), []).append(i)
        self.strata = dict(sorted(strata.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))))
        self.sides = {s: [k for k in self.strata if (k[0] == ABSENT) == s] for s in (True, False)}
        if target > 0 and not self.sides[True]:
            raise ValueError("target absent share > 0 but no gt_absent items")
        if target < 1 and not self.sides[False]:
            raise ValueError("target absent share < 1 but no non-absent items")
        self.pools: dict[tuple[str, str], list[int]] = {k: [] for k in self.strata}
        self.phase = self.rng.random()
        self.n, self.n_absent = 0, 0
        self.counts: Counter = Counter()

    def _pick_side(self) -> bool:
        want = math.floor(self.target * (self.n + 1) + self.phase)
        return self.n_absent < want

    def _next(self, key: tuple[str, str]) -> Any:
        if not self.pools[key]:
            pool = list(self.strata[key])
            self.rng.shuffle(pool)
            self.pools[key] = pool[::-1]          # pop() from the end = shuffled order
        return self.items[self.pools[key].pop()]

    def draw(self, k: int) -> list:
        out = []
        for _ in range(k):
            absent = self._pick_side()
            keys = self.sides[absent]
            key = self.rng.choices(keys, weights=[len(self.strata[s]) for s in keys])[0]
            out.append(self._next(key))
            self.n += 1
            self.n_absent += int(absent)
            self.counts[key[0]] += 1
        return out

    def skip(self, k: int) -> None:
        """Resume: replay ``k`` draws (deterministic) without keeping the items."""
        self.draw(k)

    @property
    def absent_share(self) -> float | None:
        return self.n_absent / self.n if self.n else None

    def payload(self, batch: list) -> dict:
        """wandb ``train/data/*``: running absent share plus this step's prompt counts per condition."""
        out: dict = {"train/data/absent_share": self.absent_share}
        c = Counter(condition(it) for it in batch)
        for cond in sorted({k[0] for k in self.strata}):
            out[f"train/data/n_{cond}"] = c[cond]
        return out

    def summary(self) -> dict:
        return {"target_absent_share": self.target, "realised_absent_share": self.absent_share,
                "n_draws": self.n, "condition_counts": dict(sorted(self.counts.items()))}
