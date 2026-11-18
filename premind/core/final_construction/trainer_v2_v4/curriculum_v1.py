"""Coverage curriculum v1 for PReMind v2 Stage A (owner-approved 2026-10-15). Pure Python, no torch.

Strata (one per train row): acute (image verdict or key-action row) x present|absent, non-acute unsafe x
present|absent, safe x present|absent ("partial" counts as present). Quotas over ``n_slots = steps x P_global``:
  acute present x mp / absent x ma, (mp, ma) = the first of ACUTE_LADDER that fits ((4,3) ... (2,2));
  non-acute unsafe present x2 / absent x1;  non-acute absent slots = absent_share x n_slots EXACTLY (unsafe absent
  first, safe absent fill it without replacement; acute absent rows are outside this cap);  safe present = the
  remainder, without replacement. A release that cannot satisfy this raises CurriculumConflict (never drops rows).
Order (seeded, deterministic): slot j goes to the stratum with the largest quota deficit (quota x (j+1)/n - taken),
so each step mixes strata in proportion to their quotas; inside a stratum, passes are reshuffled and the first row
not drawn in the last MIN_GAP_STEPS steps is taken.
Runtime (CurriculumSampler, drop-in for eval_v1.absent_sampler.AbsentShareSampler on the DDP path): the schedule is a
queue; DAPO refills are the next queue items; positions drawn but not rolled out are refunded; a group filtered
because ALL rollouts are wrong is re-queued ONCE, due 10 steps later, valid until 30 steps later (all-correct /
mixed-uniform groups are not re-queued); past the schedule, continuation cycles (seed + cycle) follow.
"""
from __future__ import annotations

import os

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

STRATA = ("acute_present", "acute_absent", "unsafe_present", "unsafe_absent", "safe_present", "safe_absent")
SHORT = {"acute_present": "AP", "acute_absent": "AA", "unsafe_present": "UP", "unsafe_absent": "UA",
         "safe_present": "SP", "safe_absent": "SA"}
ACUTE_LADDER = ((4, 3), (3, 3), (3, 2), (2, 2))          # (present, absent) passes; first that fits the budget
# owner 2026-10-15 (v12, option A): CUR_ACUTE_LADDER="2,2" fixes the acute passes (format "p,a;p,a;..."); the chosen
# passes are recorded in the quotas, hence in the schedule sha and the CUR pin.
if os.environ.get("CUR_ACUTE_LADDER"):
    ACUTE_LADDER = tuple(tuple(int(x) for x in p.split(",")) for p in os.environ["CUR_ACUTE_LADDER"].split(";"))
UNSAFE_PASSES = {"unsafe_present": 2, "unsafe_absent": 1}
MIN_GAP_STEPS = 10                                       # repeats of a row are >= 10 steps apart
REQUEUE = {"reason": "all_wrong", "times": 1, "min_delay_steps": 10, "max_delay_steps": 30,
           "not_requeued": ["all_correct", "uniform_mixed"], "refill": "next scheduled items"}
ABSENT = "gt_absent"


class CurriculumConflict(SystemExit):
    """The release cannot satisfy the quotas (reported, never resolved by dropping rows)."""


class CurriculumCheckFailed(SystemExit):
    """A pre-training sanity check failed."""


def stratum(row: dict, acute: bool) -> str:
    side = "absent" if row.get("condition") == ABSENT else "present"
    if acute:
        return f"acute_{side}"
    return f"{'unsafe' if row.get('risk_label') == 'unsafe' else 'safe'}_{side}"


def strata_of(items: Iterable[Any], acute_ids: set) -> dict[str, list[str]]:
    """{stratum: sorted unique row ids}; refuses a row id with two strata."""
    by: dict[str, str] = {}
    for it in items:
        s = stratum(it.row, is_acute(it, acute_ids))
        if by.setdefault(it.row_id, s) != s:
            raise CurriculumConflict(f"REFUSED: row {it.row_id} falls in two strata")
    out = {s: [] for s in STRATA}
    for rid, s in sorted(by.items()):
        out[s].append(rid)
    return out


def is_acute(item: Any, acute_ids: set) -> bool:
    from eval_v1 import acute_rule_v4 as AR                # read-only
    return AR.is_acute_row(item.row) or item.row_id in acute_ids


def compute_quotas(counts: dict[str, int], n_slots: int, absent_share: float) -> dict:
    """{passes, slots, acute_passes, n_absent_slots}; CurriculumConflict lists every unmet rule."""
    q = absent_share * n_slots
    if abs(q - round(q)) > 1e-9:
        raise CurriculumConflict(f"REFUSED: absent share {absent_share} x {n_slots} slots is not an integer")
    n_abs = int(round(q))
    up, ua = UNSAFE_PASSES["unsafe_present"] * counts["unsafe_present"], counts["unsafe_absent"]
    bad = []
    if ua > n_abs:
        bad.append(f"non-acute unsafe absent rows ({ua}) exceed the absent slots ({n_abs})")
    sa = n_abs - ua
    if sa > counts["safe_absent"]:
        bad.append(f"safe absent fill {sa} > {counts['safe_absent']} rows (without replacement)")
    if sa > 0 and counts["safe_absent"] == 0:
        bad.append("no safe absent rows to fill the absent share")
    acute_passes, sp = None, None
    for mp, ma in ACUTE_LADDER:
        rest = n_slots - mp * counts["acute_present"] - ma * counts["acute_absent"] - up - n_abs
        if rest >= 0:
            acute_passes, sp = (mp, ma), rest
            break
    if acute_passes is None:
        bad.append(f"acute x2 + unsafe (present x2, absent x1) + {n_abs} absent slots exceed {n_slots} slots")
    elif sp > counts["safe_present"]:
        bad.append(f"safe present remainder {sp} > {counts['safe_present']} rows (without replacement)")
    if bad:
        raise CurriculumConflict("REFUSED: curriculum conflict: " + "; ".join(bad) + f" (counts {counts})")
    mp, ma = acute_passes
    slots = {"acute_present": mp * counts["acute_present"], "acute_absent": ma * counts["acute_absent"],
             "unsafe_present": up, "unsafe_absent": ua, "safe_present": sp, "safe_absent": sa}
    passes = {"acute_present": mp, "acute_absent": ma, **UNSAFE_PASSES, "safe_present": None, "safe_absent": None}
    return {"n_slots": n_slots, "absent_share": absent_share, "n_absent_slots": n_abs, "acute_passes": [mp, ma],
            "counts": dict(counts), "passes": passes, "slots": slots}


def _stream(rows: list[str], passes: Optional[int], slots: int, seed: int, name: str) -> list[str]:
    """Stratum queue: ``passes`` reshuffled passes (safe strata: one shuffled pass truncated to ``slots``)."""
    out: list[str] = []
    for k in range(passes if passes is not None else 1):
        p = list(rows)
        random.Random(f"curriculum_v1|{seed}|{name}|{k}").shuffle(p)
        out += p
    return out[:slots]


def build_schedule(strata: dict[str, list[str]], quotas: dict, seed: int, p_global: int) -> list[str]:
    n = quotas["n_slots"]
    queues = {s: _stream(strata[s], quotas["passes"][s], quotas["slots"][s], seed, s) for s in STRATA}
    taken = Counter()
    last: dict[str, int] = {}
    out: list[str] = []
    for j in range(n):
        step = j // p_global
        order = sorted((s for s in STRATA if taken[s] < quotas["slots"][s]),
                       key=lambda s: (-(quotas["slots"][s] * (j + 1) / n - taken[s]), STRATA.index(s)))
        pick = None
        for s in order:
            q = queues[s]
            i = next((i for i, r in enumerate(q) if step - last.get(r, -MIN_GAP_STEPS) >= MIN_GAP_STEPS), None)
            if i is not None:
                pick = (s, q.pop(i))
                break
        if pick is None:
            raise CurriculumConflict(f"REFUSED: no row can fill slot {j} (step {step}) without a repeat within "
                                     f"{MIN_GAP_STEPS} steps; remaining {dict(taken)}")
        taken[pick[0]] += 1
        last[pick[1]] = step
        out.append(pick[1])
    return out


def schedule_sha(schedule: list[str]) -> str:
    return hashlib.sha256(json.dumps(schedule).encode()).hexdigest()


def coverage_report(schedule: list[str], strata: dict[str, list[str]], quotas: dict, p_global: int) -> dict:
    of = {r: s for s, rows in strata.items() for r in rows}
    cnt = Counter(schedule)
    per = {}
    for s in STRATA:
        c = [cnt[r] for r in strata[s]]
        per[s] = {"n_rows": len(c), "slots": sum(c), "passes": quotas["passes"][s],
                  "rows_by_count": dict(sorted(Counter(c).items())), "min": min(c, default=0), "max": max(c, default=0)}
    steps = [schedule[i:i + p_global] for i in range(0, len(schedule), p_global)]
    comp = ["".join(sorted(SHORT[of[r]] + " " for r in st)).strip() for st in steps]
    acute = [sum(of[r].startswith("acute") for r in st) for st in steps]
    n_abs = sum(of[r] in ("unsafe_absent", "safe_absent") for r in schedule)
    return {"n_slots": len(schedule), "p_global": p_global, "n_steps": len(steps), "quotas": quotas,
            "per_stratum": per, "per_row_count": {r: cnt[r] for r in sorted(of)},
            "non_acute_absent_share": n_abs / len(schedule), "all_absent_share":
                sum(of[r].endswith("absent") for r in schedule) / len(schedule),
            "acute_per_step": {"min": min(acute), "max": max(acute), "mean": sum(acute) / len(acute),
                               "hist": dict(sorted(Counter(acute).items()))},
            "step_composition_hist": dict(Counter(comp).most_common()), "per_step": comp,
            "schedule_sha256": schedule_sha(schedule)}


def sanity_checks(schedule: list[str], strata: dict[str, list[str]], quotas: dict, p_global: int,
                  release_ids: set, rebuild: Callable[[], list[str]]) -> dict:
    """The 5 owner checks; raises CurriculumCheckFailed listing every failure. Returns {check: "PASS"}."""
    of = {r: s for s, rows in strata.items() for r in rows}
    cnt = Counter(schedule)
    fail: dict[str, str] = {}
    miss = [r for s in ("acute_present", "acute_absent") for r in strata[s] if cnt[r] < 2]
    miss += [r for s in ("unsafe_present", "unsafe_absent") for r in strata[s] if cnt[r] < 1]
    wrong = [r for s in STRATA if quotas["passes"][s] for r in strata[s] if cnt[r] != quotas["passes"][s]]
    if miss or wrong:
        fail["1_coverage"] = f"under-scheduled {miss[:10]} ({len(miss)}), off-quota {wrong[:10]} ({len(wrong)})"
    n_abs = sum(of.get(r) in ("unsafe_absent", "safe_absent") for r in schedule)
    if n_abs != quotas["n_absent_slots"] or abs(n_abs / len(schedule) - quotas["absent_share"]) > 1e-12:
        fail["2_absent_share"] = f"{n_abs}/{len(schedule)} non-acute absent, want {quotas['n_absent_slots']}"
    last: dict[str, int] = {}
    rep = []
    for j, r in enumerate(schedule):
        st = j // p_global
        if r in last and st - last[r] < MIN_GAP_STEPS:
            rep.append((r, last[r], st))
        last[r] = st
    if rep:
        fail["3_no_repeat_10"] = f"{len(rep)} repeats, e.g. {rep[:5]}"
    again = rebuild()
    if schedule_sha(again) != schedule_sha(schedule):
        fail["4_deterministic"] = "a rebuild gives another schedule"
    unknown = sorted(set(schedule) - release_ids)
    uncovered = sorted(r for r in release_ids if of.get(r, "").startswith(("acute", "unsafe")) and not cnt[r])
    if unknown or set(of) != release_ids or uncovered or len(schedule) != quotas["n_slots"]:
        fail["5_release_ids"] = (f"unknown {unknown[:5]} ({len(unknown)}), strata/release mismatch "
                                 f"{len(set(of) ^ release_ids)}, uncovered {uncovered[:5]}, len {len(schedule)}")
    if fail:
        raise CurriculumCheckFailed("CURRICULUM SANITY CHECK FAILED: " + json.dumps(fail))
    return {k: "PASS" for k in ("1_coverage", "2_absent_share", "3_no_repeat_10", "4_deterministic",
                                "5_release_ids")}


def classify(successes: list[bool]) -> str:
    """Reason of a filtered (zero-variance) group: all_wrong / all_correct / uniform_mixed."""
    if not any(successes):
        return "all_wrong"
    return "all_correct" if all(successes) else "uniform_mixed"


class Plan:
    """Everything the run needs: strata, quotas, schedule (+ continuation cycles)."""

    def __init__(self, items: list, acute_ids: set, steps: int, p_global: int, absent_share: float, seed: int):
        self.strata = strata_of(items, acute_ids)
        self.counts = {s: len(v) for s, v in self.strata.items()}
        self.steps, self.p_global, self.seed = steps, p_global, seed
        self.quotas = compute_quotas(self.counts, steps * p_global, absent_share)
        self.schedule = self.build(0)
        self.sha = schedule_sha(self.schedule)

    def build(self, cycle: int) -> list[str]:
        return build_schedule(self.strata, self.quotas, self.seed + 1000 * cycle, self.p_global)

    def of_rows(self) -> list[str]:
        return [r for s in STRATA for r in self.strata[s]]

    def report(self) -> dict:
        return coverage_report(self.schedule, self.strata, self.quotas, self.p_global)

    def check(self, release_ids: set) -> dict:
        return sanity_checks(self.schedule, self.strata, self.quotas, self.p_global, release_ids,
                             lambda: Plan.build(self, 0))


class CurriculumSampler:
    """Drop-in for AbsentShareSampler (draw / skip / payload / summary; fork-safe via __copy__). ``set_step`` must be
    called before each step's draws (main_v4hcur's ddp_train_one_step hook)."""

    def __init__(self, items: list, plan: Plan, log_path: Optional[Path] = None) -> None:
        self.items = items
        self.by_id = {it.row_id: it for it in items}
        self.of = {r: s for s, rows in plan.strata.items() for r in rows}
        self.plan, self.log_path = plan, log_path
        self.rng = random.Random(plan.seed)                 # unused; fork_sampler copies it
        self.pools: dict = {}
        self.counts: Counter = Counter()
        self.queue: list[list] = [[r, "sched"] for r in plan.schedule]
        self.cycle = 0
        self.pending: list[dict] = []
        self.last: dict[str, list[int]] = {}
        self.step: Optional[int] = None
        self.step_draws: list[list] = []
        self.n = 0
        self.stats: Counter = Counter()

    # ---------------------------------------------------------------- state
    def __copy__(self) -> "CurriculumSampler":
        c = object.__new__(CurriculumSampler)
        c.__dict__.update({k: (v if k in ("items", "by_id", "of", "plan", "log_path") else copy.deepcopy(v))
                           for k, v in self.__dict__.items()})
        return c

    def state(self) -> dict:
        return {"queue": self.queue, "cycle": self.cycle, "pending": self.pending, "last": self.last, "n": self.n,
                "counts": dict(self.counts), "stats": dict(self.stats)}

    def load_state(self, st: dict) -> None:
        self.queue, self.cycle, self.pending = [list(x) for x in st["queue"]], st["cycle"], list(st["pending"])
        self.last, self.n = {k: list(v) for k, v in st["last"].items()}, st["n"]
        self.counts, self.stats = Counter(st["counts"]), Counter(st["stats"])

    def set_step(self, step: int) -> None:
        self.step, self.step_draws = step, []

    # ---------------------------------------------------------------- draws
    def _ok(self, r: str) -> bool:
        return not self.last.get(r) or self.step - self.last[r][-1] >= MIN_GAP_STEPS

    def _one(self) -> list:
        for pe in sorted(self.pending, key=lambda p: (p["due"], p["src"])):
            if pe["due"] <= self.step and self._ok(pe["row_id"]):
                self.pending.remove(pe)
                return [pe["row_id"], "requeue", pe]
        for _ in range(2):
            i = next((i for i, (r, _) in enumerate(self.queue) if self._ok(r)), None)
            if i is not None:
                r, kind = self.queue.pop(i)
                return [r, kind, None]
            self.cycle += 1                                  # schedule used up (refills): continuation cycle
            self.queue += [[r, "cont"] for r in self.plan.build(self.cycle)]
        r, kind = self.queue.pop(0)                          # no eligible row at all: counted, never silent
        self.stats["forced_repeat"] += 1
        return [r, kind, None]

    def _expire(self) -> None:
        for pe in [p for p in self.pending if p["deadline"] < self.step]:
            self.pending.remove(pe)
            self.stats["requeue_expired"] += 1

    def draw(self, k: int) -> list:
        if k <= 0:
            return []
        if self.step is None:
            raise RuntimeError("curriculum v1: set_step was not called (the curriculum runs on the DDP path only)")
        self._expire()
        out = []
        for _ in range(k):
            r, kind, pe = self._one()
            self.last.setdefault(r, []).append(self.step)
            self.step_draws.append([r, kind, pe])
            self.n += 1
            self.counts[self.of[r]] += 1
            out.append(self.by_id[r])
        return out

    def skip(self, k: int) -> None:
        """Within a step: consume ``k`` positions (DAPO refill bookkeeping). Before the first step (resume): restore
        the state logged after the step whose cumulative cursor is ``k``."""
        if self.step is None:
            if k:
                self.restore(k)
            return
        self.draw(k)

    def restore(self, cursor: int) -> None:
        hit = None
        if self.log_path and Path(self.log_path).is_file():
            for line in Path(self.log_path).read_text().splitlines():
                rec = json.loads(line)
                if rec.get("cursor_after") == cursor and rec.get("schedule_sha256") == self.plan.sha:
                    hit = rec
        if hit is None:
            raise SystemExit(f"REFUSED: curriculum resume: no logged state with cursor {cursor} in {self.log_path}")
        self.load_state(hit["state"])

    def refund(self, unused: list[int]) -> list[str]:
        """Positions (indices into this step's draws) drawn but not rolled out go back: queue front / pending."""
        back = []
        for i in sorted(unused, reverse=True):
            r, kind, pe = self.step_draws[i]
            self.last[r].pop()
            self.n -= 1
            self.counts[self.of[r]] -= 1
            if kind == "requeue":
                self.pending.append(pe)
            else:
                self.queue.insert(0, [r, kind])
            back.append(r)
        keep = set(unused)
        self.step_draws = [d for i, d in enumerate(self.step_draws) if i not in keep]
        return back[::-1]

    def requeue(self, filtered: list[tuple[str, str]]) -> list[dict]:
        """filtered = [(row_id, reason)] of this step (all ranks). all_wrong -> re-queued once."""
        kinds = {r: kind for r, kind, _ in self.step_draws}
        out = []
        for r, why in filtered:
            self.stats[f"filtered_{why}"] += 1
            if why != REQUEUE["reason"]:
                continue
            if kinds.get(r) == "requeue":
                self.stats["requeue_once_done"] += 1
                continue
            pe = {"row_id": r, "src": self.step, "due": self.step + REQUEUE["min_delay_steps"],
                  "deadline": self.step + REQUEUE["max_delay_steps"]}
            self.pending.append(pe)
            self.stats["requeued"] += 1
            out.append(pe)
        return out

    # ---------------------------------------------------------------- logging
    @property
    def absent_share(self) -> Optional[float]:
        tot = sum(self.counts.values())
        return (self.counts["unsafe_absent"] + self.counts["safe_absent"]) / tot if tot else None

    def payload(self, batch: list) -> dict:
        c = Counter(self.of[it.row_id] for it in batch)
        out: dict = {"train/data/absent_share": self.absent_share}
        out.update({f"train/data/cur_n_{SHORT[s]}": c[s] for s in STRATA})
        out["train/data/cur_n_acute"] = c["acute_present"] + c["acute_absent"]
        return out

    def summary(self) -> dict:
        return {"curriculum": "v1", "schedule_sha256": self.plan.sha, "non_acute_absent_share": self.absent_share,
                "n_draws": self.n, "stratum_counts": dict(self.counts), "cycle": self.cycle, "stats": dict(self.stats),
                "pending": len(self.pending)}
