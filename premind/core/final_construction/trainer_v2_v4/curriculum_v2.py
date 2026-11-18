"""Mastery-guided replay curriculum v2 (owner 2026-10-15: "reuse what was NOT learned, e.g. unlearned G1G2 acute rows,
instead of a blind acute second pass"). Pure Python, no torch. curriculum_v1.py is imported read-only, never edited.

Plan (same strata, interleaving, gap and absent rule as v1; quotas from v1 with the LIVE pilot CUR ladder V1_LADDER):
  v1 (CUR, ladder (2,2)) schedules acute present x2 / absent x2. v2 schedules every acute row ONCE and turns the
  (2-1) x AP + (2-1) x AA second-pass slots into REPLAY placeholders (same total slot count, n_slots = steps x P).
  Non-acute strata (unsafe present x2 / absent x1, safe fills) and the non-acute absent slots are exactly v1's.
  Placeholders are a seventh stratum interleaved by the same quota-deficit rule from step REPLAY_START_STEP on.
Mastery (``rollout_ok`` / ``group_outcome``): after each rolled-out group, mastery = share of its rollouts that are
  FULLY correct on the row's applicable criteria (weights = reward_v5.W, never hard-coded); see ``rollout_ok``.
Replay queue (runtime): a visited row with mastery < MASTERY is queued (due +10 steps, deadline +30, as v1's
  re-queue) unless it hit its visit cap (acute 3, non-acute 1 = ACUTE-ONLY replay, orchestrator 2026-10-15: the
  non-acute draws stay exactly v1's) or was DAPO re-queued this step (v1 rule,
  unchanged; a replayed row is never also DAPO re-queued). mastery >= MASTERY removes it. A placeholder takes the
  eligible entry with the smallest (tier, -priority, due, row_id): tier 0 = acute G1G2 target-present rows whose
  personal_content rate < MASTERY (owner-named), tier 1 = the rest; priority = 1 - mastery. (Inert with the
  acute-only cap, kept for a non-acute cap > 1 and tested:) a non-acute-absent replay cancels the next scheduled safe-absent fill (swap; the schedule gets one position shorter), so the replay
  budget stays v1's acute second-pass count and the non-acute absent draw count of the schedule stays exactly v1's;
  with no fill left such a row is not replayed. Empty queue: the placeholder is
  consumed and the next scheduled item is drawn instead (``replay_fallback``, logged with the step).
Deterministic given seed + observed outcomes (no RNG at runtime).
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from trainer_v2_v4 import curriculum_v1 as CV

V1_LADDER = (2, 2)            # the live pilot CUR acute passes (CUR_ACUTE_LADDER="2,2"); fixes the replay budget
MASTERY = 0.75                # mastery >= MASTERY -> never replayed
MAX_VISITS = {"acute": 3, "non_acute": 1}   # acute-only replay (orchestrator 2026-10-15)
REPLAY = {"min_delay_steps": 10, "max_delay_steps": 30}
REPLAY_START_STEP = 10        # no placeholder before outcomes can be due
REPLAY_TOKEN = "@replay"
STRATA2 = CV.STRATA + ("replay",)
NAA = ("unsafe_absent", "safe_absent")
EPS = 1e-9


# ----------------------------------------------------------------------------- mastery criterion
def _full(comps: dict, name: str, w: dict) -> Optional[bool]:
    """None = not applicable; else the component earned its full weight."""
    c = comps.get(name) or {}
    if not c.get("applicable") or c.get("value") is None:
        return None
    return c["value"] >= w[name] - EPS


def rollout_ok(ledger: dict, acute: bool, w: Optional[dict] = None) -> dict:
    """Is one scored rollout FULLY correct on its row's applicable criteria? -> {"ok", "pc"} (pc = personal_content
    full, None if not applicable). Degenerate -> not ok.
      acute rows (hazard state present in ledger["v4h"]): hazard state == specific AND not contraindicated AND action
                 correct AND personal_content full when applicable (G1 / G1G2 target-present rows)
      other rows: action correct AND risk full when applicable AND personal_content / withhold full when applicable
    (an acute row without a hazard state in its ledger falls back to the non-acute criterion)."""
    if w is None:
        from trainer_v2 import reward_v5 as R5
        w = R5.W
    comps = ledger.get("components") or {}
    if (comps.get("degenerate") or {}).get("applicable"):
        return {"ok": False, "pc": False if (comps.get("personal_content") or {}).get("applicable") else None}
    pc = _full(comps, "personal_content", w)
    checks = [_full(comps, "action", w) is True, pc is not False]
    hz = ledger.get("v4h") or {}
    if acute and hz.get("acute") and "state" in hz:
        checks.append(hz["state"] == "specific" and not hz.get("contraindicated"))
    else:
        checks += [_full(comps, "risk", w) is not False, _full(comps, "withhold", w) is not False]
    return {"ok": all(checks), "pc": pc}


def group_outcome(row_id: str, ledgers: list[dict], acute: bool, w: Optional[dict] = None) -> dict:
    """{row_id, mastery, pc_rate (None if pc never applicable), n} of one group's rollouts."""
    res = [rollout_ok(led, acute, w) for led in ledgers]
    pcs = [r["pc"] for r in res if r["pc"] is not None]
    return {"row_id": row_id, "mastery": round(sum(r["ok"] for r in res) / len(res), 6) if res else 0.0,
            "pc_rate": round(sum(pcs) / len(pcs), 6) if pcs else None, "n": len(res)}


# ----------------------------------------------------------------------------- plan
def category(row: dict, acute: bool) -> str:
    """row_type (G0 / G1 / G1G2) x acute x condition (partial counts as present)."""
    rt = str(row.get("row_type") or "")
    coarse = "G1G2" if rt.startswith("G1G2") else ("G1" if rt.startswith("G1") else ("G0" if rt.startswith("G0")
                                                                                     else rt or "?"))
    return f"{coarse}|{'acute' if acute else 'non'}|{'absent' if row.get('condition') == CV.ABSENT else 'present'}"


def compute_quotas(counts: dict[str, int], n_slots: int, absent_share: float) -> dict:
    """v1's quotas under V1_LADDER (v1's conflict checks), acute passes -> 1, the rest -> ``replay`` slots."""
    saved = CV.ACUTE_LADDER
    CV.ACUTE_LADDER = (V1_LADDER,)                      # in-process only, restored; curriculum_v1.py is untouched
    try:
        v1 = CV.compute_quotas(counts, n_slots, absent_share)
    finally:
        CV.ACUTE_LADDER = saved
    slots = dict(v1["slots"], acute_present=counts["acute_present"], acute_absent=counts["acute_absent"])
    slots["replay"] = v1["slots"]["acute_present"] + v1["slots"]["acute_absent"] - slots["acute_present"] \
        - slots["acute_absent"]
    passes = dict(v1["passes"], acute_present=1, acute_absent=1, replay=None)
    return {**v1, "passes": passes, "slots": slots, "v1_slots": v1["slots"], "v1_acute_passes": list(V1_LADDER)}


def build_schedule(strata: dict[str, list[str]], quotas: dict, seed: int, p_global: int) -> list[str]:
    """v1's quota-deficit interleaving over STRATA2; replay placeholders paced from REPLAY_START_STEP on and exempt
    from the repeat gap."""
    n, slots = quotas["n_slots"], quotas["slots"]
    j0 = min(REPLAY_START_STEP * p_global, n - 1)
    queues = {s: CV._stream(strata[s], quotas["passes"][s], slots[s], seed, s) for s in CV.STRATA}
    queues["replay"] = [REPLAY_TOKEN] * slots["replay"]
    taken: Counter = Counter()
    last: dict[str, int] = {}
    out: list[str] = []

    def target(s: str, j: int) -> float:
        if s != "replay":
            return slots[s] * (j + 1) / n
        return slots[s] * (j + 1 - j0) / (n - j0) if j >= j0 else -1.0
    for j in range(n):
        step = j // p_global
        order = sorted((s for s in STRATA2 if taken[s] < slots[s] and (s != "replay" or j >= j0)),
                       key=lambda s: (-(target(s, j) - taken[s]), STRATA2.index(s)))
        pick = None
        for s in order:
            q = queues[s]
            if s == "replay":
                pick = (s, q.pop(0))
                break
            i = next((i for i, r in enumerate(q) if step - last.get(r, -CV.MIN_GAP_STEPS) >= CV.MIN_GAP_STEPS), None)
            if i is not None:
                pick = (s, q.pop(i))
                break
        if pick is None:
            raise CV.CurriculumConflict(f"REFUSED: no row can fill slot {j} (step {step}); remaining {dict(taken)}")
        taken[pick[0]] += 1
        if pick[1] != REPLAY_TOKEN:
            last[pick[1]] = step
        out.append(pick[1])
    return out


class Plan2(CV.Plan):
    """v1 Plan with the v2 quotas / schedule (placeholders = REPLAY_TOKEN)."""

    def __init__(self, items: list, acute_ids: set, steps: int, p_global: int, absent_share: float, seed: int):
        self.strata = CV.strata_of(items, acute_ids)
        self.counts = {s: len(v) for s, v in self.strata.items()}
        self.steps, self.p_global, self.seed = steps, p_global, seed
        self.quotas = compute_quotas(self.counts, steps * p_global, absent_share)
        self.schedule = self.build(0)
        self.sha = CV.schedule_sha(self.schedule)

    def build(self, cycle: int) -> list[str]:
        return build_schedule(self.strata, self.quotas, self.seed + 1000 * cycle, self.p_global)

    def report(self) -> dict:
        of = {r: s for s, rows in self.strata.items() for r in rows}
        cnt = Counter(self.schedule)
        rows = [r for r in self.schedule if r != REPLAY_TOKEN]
        rep_steps = [j // self.p_global for j, r in enumerate(self.schedule) if r == REPLAY_TOKEN]
        per = {s: {"n_rows": len(self.strata[s]), "slots": sum(cnt[r] for r in self.strata[s]),
                   "rows_by_count": dict(sorted(Counter(cnt[r] for r in self.strata[s]).items()))} for s in CV.STRATA}
        return {"curriculum": "v2", "n_slots": len(self.schedule), "p_global": self.p_global, "quotas": self.quotas,
                "per_stratum": per, "replay_slots": len(rep_steps),
                "replay_first_step": min(rep_steps, default=None), "replay_per_50_steps": dict(sorted(Counter(
                    s // 50 * 50 for s in rep_steps).items())),
                "non_acute_absent_share": sum(of[r] in NAA for r in rows) / len(self.schedule),
                "schedule_sha256": self.sha}

    def check(self, release_ids: set) -> dict:
        """v2 checks (CurriculumCheckFailed lists every failure): coverage (acute x1, unsafe x passes, nothing over
        quota), exact non-acute absent share, no repeat within 10 steps, determinism, release ids, budget = v1."""
        import json
        of = {r: s for s, rows in self.strata.items() for r in rows}
        cnt = Counter(self.schedule)
        q, fail = self.quotas, {}
        wrong = [r for s in CV.STRATA if q["passes"][s] for r in self.strata[s] if cnt[r] != q["passes"][s]]
        if wrong:
            fail["1_coverage"] = f"off-quota {wrong[:10]} ({len(wrong)})"
        n_abs = sum(of.get(r) in NAA for r in self.schedule)
        if n_abs != q["n_absent_slots"]:
            fail["2_absent_share"] = f"{n_abs}/{len(self.schedule)} non-acute absent, want {q['n_absent_slots']}"
        last: dict[str, int] = {}
        rep = []
        for j, r in enumerate(self.schedule):
            if r == REPLAY_TOKEN:
                continue
            if r in last and j // self.p_global - last[r] < CV.MIN_GAP_STEPS:
                rep.append(r)
            last[r] = j // self.p_global
        if rep:
            fail["3_no_repeat_10"] = f"{len(rep)} repeats, e.g. {rep[:5]}"
        if CV.schedule_sha(self.build(0)) != self.sha:
            fail["4_deterministic"] = "a rebuild gives another schedule"
        if set(of) != release_ids or set(self.schedule) - set(of) - {REPLAY_TOKEN}:
            fail["5_release_ids"] = f"strata/release mismatch {len(set(of) ^ release_ids)}"
        v1_acute2 = q["v1_slots"]["acute_present"] + q["v1_slots"]["acute_absent"]
        if (len(self.schedule) != q["n_slots"] or cnt[REPLAY_TOKEN] + q["slots"]["acute_present"]
                + q["slots"]["acute_absent"] != v1_acute2):
            fail["6_budget_v1"] = f"replay {cnt[REPLAY_TOKEN]} + acute != v1 acute slots {v1_acute2}"
        if fail:
            raise CV.CurriculumCheckFailed("CURRICULUM V2 SANITY CHECK FAILED: " + json.dumps(fail))
        return {k: "PASS" for k in ("1_coverage", "2_absent_share", "3_no_repeat_10", "4_deterministic",
                                    "5_release_ids", "6_budget_v1")}


# ----------------------------------------------------------------------------- runtime
class CurriculumSamplerV2(CV.CurriculumSampler):
    """v1 sampler + replay placeholders, the mastery-guided replay queue and per-row mastery history.
    Call order per step (main_v4hcur2): set_step -> draws -> refund -> requeue(filtered) -> observe(outcomes)."""

    SHARED = ("items", "by_id", "of", "plan", "log_path", "cat", "meta_acute")

    def __init__(self, items: list, plan: Plan2, log_path: Optional[Path] = None) -> None:
        super().__init__(items, plan, log_path)
        self.queue = [[r, "replay" if r == REPLAY_TOKEN else "sched"] for r in plan.schedule]
        self.meta_acute = {r: s.startswith("acute") for r, s in self.of.items()}
        self.cat = {it.row_id: category(it.row, self.meta_acute[it.row_id]) for it in items}
        self.replay: list[dict] = []
        self.hist: dict[str, list] = {}
        self.events: list = []                               # this step's replay events (logged)

    def __copy__(self) -> "CurriculumSamplerV2":
        c = object.__new__(type(self))
        c.__dict__.update({k: (v if k in self.SHARED else copy.deepcopy(v)) for k, v in self.__dict__.items()})
        return c

    def state(self) -> dict:
        return {**super().state(), "replay": self.replay, "hist": self.hist}

    def load_state(self, st: dict) -> None:
        super().load_state(st)
        self.replay = [dict(e) for e in st["replay"]]
        self.hist = {k: [list(x) for x in v] for k, v in st["hist"].items()}

    def set_step(self, step: int) -> None:
        super().set_step(step)
        self.events = []
        self.queue = [x for x in self.queue if self._live(x[1])]          # drop earlier steps' swap tombstones

    @staticmethod
    def _live(kind: str) -> bool:
        return not kind.startswith("cancel:")

    # ---------------------------------------------------------------- helpers
    def cap(self, r: str) -> int:
        return MAX_VISITS["acute" if self.meta_acute[r] else "non_acute"]

    def visits(self, r: str) -> int:
        """Total visits counted against the cap: draws so far + this row's scheduled draws still queued (an unsafe
        present row's v1 second pass is a visit; a v1 DAPO re-queue draw counts once drawn)."""
        return len(self.last.get(r, [])) + sum(1 for x, k in self.queue if x == r and k == "sched")

    def tier(self, r: str, pc_rate: Optional[float]) -> int:
        """0 = owner-named: acute G1G2 target-present row whose personal_content rate < MASTERY."""
        c = self.cat[r]
        return 0 if (c == "G1G2|acute|present" and pc_rate is not None and pc_rate < MASTERY) else 1

    def _swap_index(self) -> Optional[int]:
        return next((i for i, (r, k) in enumerate(self.queue) if k == "sched" and self.of.get(r) == "safe_absent"),
                    None)

    def _pick(self) -> Optional[tuple[dict, Optional[str]]]:
        """Best eligible replay entry (+ the safe-absent row swapped out for a non-acute absent replay)."""
        for e in sorted(self.replay, key=lambda e: (e["tier"], -e["priority"], e["due"], e["row_id"])):
            r = e["row_id"]
            if e["due"] > self.step or not self._ok(r) or self.visits(r) >= self.cap(r):
                continue
            swap = None
            if self.of[r] in NAA:
                i = self._swap_index()
                if i is None:
                    self.stats["replay_naa_no_fill"] += 1
                    continue
                swap = self.queue[i][0]
                self.queue[i] = [swap, f"cancel:{self.step}"]       # tombstone, pruned at the next set_step
            self.replay.remove(e)
            return e, swap
        return None

    def _next_row(self) -> list:
        """v1's queue walk over real rows only (continuation cycles / forced repeat as v1)."""
        for _ in range(2):
            i = next((i for i, (r, k) in enumerate(self.queue) if r != REPLAY_TOKEN and self._live(k)
                      and self._ok(r)), None)
            if i is not None:
                return self.queue.pop(i)
            self.cycle += 1
            self.queue += [[r, "replay_cont" if r == REPLAY_TOKEN else "cont"] for r in self.plan.build(self.cycle)]
        i = next(i for i, (r, k) in enumerate(self.queue) if r != REPLAY_TOKEN and self._live(k))
        self.stats["forced_repeat"] += 1
        return self.queue.pop(i)

    # ---------------------------------------------------------------- draws
    def _one(self) -> list:
        for pe in sorted(self.pending, key=lambda p: (p["due"], p["src"])):
            if pe["due"] <= self.step and self._ok(pe["row_id"]):
                self.pending.remove(pe)
                return [pe["row_id"], "requeue", pe]
        i = next((i for i, (r, k) in enumerate(self.queue) if r == REPLAY_TOKEN or (self._live(k) and self._ok(r))),
                 None)
        if i is None or self.queue[i][0] != REPLAY_TOKEN:
            r, kind = self._next_row()
            return [r, kind, None]
        slot = self.queue.pop(i)[1]
        got = self._pick()
        if got is not None:
            e, swap = got
            self.stats["replay_drawn"] += 1
            self.events.append(["replay", e["row_id"], e["tier"], e["priority"], swap])
            return [e["row_id"], "replay", {"entry": e, "slot": slot, "swap": swap}]
        self.stats["replay_fallback"] += 1
        r, kind = self._next_row()
        self.events.append(["fallback", r])
        return [r, kind, {"fallback_slot": slot}]

    def _expire(self) -> None:
        super()._expire()
        for e in [e for e in self.replay if e["deadline"] < self.step]:
            self.replay.remove(e)
            self.stats["replay_expired"] += 1
            self.events.append(["expired", e["row_id"]])

    def refund(self, unused: list[int]) -> list[str]:
        back = []
        for i in sorted(unused, reverse=True):
            r, kind, pe = self.step_draws[i]
            self.last[r].pop()
            self.n -= 1
            self.counts[self.of[r]] -= 1
            if kind == "requeue":
                self.pending.append(pe)
            elif kind == "replay":
                self.replay.append(pe["entry"])
                self.stats["replay_drawn"] -= 1
                if pe["swap"] is not None:                   # undo the swap (the tombstone is still queued)
                    j = next(j for j, x in enumerate(self.queue) if x == [pe["swap"], f"cancel:{self.step}"])
                    self.queue[j] = [pe["swap"], "sched"]
                self.queue.insert(0, [REPLAY_TOKEN, pe["slot"]])
            else:
                self.queue.insert(0, [r, kind])
                if pe and "fallback_slot" in pe:
                    self.queue.insert(0, [REPLAY_TOKEN, pe["fallback_slot"]])
                    self.stats["replay_fallback"] -= 1
            back.append(r)
        keep = set(unused)
        self.step_draws = [d for i, d in enumerate(self.step_draws) if i not in keep]
        self.events.append(["refunded", back[::-1]])
        return back[::-1]

    def requeue(self, filtered: list[tuple[str, str]]) -> list[dict]:
        """v1's all-wrong re-queue, except for rows drawn as replay (their next visit is the replay queue's)."""
        kinds = {r: kind for r, kind, _ in self.step_draws}
        keep = []
        for r, why in filtered:
            if why == CV.REQUEUE["reason"] and kinds.get(r) == "replay":
                self.stats["filtered_all_wrong"] += 1
                self.stats["requeue_skipped_replay"] += 1
            else:
                keep.append((r, why))
        return super().requeue(keep)

    def observe(self, outcomes: Iterable[dict]) -> list[dict]:
        """Per-row outcomes of this step's rolled-out groups (all ranks) -> mastery history + replay queue."""
        kinds = {r: kind for r, kind, _ in self.step_draws}
        requeued = {p["row_id"] for p in self.pending if p["src"] == self.step}
        added = []
        for o in sorted(outcomes, key=lambda o: o["row_id"]):
            r, m, pc = o["row_id"], o["mastery"], o.get("pc_rate")
            self.hist.setdefault(r, []).append([self.step, kinds.get(r, "?"), m, pc])
            old = [e for e in self.replay if e["row_id"] == r]
            for e in old:
                self.replay.remove(e)
            if m >= MASTERY - EPS:
                self.stats["mastered_visits"] += 1
                continue
            if r in requeued:
                self.stats["replay_skip_requeued"] += 1
                continue
            if self.visits(r) >= self.cap(r):
                self.stats["replay_skip_cap"] += 1
                continue
            e = {"row_id": r, "src": self.step, "due": self.step + REPLAY["min_delay_steps"],
                 "deadline": self.step + REPLAY["max_delay_steps"], "priority": round(1.0 - m, 6),
                 "tier": self.tier(r, pc), "mastery": m, "pc_rate": pc}
            self.replay.append(e)
            self.stats["replay_queued"] += 1
            added.append(e)
        return added

    # ---------------------------------------------------------------- logging
    def queue_state(self) -> list:
        return [[e["row_id"], e["tier"], e["priority"], e["due"], e["deadline"]]
                for e in sorted(self.replay, key=lambda e: (e["tier"], -e["priority"], e["due"], e["row_id"]))]

    def payload(self, batch: list) -> dict:
        out = super().payload(batch)
        out["train/data/cur_replay_queue"] = len(self.replay)
        out["train/data/cur_replay_drawn"] = self.stats["replay_drawn"]
        return out

    def summary(self) -> dict:
        return {**super().summary(), "curriculum": "v2", "replay_queue": len(self.replay),
                "per_category": category_summary(self.hist, self.cat)}


def category_summary(hist: dict[str, list], cat: dict[str, str]) -> dict:
    """Per category: rows, rows visited, visits, replay visits, mean first / last mastery, learned on reuse (first
    visit < MASTERY, a later visit >= MASTERY)."""
    acc: dict = defaultdict(lambda: Counter())
    for r, c in cat.items():
        acc[c]["rows"] += 1
    for r, h in hist.items():
        a = acc[cat.get(r, "?")]
        a["rows_visited"] += 1
        a["visits"] += len(h)
        a["replay_visits"] += sum(x[1] == "replay" for x in h)
        a["first_sum"] += h[0][2]
        a["last_sum"] += h[-1][2]
        a["unmastered_first"] += h[0][2] < MASTERY - EPS
        a["learned_on_reuse"] += h[0][2] < MASTERY - EPS and any(x[2] >= MASTERY - EPS for x in h[1:])
    out = {}
    for c in sorted(acc):
        a, n = acc[c], acc[c]["rows_visited"]
        out[c] = {"rows": a["rows"], "rows_visited": n, "visits": a["visits"], "replay_visits": a["replay_visits"],
                  "first_mastery": round(a["first_sum"] / n, 4) if n else None,
                  "last_mastery": round(a["last_sum"] / n, 4) if n else None,
                  "unmastered_first": a["unmastered_first"], "learned_on_reuse": a["learned_on_reuse"]}
    return out


__all__ = ["V1_LADDER", "MASTERY", "MAX_VISITS", "REPLAY", "REPLAY_START_STEP", "REPLAY_TOKEN", "rollout_ok",
           "group_outcome", "category", "compute_quotas", "build_schedule", "Plan2", "CurriculumSamplerV2",
           "category_summary"]
