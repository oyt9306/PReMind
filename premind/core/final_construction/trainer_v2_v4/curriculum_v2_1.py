"""Mastery-guided replay curriculum v2.1 = curriculum v2 with every EMITTED draw gated (Codex review K, 2026-10-15,
blockers 1 + 2). NEW file: curriculum_v1.py / curriculum_v2.py are imported read-only, never edited. Pure Python.

Plan (owner 2026-10-15, v14: ~330 acute rows): ``Plan21`` = every acute row ONCE + REPLAY_BUDGET replay placeholders
  (default 80, ``--replay-budget``, pinned) + v1's non-acute rule derived from the data (``compute_quotas21``):
  non-acute absent slots = absent_share x n_slots (unsafe absent first, safe absent fill); unsafe present x1 when the
  plan is acute-heavy (acute x1 + budget > ACUTE_HEAVY_FRAC = 35 % of the slots: keeps safe exposure, owner
  2026-10-15 after CUR's rising safe-row false alarms), else x2 if it fits (else x1); safe present = the remainder; CurriculumConflict only if acute x1 + budget + absent + unsafe present x1 >
  n_slots. Interleaving / gap / placeholder pacing = curriculum_v2.build_schedule. (``Plan2``, v2's plan, still works.)
  Acute draws beyond the first pass (replay, re-queue, fill) are capped GLOBALLY at the budget (predicate (e)), so
  realized acute draws <= n_acute + REPLAY_BUDGET; unused replay slots fall back to scheduled / non-acute draws.
Runtime differences (CURRICULUM_V2_1.md):
  * ONE eligibility predicate ``_elig`` for every draw kind (sched, replay, requeue, fill; a fallback placeholder draws
    a sched / fill row through the same predicate):
      (a) never a mastered acute row (any observed group mastery >= MASTERY)
      (b) visit cap: draws so far + this row's sched items still queued (+1 unless the draw IS that sched item)
          <= cap; cap = 3 for acute rows, max(1, cycle-0 scheduled visits) for non-acute rows (the inherited unsafe
          present x2 pass is kept, nothing beyond it; a v1 DAPO re-queue counts as a visit)
      (c) the 10-step repeat gap (v1)
      (d) realized non-acute-absent share: with n = emitted draws so far net of refunds (every stratum, acute
          included; this is the denominator) and a = their non-acute-absent draws, a non-acute-absent row needs
          a + 1 <= S (n + 1) + SHARE_TOL and any other row (a replay placeholder counts as other) needs
          a >= S (n + 1) - SHARE_TOL (S = the plan's absent share, SHARE_TOL = 2 draws).
      (e) an acute row's repeat draw (not its scheduled first pass) needs acute repeats so far < replay budget.
  * No continuation cycles. When the queue has no eligible item, ``fill``: the least-visited eligible unmastered row
    (observed mastery < MASTERY, not pending re-queue / replay), else a fresh (never drawn, not queued) non-acute row,
    both in a fixed seeded order. Only if nothing is eligible at all: least-visited non-acute row with the cap relaxed
    (stats relax_cap), then also the share (relax_share), then also the gap (relax_gap). Never a mastered acute row.
  * No safe-absent swap (the share rule (d) replaces it).
  * Mastery (``rollout_ok``): v2's criterion AND withhold full when applicable on every row; on acute rows where
    withhold applies (acute gt_absent) also no wrong_concept / distractor penalty (the F rule acute-absent clarify
    accept: hazard specific & not contraindicated AND correct withhold).
  * Stuck rows (owner 2026-10-15, noisy / unlearnable rows): an acute row with group mastery 0 on 2 visits is never
    drawn again except its scheduled first pass (frees the replay slot) and is reported once (``new_suspects`` ->
    main_v4hcur21 appends it to OUT.curriculum/suspect_rows.jsonl: row_id, image, visits, per-visit mastery, component
    means, hazard states / reasons when the ledger has them). Image suspects (``image_suspects``, logging only, no
    exclusion): an image (source_image_sha256) with >= IMG_MIN_ROWS distinct visited rows whose mean row mastery is
    < IMG_MAX_MASTERY while the run's overall acute mastery (mean over acute visits) is >= IMG_RUN_MASTERY.
  * Resume: a truncated (partially written) LAST line of the curriculum log is skipped with a warning; any other
    malformed line refuses.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from trainer_v2_v4 import curriculum_v1 as CV
from trainer_v2_v4 import curriculum_v2 as C2

logger = logging.getLogger(__name__)

MASTERY, MAX_VISITS, REPLAY, REPLAY_TOKEN, NAA, EPS = C2.MASTERY, C2.MAX_VISITS, C2.REPLAY, C2.REPLAY_TOKEN, C2.NAA, C2.EPS
STUCK_ZERO_VISITS = 2         # acute row with mastery 0 on this many visits -> stuck (no further replay)
IMG_MIN_ROWS, IMG_MAX_MASTERY, IMG_RUN_MASTERY = 3, 0.1, 0.3
HZ_KEYS = ("state", "contraindicated", "backend", "reason", "reasons", "judge_reason")
SHARE_TOL = 2.0               # draws; realized |a - S n| bound at draw time
PENALTIES_ACUTE_WITHHOLD = ("wrong_concept", "distractor")
ACUTE_HEAVY_FRAC = 0.35      # acute x1 + replay budget above this share of the slots -> unsafe present x1 (pinned)
REPLAY_BUDGET = 80             # owner 2026-10-15 (v14); overridable via --replay-budget, pinned
Plan2, category, category_summary = C2.Plan2, C2.category, C2.category_summary


# ----------------------------------------------------------------------------- plan (owner 2026-10-15, v14)
def compute_quotas21(counts: dict[str, int], n_slots: int, absent_share: float, replay_budget: int) -> dict:
    """Acute x1 + replay_budget placeholders + v1's non-acute rule (see the module docstring)."""
    q = absent_share * n_slots
    if abs(q - round(q)) > 1e-9:
        raise CV.CurriculumConflict(f"REFUSED: absent share {absent_share} x {n_slots} slots is not an integer")
    n_abs, bad = int(round(q)), []
    if replay_budget < 0:
        bad.append(f"replay budget {replay_budget} < 0")
    ua = counts["unsafe_absent"]
    if ua > n_abs:
        bad.append(f"non-acute unsafe absent rows ({ua}) exceed the absent slots ({n_abs})")
    sa = n_abs - ua
    if sa > counts["safe_absent"]:
        bad.append(f"safe absent fill {sa} > {counts['safe_absent']} rows (without replacement)")
    acute = counts["acute_present"] + counts["acute_absent"]
    up_pass, sp = None, None
    heavy = acute + replay_budget > ACUTE_HEAVY_FRAC * n_slots
    for k in ((1,) if heavy else (2, 1)):
        rest = n_slots - acute - replay_budget - n_abs - k * counts["unsafe_present"]
        if rest >= 0:
            up_pass, sp = k, rest
            break
    if up_pass is None:
        bad.append(f"acute x1 ({acute}) + replay {replay_budget} + {n_abs} absent + unsafe present x1 exceed "
                   f"{n_slots} slots")
    elif sp > counts["safe_present"]:
        bad.append(f"safe present remainder {sp} > {counts['safe_present']} rows (without replacement)")
    if bad:
        raise CV.CurriculumConflict("REFUSED: curriculum v2.1 conflict: " + "; ".join(bad) + f" (counts {counts})")
    slots = {"acute_present": counts["acute_present"], "acute_absent": counts["acute_absent"],
             "unsafe_present": up_pass * counts["unsafe_present"], "unsafe_absent": ua, "safe_present": sp,
             "safe_absent": sa, "replay": replay_budget}
    passes = {"acute_present": 1, "acute_absent": 1, "unsafe_present": up_pass, "unsafe_absent": 1,
              "safe_present": None, "safe_absent": None, "replay": None}
    return {"n_slots": n_slots, "absent_share": absent_share, "n_absent_slots": n_abs, "replay_budget": replay_budget,
            "acute_heavy": heavy, "acute_heavy_frac": ACUTE_HEAVY_FRAC, "unsafe_present_passes": up_pass, "counts": dict(counts), "passes": passes, "slots": slots}


class Plan21(C2.Plan2):
    """Plan2 with compute_quotas21 (acute x1 + replay budget); v2's interleaving."""

    def __init__(self, items: list, acute_ids: set, steps: int, p_global: int, absent_share: float, seed: int,
                 replay_budget: int = REPLAY_BUDGET):
        self.strata = CV.strata_of(items, acute_ids)
        self.counts = {s: len(v) for s, v in self.strata.items()}
        self.steps, self.p_global, self.seed = steps, p_global, seed
        self.quotas = compute_quotas21(self.counts, steps * p_global, absent_share, replay_budget)
        self.schedule = self.build(0)
        self.sha = CV.schedule_sha(self.schedule)

    def report(self) -> dict:
        return {**super().report(), "curriculum": "v2.1", "replay_budget": self.quotas["replay_budget"]}

    def check(self, release_ids: set) -> dict:
        """coverage (acute x1, unsafe x passes), exact non-acute absent slots, no repeat within 10 steps, determinism,
        release ids, replay slots == budget and total == n_slots (CurriculumCheckFailed lists every failure)."""
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
            if r != REPLAY_TOKEN:
                if r in last and j // self.p_global - last[r] < CV.MIN_GAP_STEPS:
                    rep.append(r)
                last[r] = j // self.p_global
        if rep:
            fail["3_no_repeat_10"] = f"{len(rep)} repeats, e.g. {rep[:5]}"
        if CV.schedule_sha(self.build(0)) != self.sha:
            fail["4_deterministic"] = "a rebuild gives another schedule"
        if set(of) != release_ids or set(self.schedule) - set(of) - {REPLAY_TOKEN}:
            fail["5_release_ids"] = f"strata/release mismatch {len(set(of) ^ release_ids)}"
        if len(self.schedule) != q["n_slots"] or cnt[REPLAY_TOKEN] != q["replay_budget"]:
            fail["6_replay_budget"] = f"replay slots {cnt[REPLAY_TOKEN]} != budget {q['replay_budget']}"
        if fail:
            raise CV.CurriculumCheckFailed("CURRICULUM V2.1 SANITY CHECK FAILED: " + json.dumps(fail))
        return {k: "PASS" for k in ("1_coverage", "2_absent_share", "3_no_repeat_10", "4_deterministic",
                                    "5_release_ids", "6_replay_budget")}


# ----------------------------------------------------------------------------- mastery criterion (blocker 2)
def _no_penalty(comps: dict, name: str) -> Optional[bool]:
    c = comps.get(name) or {}
    if not c.get("applicable") or c.get("value") is None:
        return None
    return c["value"] >= -EPS


def rollout_ok(ledger: dict, acute: bool, w: Optional[dict] = None) -> dict:
    """Is one scored rollout FULLY correct? -> {"ok", "pc"}. Mastery = grounded personal response / withhold AND hazard
    handling:
      every row:  not degenerate, action full, personal_content full when applicable, withhold full when applicable
      acute rows with a hazard state: + state specific and not contraindicated; where withhold applies (acute
                  gt_absent) + no wrong_concept / distractor penalty
      other rows (or acute without a hazard state): + risk full when applicable."""
    if w is None:
        from trainer_v2 import reward_v5 as R5
        w = R5.W
    comps = ledger.get("components") or {}
    if (comps.get("degenerate") or {}).get("applicable"):
        return {"ok": False, "pc": False if (comps.get("personal_content") or {}).get("applicable") else None}
    pc = C2._full(comps, "personal_content", w)
    wh = C2._full(comps, "withhold", w)
    checks = [C2._full(comps, "action", w) is True, pc is not False, wh is not False]
    hz = ledger.get("v4h") or {}
    if acute and hz.get("acute") and "state" in hz:
        checks.append(hz["state"] == "specific" and not hz.get("contraindicated"))
        if wh is not None:
            checks += [_no_penalty(comps, p) is not False for p in PENALTIES_ACUTE_WITHHOLD]
    else:
        checks.append(C2._full(comps, "risk", w) is not False)
    return {"ok": all(checks), "pc": pc}


def diagnostics(ledgers: list[dict]) -> dict:
    """Component means (applicable values) and hazard states / reasons of one group (suspect-row report)."""
    vals: dict = {}
    hz: list = []
    for led in ledgers:
        for k, c in (led.get("components") or {}).items():
            if c.get("applicable") and isinstance(c.get("value"), (int, float)):
                vals.setdefault(k, []).append(float(c["value"]))
        v = led.get("v4h") or {}
        if v.get("acute"):
            hz.append({k: (str(v[k])[:300] if k in ("reason", "reasons", "judge_reason") else v[k])
                       for k in HZ_KEYS if k in v})
    return {"component_means": {k: round(sum(x) / len(x), 6) for k, x in sorted(vals.items())},
            "hazard_states": dict(Counter(str(h.get("state")) for h in hz)),
            "contraindicated": sum(bool(h.get("contraindicated")) for h in hz),
            "hazard_reasons": [h for h in hz if any(k in h for k in ("reason", "reasons", "judge_reason"))][:3]}


def group_outcome(row_id: str, ledgers: list[dict], acute: bool, w: Optional[dict] = None) -> dict:
    res = [rollout_ok(led, acute, w) for led in ledgers]
    pcs = [r["pc"] for r in res if r["pc"] is not None]
    out = {"row_id": row_id, "mastery": round(sum(r["ok"] for r in res) / len(res), 6) if res else 0.0,
           "pc_rate": round(sum(pcs) / len(pcs), 6) if pcs else None, "n": len(res)}
    if acute:
        out["diag"] = diagnostics(ledgers)
    return out


def image_of(row: dict) -> Optional[str]:
    ev = row.get("evaluator") or {}
    return row.get("source_image_sha256") or ev.get("source_image_sha256") or row.get("image") or None


# ----------------------------------------------------------------------------- runtime (blocker 1)
class CurriculumSamplerV21(C2.CurriculumSamplerV2):
    """v2 sampler whose every emitted draw passes ``_elig`` (module docstring). Call order per step as v2:
    set_step -> draws -> refund -> requeue(filtered) -> observe(outcomes)."""

    SHARED = C2.CurriculumSamplerV2.SHARED + ("planned", "fill_rank", "share", "budget", "image")

    def __init__(self, items: list, plan: C2.Plan2, log_path: Optional[Path] = None) -> None:
        super().__init__(items, plan, log_path)
        self.planned = Counter(r for r in plan.schedule if r != REPLAY_TOKEN)
        self.fill_rank = {r: hashlib.sha256(f"curriculum_v2_1|{plan.seed}|{r}".encode()).hexdigest() for r in self.of}
        self.share = float(plan.quotas["absent_share"])
        self.budget = int(plan.quotas.get("replay_budget", plan.quotas["slots"]["replay"]))
        self._xa = 0
        self.image = {it.row_id: image_of(it.row) for it in items}
        self.stuck: list[str] = []                         # sorted, logged in state
        self.diag: dict[str, list] = {}                    # acute rows: per-visit diagnostics (logged in state)
        self.new_suspects: list[dict] = []

    # ---------------------------------------------------------------- predicates
    def cap(self, r: str) -> int:
        return MAX_VISITS["acute"] if self.meta_acute[r] else max(MAX_VISITS["non_acute"], self.planned[r])

    def draws(self, r: str) -> int:
        return len(self.last.get(r, []))

    def _qc(self) -> Counter:
        return Counter(r for r, k in self.queue if k == "sched")

    def visits(self, r: str) -> int:
        return self.draws(r) + self._qc()[r]

    def mastered(self, r: str) -> bool:
        return any(x[2] is not None and x[2] >= MASTERY - EPS for x in self.hist.get(r, []))

    @property
    def n_naa(self) -> int:
        return self.counts["unsafe_absent"] + self.counts["safe_absent"]

    def _share_ok(self, naa: bool) -> bool:
        t = self.share * (self.n + 1)
        return self.n_naa + 1 <= t + SHARE_TOL + EPS if naa else self.n_naa >= t - SHARE_TOL - EPS

    def _elig(self, r: str, kind: str, qc: Counter, *, cap: bool = True, share: bool = True, gap: bool = True) -> bool:
        if self.meta_acute[r] and self.mastered(r):
            return False
        if gap and not self._ok(r):
            return False
        if cap and self.draws(r) + qc[r] + (0 if kind == "sched" else 1) > self.cap(r):
            return False
        if kind != "sched" and r in self.stuck:
            return False                                   # stuck (mastery 0 on 2 visits): no further replay
        if self.meta_acute[r] and kind != "sched" and self._xa >= self.budget:
            return False                                   # (e) acute repeats are capped at the replay budget
        return not share or self._share_ok(self.of[r] in NAA)

    # ---------------------------------------------------------------- draws
    def _pick(self, qc: Counter) -> Optional[tuple[dict, int]]:
        """Best eligible due replay entry (smallest tier, -priority, due, row_id) and its list index."""
        for e in sorted(self.replay, key=lambda e: (e["tier"], -e["priority"], e["due"], e["row_id"])):
            if e["due"] <= self.step and self._elig(e["row_id"], "replay", qc):
                ri = self.replay.index(e)
                self.replay.pop(ri)
                return e, ri
        return None

    def _fill(self, qc: Counter) -> Optional[str]:
        busy = {p["row_id"] for p in self.pending} | {e["row_id"] for e in self.replay}
        unm = [r for r in self.of if r in self.hist and r not in busy and not self.mastered(r)
               and self._elig(r, "fill", qc)]
        if unm:
            return min(unm, key=lambda r: (self.draws(r), self.hist[r][-1][2], self.fill_rank[r]))
        fresh = [r for r in self.of if not self.meta_acute[r] and not self.draws(r) and not qc[r]
                 and self._elig(r, "fill", qc)]
        return min(fresh, key=self.fill_rank.__getitem__) if fresh else None

    def _real(self, qc: Counter) -> Optional[list]:
        """First eligible scheduled row, else a fill row: [row, kind, queue index (None for fill)] or None."""
        i = next((i for i, (r, k) in enumerate(self.queue) if r != REPLAY_TOKEN and k == "sched"
                  and self._elig(r, "sched", qc)), None)
        if i is not None:
            return self.queue.pop(i) + [i]
        r = self._fill(qc)
        return None if r is None else [r, "fill", None]

    def _relaxed(self, qc: Counter) -> Optional[list]:
        """Nothing eligible: least-visited non-acute row, relaxing cap, then share, then gap (counted)."""
        for name, kw in (("relax_cap", {"cap": False}), ("relax_share", {"cap": False, "share": False}),
                         ("relax_gap", {"cap": False, "share": False, "gap": False})):
            c = [r for r in self.of if not self.meta_acute[r] and self._elig(r, "fill", qc, **kw)]
            if c:
                self.stats[name] += 1
                r = min(c, key=lambda r: (self.draws(r), self.fill_rank[r]))
                self.events.append([name, r])
                return [r, "fill", {"relaxed": name}]
        return None

    def acute_repeats(self) -> int:
        return sum(max(0, len(v) - 1) for r, v in self.last.items() if self.meta_acute[r])

    def _one(self) -> list:
        qc = self._qc()
        self._xa = self.acute_repeats()
        for pe in sorted(self.pending, key=lambda p: (p["due"], p["src"])):
            if pe["due"] <= self.step and self._elig(pe["row_id"], "requeue", qc):
                self.pending.remove(pe)
                return [pe["row_id"], "requeue", pe]
        for i, (r, k) in enumerate(self.queue):
            if r != REPLAY_TOKEN:
                if k == "sched" and self._elig(r, "sched", qc):
                    return self.queue.pop(i) + [{"qi": i}]
                continue
            if not self._share_ok(False):                    # a placeholder counts as a non-absent draw
                continue
            got = self._pick(qc)
            if got is not None:
                e, ri = got
                slot = self.queue.pop(i)[1]
                self.stats["replay_drawn"] += 1
                self.events.append(["replay", e["row_id"], e["tier"], e["priority"]])
                return [e["row_id"], "replay", {"entry": e, "slot": slot, "pi": i, "ri": ri}]
            got = self._real(qc)
            if got is None:
                break
            pi = next(j for j, x in enumerate(self.queue) if x[0] == REPLAY_TOKEN)
            slot = self.queue.pop(pi)[1]
            self.stats["replay_fallback"] += 1
            self.events.append(["fallback", got[0], got[1]])
            return [got[0], got[1], {"fallback_slot": slot, "pi": pi, "qi": got[2]}]
        got = self._real(qc)
        if got is not None:
            if got[1] == "fill":
                self.stats["fill"] += 1
            return got[:2] + [None if got[2] is None else {"qi": got[2]}]
        got = self._relaxed(qc)
        if got is None:
            raise CV.CurriculumConflict("REFUSED: curriculum v2.1: no non-mastered row can be drawn")
        self.stats["fill"] += 1
        return got

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
                self.replay.insert(pe["ri"], pe["entry"])
                self.stats["replay_drawn"] -= 1
                self.queue.insert(pe["pi"], [REPLAY_TOKEN, pe["slot"]])
            else:                                            # exact positions, undone in reverse draw order
                pe = pe or {}
                if "fallback_slot" in pe:
                    self.queue.insert(pe["pi"], [REPLAY_TOKEN, pe["fallback_slot"]])
                    self.stats["replay_fallback"] -= 1
                elif kind == "fill":
                    self.stats["fill"] -= 1
                if kind == "sched":
                    self.queue.insert(pe["qi"], [r, kind])
                if "relaxed" in pe:
                    self.stats[pe["relaxed"]] -= 1
            back.append(r)
        keep = set(unused)
        self.step_draws = [d for i, d in enumerate(self.step_draws) if i not in keep]
        self.events.append(["refunded", back[::-1]])
        return back[::-1]

    def requeue(self, filtered: list[tuple[str, str]]) -> list[dict]:
        """v2's rule, plus: a row already at its visit cap (or a mastered acute row) is not re-queued (counted)."""
        keep = []
        for r, why in filtered:
            if why == CV.REQUEUE["reason"] and (self.visits(r) >= self.cap(r)
                                                or (self.meta_acute[r] and self.mastered(r))):
                self.stats["filtered_all_wrong"] += 1
                self.stats["requeue_skip_cap"] += 1
            else:
                keep.append((r, why))
        return super().requeue(keep)

    # ---------------------------------------------------------------- stuck rows / suspects
    def set_step(self, step: int) -> None:
        super().set_step(step)
        self.new_suspects = []

    def state(self) -> dict:
        return {**super().state(), "stuck": self.stuck, "diag": self.diag}

    def load_state(self, st: dict) -> None:
        super().load_state(st)
        self.stuck = list(st.get("stuck", []))
        self.diag = {k: [dict(x) for x in v] for k, v in (st.get("diag") or {}).items()}

    def observe(self, outcomes: Iterable[dict]) -> list[dict]:
        outcomes = sorted(outcomes, key=lambda o: o["row_id"])
        for o in outcomes:
            if self.meta_acute.get(o["row_id"]) and "diag" in o:
                self.diag.setdefault(o["row_id"], []).append({"step": self.step, "mastery": o["mastery"], **o["diag"]})
        added = super().observe([{k: v for k, v in o.items() if k != "diag"} for o in outcomes])
        for o in outcomes:
            r = o["row_id"]
            if not self.meta_acute.get(r) or r in self.stuck:
                continue
            zeros = sum(1 for x in self.hist.get(r, []) if x[2] is not None and x[2] <= EPS)
            if zeros >= STUCK_ZERO_VISITS:
                self.stuck = sorted(self.stuck + [r])
                for e in [e for e in self.replay if e["row_id"] == r]:
                    self.replay.remove(e)
                added = [e for e in added if e["row_id"] != r]
                self.stats["stuck_rows"] += 1
                self.events.append(["stuck", r])
                self.new_suspects.append(self.suspect_record(r))
        return added

    def suspect_record(self, r: str) -> dict:
        d = self.diag.get(r, [])
        return {"row_id": r, "image": self.image.get(r), "category": self.cat.get(r), "step": self.step,
                "visits": [[x[0], x[1]] for x in self.hist.get(r, [])],
                "per_visit_mastery": [x[2] for x in self.hist.get(r, [])],
                "component_means": [x.get("component_means") for x in d],
                "hazard_states": [x.get("hazard_states") for x in d],
                "contraindicated": [x.get("contraindicated") for x in d],
                "hazard_reasons": [x.get("hazard_reasons") for x in d if x.get("hazard_reasons")]}

    def image_suspects(self) -> dict:
        """Logging only: images whose visited rows have mean mastery < IMG_MAX_MASTERY (>= IMG_MIN_ROWS rows) while
        the run's overall acute mastery is >= IMG_RUN_MASTERY."""
        acute_m = [x[2] for r, h in self.hist.items() if self.meta_acute.get(r) for x in h if x[2] is not None]
        run_m = sum(acute_m) / len(acute_m) if acute_m else None
        by: dict = {}
        for r, h in self.hist.items():
            ms = [x[2] for x in h if x[2] is not None]
            if ms and self.image.get(r):
                by.setdefault(self.image[r], {})[r] = sum(ms) / len(ms)
        flagged = []
        if run_m is not None and run_m >= IMG_RUN_MASTERY:
            for img, rows in sorted(by.items()):
                mean = sum(rows.values()) / len(rows)
                if len(rows) >= IMG_MIN_ROWS and mean < IMG_MAX_MASTERY:
                    flagged.append({"image": img, "n_rows": len(rows), "mean_mastery": round(mean, 6),
                                    "rows": {k: round(v, 6) for k, v in sorted(rows.items())}})
        return {"step": self.step, "run_acute_mastery": None if run_m is None else round(run_m, 6),
                "rule": {"min_rows": IMG_MIN_ROWS, "max_mean_mastery": IMG_MAX_MASTERY,
                         "min_run_acute_mastery": IMG_RUN_MASTERY}, "flagged": flagged}

    # ---------------------------------------------------------------- resume
    def restore(self, cursor: int) -> None:
        hit, self.resume_truncated_skipped = None, False
        if self.log_path and Path(self.log_path).is_file():
            lines = Path(self.log_path).read_text(encoding="utf-8").splitlines()
            for j, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    if j == len(lines) - 1:
                        logger.warning("curriculum v2.1 resume: skipping a truncated last line of %s", self.log_path)
                        self.resume_truncated_skipped = True      # not part of the logged state
                        continue
                    raise SystemExit(f"REFUSED: curriculum resume: malformed line {j + 1} of {self.log_path}")
                if rec.get("cursor_after") == cursor and rec.get("schedule_sha256") == self.plan.sha:
                    hit = rec
        if hit is None:
            raise SystemExit(f"REFUSED: curriculum resume: no logged state with cursor {cursor} in {self.log_path}")
        self.load_state(hit["state"])

    def summary(self) -> dict:
        return {**super().summary(), "curriculum": "v2.1", "stuck_rows": len(self.stuck),
                "realized_non_acute_absent": [self.n_naa, self.n], "share_tol_draws": SHARE_TOL}


__all__ = ["STUCK_ZERO_VISITS", "diagnostics", "image_of", "REPLAY_BUDGET", "compute_quotas21", "Plan21", "MASTERY", "MAX_VISITS", "REPLAY", "REPLAY_TOKEN", "SHARE_TOL", "rollout_ok", "group_outcome", "Plan2",
           "category", "category_summary", "CurriculumSamplerV21"]
