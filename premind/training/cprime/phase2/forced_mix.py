"""Forced-call mixing + corrected-GT + P6 hooks for the isolated forced_train_v1 trial (CPU-testable, no I/O at import).

Everything is an in-process hook (same pattern as rl_from_sft_v1/pkg*/p2sft.py); no trainer file is edited.

1. FORCED CALL (per item = per DAPO group).  ``split_rollout`` wraps the ``rollout`` callable that
   trainer_v2.algo_v2.dynamic_rollout receives.  For each item of a batch it draws forced / unforced from a hash of
   (seed, row_id, occurrence).  Forced items are rolled out in ONE call of the original rollout with the trainer's own
   ``forced_stage1`` argument (Stage-1 text = TAG for all G samples of the item); unforced items in another call; the
   episodes are re-assembled in the original item order.  A group is exactly the G episodes of one item, so a group is
   forced or unforced as a whole (structural guarantee; test_groups_not_mixed).
2. NO STAGE-1 LOSS / NO CALL REWARD on forced episodes: ``mask_stage1`` sets env_mask to 0 on the Stage-1 tokens (so
   they are outside model_positions -> no policy loss, no KL, no DAPO token weight); ``adjust_ledger`` removes the
   ``decision`` component (correct-call credit, missed-call and unneeded-call terms) from the total.  All Stage-2 terms
   stay (they are additive in ``total``): adequacy/hazard terms, MIE, fact/citation, withhold, still-stay penalty.
3. P6: additive penalty -W6 on acute rows whose answer contains an unconditioned recovery-position / roll-onto-side
   instruction (no neck/spine/unresponsive/'if' condition in the same sentence).  Applied to forced and unforced.
4. CORRECTED GT: ``install_gt_clean`` wraps eval_v1.release_data.convert so ``gold_answer`` of the 159 edited training
   rows is replaced by the cleaned text (first_filter_v2_rule/joint_v3c GT_EDITS); refuses if the row's current text
   differs from the recorded ``before``.
Eligibility (never force on no-need / image-only G0 / neutral panic-calm rows): see ``eligibility``.
5. FACT BONUS (factbonus_v1, default OFF, w_f=0): +w_f on an episode whose shown target memory is fully used in the final answer
   (personal_content full, gold cited, no harm flag); see ``fact_bonus_check``.  Not a penalty; no bonus -> ledger unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

TAG = "<memory_call>call_pdb</memory_call>"
SALT = "forced_mix_v1"
P_PRESENT = 0.40          # unsafe target-present (conflict incl.) memory-needed rows (owner 2026-11-03)
P_SAFE = 0.50             # SAFE target-present rows (personal-fact requests; where Fact matters most)
P_ABSENT = 0.40           # target-absent (withhold) memory-needed rows
W6 = 0.10                 # P6 weight (reward_ext.py recommended value; paper stillpen weights are unchanged)
EXCLUDE_SAFE = False      # coordinator 2026-11-03: safe target-present rows ARE eligible (only no-need / G0 / neutral are excluded)
NEUTRAL_STYLES = ("panic", "calm")

ROLL = re.compile(r"recovery position|\b(?:roll|turn|place|put|lay)\w*\b[^.;]{0,40}\b(?:on|onto)\s+(?:his|her|their|the)\s+side\b", re.I)
ROLL_COND = re.compile(r"\b(?:if|unless|only|once|when|whenever|provided)\b|\bneck\b|\bspine\b|\bspinal\b|\bunresponsive\b"
                       r"|\bnot (?:responding|responsive|breathing)\b|\bbreathing (?:but|normally)\b", re.I)
_ANS = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S | re.I)


# ----------------------------------------------------------------------------- pure helpers
def answer_text(stage2: str) -> str:
    m = _ANS.search(stage2 or "")
    return (m.group(1) if m else (stage2 or "")).strip()


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s]


def unconditioned_roll(text: str) -> bool:
    """True iff a sentence orders a recovery-position / roll-onto-side without any condition word in that sentence."""
    return any(ROLL.search(s) and not ROLL_COND.search(s) for s in sentences(text))


def eligibility(item: Any, exclude_safe: bool = EXCLUDE_SAFE) -> Optional[str]:
    """'present' | 'absent' for rows that may be forced, None otherwise (never forced)."""
    t, row = item.target, item.row
    if item.arm != "memory_call" or not t.get("memory_required"):
        return None                                           # passive arm / no-need rows
    if str(t.get("row_type") or row.get("row_type") or "").startswith("G0"):
        return None                                           # image-only G0 rows
    if (t.get("query_style") or row.get("query_style")) in NEUTRAL_STYLES:
        return None                                           # generic panic / calm queries
    if exclude_safe and t.get("risk_label") == "safe":
        return None
    return "absent" if t.get("withhold") else "present"


def draw(seed: int, row_id: str, occurrence: int) -> float:
    h = hashlib.sha256(f"{seed}|{SALT}|{row_id}|{occurrence}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def plan_batch(items: list, occ: Counter, seed: int, p_present: float = P_PRESENT, p_absent: float = P_ABSENT,
               exclude_safe: bool = EXCLUDE_SAFE, p_safe: float = P_SAFE) -> list[Optional[str]]:
    """Per item: None (unforced, ineligible), 'u:<kind>' (eligible, unforced) or 'f:<kind>' (forced)."""
    out: list[Optional[str]] = []
    for it in items:
        kind = eligibility(it, exclude_safe)
        if kind is None:
            out.append(None)
            continue
        k = occ[it.row_id]
        occ[it.row_id] += 1
        safe_present = kind == "present" and it.target.get("risk_label") == "safe"
        p = p_safe if safe_present else (p_present if kind == "present" else p_absent)
        out.append(("f:" if draw(seed, it.row_id, k) < p else "u:") + kind)
    return out


def mask_stage1(ep: Any) -> None:
    """Stage-1 tokens leave model_positions (no loss / KL / token weight); the sequence itself is unchanged.
    The trainer's rollout hook (patch._rollout) already computed ref_logp over ALL model tokens (Stage 1 first), so the
    first n_stage1 entries are dropped to keep len(ref_logp) == len(model_positions) (the 2026-11-03 crash)."""
    n = ep.n_stage1
    old_mask = tuple(ep.env_mask)
    if not 0 <= n <= len(old_mask):
        raise RuntimeError("Invalid Stage-1 token count")
    ref = getattr(ep, "ref_logp", None)
    removed = sum(old_mask[:n])
    if ref is not None:
        if len(ref) != sum(old_mask):
            raise RuntimeError(f"Reference/mask length mismatch before Stage-1 masking: {len(ref)} vs {sum(old_mask)}")
        # Stage 1 precedes all other model positions; keep the full input sequence.
        ep.ref_logp = ref[removed:]
    ep.env_mask = (0,) * n + old_mask[n:]
    if ref is not None and len(ep.ref_logp) != sum(ep.env_mask):
        raise RuntimeError("Reference/mask length mismatch after Stage-1 masking")
    ep.forced = True


def adjust_ledger(led: dict, forced: bool, w6: float = 0.0, acute: Optional[bool] = None,
                  answer: str = "", w_f: float = 0.0, item: Any = None) -> dict:
    """Ledger copy with the forced-episode decision term removed and the P6 penalty applied (both additive in total)."""
    out = {**led, "components": dict(led["components"])}
    total = float(led["total"])
    if forced:
        dec = led["components"].get("decision") or {}
        old = dec.get("value") if dec.get("applicable") and dec.get("value") is not None else 0.0
        out["components"]["decision"] = {"applicable": False, "value": None, "status": "forced_stage1:no_call_reward"}
        if "applicable_components" in out:
            out["applicable_components"] = [c for c in out["applicable_components"] if c != "decision"]
        total -= float(old)
        out["forced_stage1"] = {"removed_decision": float(old)}
    if acute is None:
        acute = bool((led.get("v4h") or {}).get("acute"))
    if w6 > 0 and acute:
        fired = unconditioned_roll(answer)
        out["p6"] = {"weight": w6, "fired": fired}
        if fired:
            total -= w6
    if w_f > 0:
        fb = fact_bonus_check(led, item, p6_fired=bool((out.get("p6") or {}).get("fired")), answer=answer)
        out["fact_bonus"] = {"weight": w_f, **fb}
        if fb["fired"]:
            total += w_f
    out["total"] = round(total, 10)
    return out


# ----------------------------------------------------------------------------- fact-in-answer bonus (factbonus_v1; OFF unless w_f > 0)
W_F = 0.15                # default weight of the bonus when enabled (factpc_v1: one trial value, not claimed optimal; factbonus_v1 was 0.10)
FACT_EPS = 1e-9
MIN_WORDS = 8             # anti-hack: no fact credit for an answer shorter than this (the only length rule)
ECHO_WORDS = 15           # log only: a fired answer with fewer words counts as "fact echo only"


def _cv(led: dict, name: str) -> Optional[float]:
    c = (led.get("components") or {}).get(name) or {}
    return c.get("value") if c.get("applicable") and c.get("value") is not None else None


def fact_bonus_check(led: dict, item: Any, p6_fired: bool = False, w_pc: Optional[float] = None, answer: str = "") -> dict:
    """Pure decision of the bonus on one scored ledger (reads only existing ledger keys; never raises on missing keys).
    eligible = target memory shown through a call (decision call_pdb, branch != not_called, forced or on-demand) on a memory-needed,
               non-withhold, non-G0, non-neutral row whose personal_content component is applicable (support patterns exist);
    fraction = personal_content / W_personal_content (logged only; the existing partial personal_content reward is untouched);
    fired    = eligible AND fraction == 1 (FULL use, as in factbonus_v1) AND gold all cited AND no critical miss / wrong_concept / contraindicated / hazard_guard /
               hazard_appropriate penalty / guard note / P6 hit / withhold-style answer AND answer has >= MIN_WORDS words (the only length rule);
    miss     = eligible AND fraction < 1 AND gold cited AND no other blocker (fact omitted though cited and not withheld): drives the targeted replay."""
    out = {"eligible": False, "fired": False, "fraction": 0.0, "miss": False, "words": len((answer or "").split()), "blocked_by": []}
    if item is None or eligibility(item) != "present":
        out["blocked_by"].append("row_not_eligible(no-need/G0/neutral/absent/withhold)")
        return out
    if led.get("branch") == "not_called" or led.get("decision") != "call_pdb":
        out["blocked_by"].append("target_memory_not_shown")
        return out
    pc = _cv(led, "personal_content")
    if pc is None or (led.get("components") or {}).get("degenerate", {}).get("applicable"):
        out["blocked_by"].append("no_personal_content_or_degenerate")
        return out
    out["eligible"] = True
    if w_pc is None:
        from trainer_v2 import reward_v5 as R5
        w_pc = R5.W["personal_content"]
    frac = max(0.0, min(1.0, pc / w_pc))
    out["fraction"] = frac
    gold, cited = set((led.get("target") or {}).get("gold") or []), set(led.get("cited") or [])
    parse, v4h, why = led.get("parse") or {}, led.get("v4h") or {}, out["blocked_by"]
    if not gold or not gold <= cited:
        why.append("gold_not_cited")
    if (_cv(led, "critical_miss") or 0.0) < 0:
        why.append("critical_miss")
    if (_cv(led, "wrong_concept") or 0.0) < 0 or parse.get("wrong_concept_fired"):
        why.append("wrong_concept")
    if v4h.get("contraindicated") or (_cv(led, "hazard_appropriate") or 0.0) < 0 or (_cv(led, "hazard_guard") or 0.0) < 0 or led.get("guard"):
        why.append("harm_flag")
    if p6_fired:
        why.append("p6")
    if parse.get("withholds"):
        why.append("withhold_answer")
    out["miss"] = frac < 1 - FACT_EPS and not why
    if frac < 1 - FACT_EPS:
        why.append("support_not_fully_hit_and_cited")
    elif out["words"] < MIN_WORDS:
        why.append("answer_lt_min_words")                    # anti-hack: a full-use answer shorter than MIN_WORDS gets no bonus
    out["fired"] = not why
    return out


# ----------------------------------------------------------------------------- stateful hook
class ForcedMix:
    def __init__(self, seed: int = 0, p_present: float = P_PRESENT, p_absent: float = P_ABSENT, w6: float = W6,
                 exclude_safe: bool = EXCLUDE_SAFE, log_path: Optional[Path] = None, p_safe: float = P_SAFE, w_f: float = 0.0):
        self.w_f = w_f
        self.seed, self.p_present, self.p_absent, self.w6 = seed, p_present, p_absent, w6
        self.exclude_safe, self.log_path, self.p_safe = exclude_safe, log_path, p_safe
        self.nstep = 0
        self.occ: Counter = Counter()
        self.reset()

    def reset(self) -> None:
        self.t: Counter = Counter()
        self.forced_ids: set[int] = set()
        self.forced_rows: list[str] = []

    def text(self, _item: Any = None) -> str:
        return TAG

    def split_rollout(self, rollout: Callable) -> Callable:
        def rollout_forced(items, images, backend, cfg, **kw):
            if kw.get("forced_stage1") is not None:                      # teacher-table path: untouched
                return rollout(items, images, backend, cfg, **kw)
            g = kw.get("num_generations") or cfg.num_generations
            plan = plan_batch(list(items), self.occ, self.seed, self.p_present, self.p_absent, self.exclude_safe, self.p_safe)
            for it, tag in zip(items, plan):
                self.t["items_" + ("inelig" if tag is None else tag)] += 1
            fidx = [i for i, p in enumerate(plan) if p and p.startswith("f:")]
            if not fidx:
                eps = rollout(items, images, backend, cfg, **kw)
                self._tally(items, plan, eps, g)
                return eps
            uidx = [i for i in range(len(items)) if i not in set(fidx)]
            f_eps = rollout([items[i] for i in fidx], [images[i] for i in fidx], backend, cfg,
                            **{**kw, "forced_stage1": self.text})
            u_eps = rollout([items[i] for i in uidx], [images[i] for i in uidx], backend, cfg, **kw) if uidx else []
            for e in f_eps:
                mask_stage1(e)
            for i in fidx:
                self.forced_ids.add(id(items[i]))
                self.forced_rows.append(items[i].row_id)
            by = {}
            for j, i in enumerate(fidx):
                by[i] = f_eps[j * g:(j + 1) * g]
            for j, i in enumerate(uidx):
                by[i] = u_eps[j * g:(j + 1) * g]
            eps = [e for i in range(len(items)) for e in by[i]]
            self._tally(items, plan, eps, g)
            return eps
        return rollout_forced

    def _tally(self, items, plan, eps, g) -> None:
        for j, tag in enumerate(plan):
            grp = eps[j * g:(j + 1) * g]
            if tag and tag.startswith("u:"):
                self.t["unforced_eligible_eps"] += len(grp)
                self.t["unforced_eligible_calls"] += sum(e.decision == "call_pdb" for e in grp)
            if tag and tag.startswith("f:"):
                self.t["forced_eps"] += len(grp)

    def wrap_score(self, score: Callable) -> Callable:
        def score_forced(ep, cfg):
            led = score(ep, cfg)
            forced = bool(getattr(ep, "forced", False))
            led2 = adjust_ledger(led, forced, self.w6, answer=answer_text(getattr(ep, "stage2_text", "")), w_f=self.w_f,
                                 item=getattr(ep, "item", None))
            if forced:
                self.t["scored_forced"] += 1
            if "fact_bonus" in led2:
                self.t["fact_eligible"] += int(led2["fact_bonus"]["eligible"])
                self.t["fact_fired"] += int(led2["fact_bonus"]["fired"])
                self.t["fact_eligible_forced"] += int(led2["fact_bonus"]["eligible"] and forced)
                self.t["fact_fired_forced"] += int(led2["fact_bonus"]["fired"] and forced)
                fb = led2["fact_bonus"]
                self.t["fact_fraction_sum_eligible"] += fb["fraction"] if fb["eligible"] else 0.0
                self.t["fact_miss"] += int(fb["miss"])
                self.t["fact_words_eligible"] += fb["words"] if fb["eligible"] else 0
                self.t["fact_words_fired"] += fb["words"] if fb["fired"] else 0
                self.t["fact_echo_fired"] += int(fb["fired"] and fb["words"] < ECHO_WORDS)
                self.t["fact_short_blocked"] += int("answer_lt_min_words" in fb["blocked_by"])
            if "fact_bonus" in led2 and led2["fact_bonus"]["fired"] and led2["fact_bonus"]["words"] < ECHO_WORDS and self.log_path is not None \
                    and self.t["echo_written"] < 50:                       # sample file for manual reading of fact-echo-only answers
                self.t["echo_written"] += 1
                ep_path = Path(str(self.log_path).replace("forced_mix_steps.jsonl", "fact_echo_samples.jsonl"))
                ep_path.parent.mkdir(parents=True, exist_ok=True)
                with ep_path.open("a") as fh:
                    fh.write(json.dumps({"row_id": getattr(getattr(ep, "item", None), "row_id", None), "forced": forced,
                                         "words": led2["fact_bonus"]["words"], "answer": answer_text(getattr(ep, "stage2_text", ""))}) + "\n")
            if "p6" in led2:
                self.t["p6_acute"] += 1
                self.t["p6_fired"] += int(led2["p6"]["fired"])
            return led2
        return score_forced

    def report(self, used_items: Optional[list] = None) -> dict:
        t = dict(self.t)
        kept_f = None if used_items is None else sum(id(it) in self.forced_ids for it in used_items)
        rep = {"drawn_items": {k[6:]: v for k, v in t.items() if k.startswith("items_")},
               "forced_episodes": t.get("forced_eps", 0), "scored_forced": t.get("scored_forced", 0),
               "unforced_eligible_episodes": t.get("unforced_eligible_eps", 0),
               "unforced_eligible_calls": t.get("unforced_eligible_calls", 0),
               "unforced_eligible_call_rate": (t["unforced_eligible_calls"] / t["unforced_eligible_eps"]
                                               if t.get("unforced_eligible_eps") else None),
               "kept_groups": None if used_items is None else len(used_items), "kept_forced_groups": kept_f,
               "p6_acute": t.get("p6_acute", 0), "p6_fired": t.get("p6_fired", 0), "forced_rows": list(self.forced_rows),
               "p_present": self.p_present, "p_absent": self.p_absent, "p_safe": self.p_safe, "w6": self.w6}
        if self.w_f > 0:
            rep.update({"fact_w": self.w_f, **{k: t.get(k, 0) for k in FACT_KEYS}})
            ne_f, nf = rep["fact_eligible"], rep["fact_fired"]
            rep.update({"fact_mean_fraction_eligible": rep["fact_fraction_sum_eligible"] / ne_f if ne_f else None,
                        "fact_mean_words_fired": rep["fact_words_fired"] / nf if nf else None,
                        "fact_echo_share_fired": rep["fact_echo_fired"] / nf if nf else None})
        n_f = sum(v for k, v in rep["drawn_items"].items() if k.startswith("f:"))
        n_e = sum(v for k, v in rep["drawn_items"].items() if k[:2] in ("f:", "u:"))
        rep["forced_share_of_eligible"] = (n_f / n_e) if n_e else None
        return rep


FACT_KEYS = ("fact_eligible", "fact_fired", "fact_eligible_forced", "fact_fired_forced", "fact_fraction_sum_eligible", "fact_miss",
             "fact_words_eligible", "fact_words_fired", "fact_echo_fired", "fact_short_blocked")


def fact_group_stats(details: list) -> dict:
    """Per-step group statistics over the KEPT DAPO groups (one detail per group = G episodes of one row, all forced or all on-demand):
    groups with >=1 eligible episode; bonus-differentiated (0 < fired < G); all-zero (no episode fired); all-full; hit+miss mixed
    (>=1 fired and >=1 fact-omission miss = the replay-target definition).  Split by forced / on-demand."""
    out = {}
    for tag, sel in (("", lambda d: True), ("_forced", lambda d: d["forced"]), ("_unforced", lambda d: not d["forced"])):
        g = [d for d in details if d.get("fact_n_eligible", 0) > 0 and sel(d)]
        out.update({"fact_groups" + tag: len(g),
                    "fact_groups_diff" + tag: sum(0 < d["fact_n_fired"] < d["n_episodes"] for d in g),
                    "fact_groups_all_zero" + tag: sum(d["fact_n_fired"] == 0 for d in g),
                    "fact_groups_all_full" + tag: sum(d["fact_n_fired"] == d["n_episodes"] for d in g),
                    "fact_groups_mixed_hit_miss" + tag: sum(d["fact_n_fired"] > 0 and d["fact_n_miss"] > 0 for d in g)})
    return out


def _gather(rep: dict) -> dict:
    """Sum the additive counters over DDP ranks when torch.distributed is up (every rank calls it once per step)."""
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return rep
        objs = [None] * dist.get_world_size()
        dist.all_gather_object(objs, rep)
    except Exception:                                   # noqa: BLE001 - never let logging break a step
        return rep
    return merge_reports(objs)


def merge_reports(objs: list[dict]) -> dict:
    """Merge rank-local counters; kept counts come from on_episode BEFORE DDP serialization."""
    out = copy.deepcopy(objs[0])
    for o in objs[1:]:
        out["forced_rows"] += o["forced_rows"]
        out.setdefault("kept_group_details", []).extend(o.get("kept_group_details", []))
        for k in ("forced_episodes", "scored_forced", "unforced_eligible_episodes", "unforced_eligible_calls", "p6_acute", "p6_fired"):
            out[k] += o[k]
        for k in FACT_KEYS:
            if k in o:
                out[k] = out.get(k, 0) + o[k]
        for k, v in o["drawn_items"].items():
            out["drawn_items"][k] = out["drawn_items"].get(k, 0) + v
        for k in ("kept_groups", "kept_forced_groups"):
            if out[k] is not None and o[k] is not None:
                out[k] += o[k]
    n_f = sum(v for k, v in out["drawn_items"].items() if k.startswith("f:"))
    n_e = sum(v for k, v in out["drawn_items"].items() if k[:2] in ("f:", "u:"))
    out["forced_share_of_eligible"] = (n_f / n_e) if n_e else None
    if "fact_eligible" in out:
        ne_f, nf = out["fact_eligible"], out["fact_fired"]
        out["fact_mean_fraction_eligible"] = out["fact_fraction_sum_eligible"] / ne_f if ne_f else None
        out["fact_mean_words_fired"] = out["fact_words_fired"] / nf if nf else None
        out["fact_echo_share_fired"] = out["fact_echo_fired"] / nf if nf else None
    ne = out.get("unforced_eligible_episodes", 0)
    out["unforced_eligible_call_rate"] = out.get("unforced_eligible_calls", 0) / ne if ne else None
    return out


_STATE: dict = {"installed": False}


def install(out: Path, *, seed: int = 0, p_present: float = P_PRESENT, p_absent: float = P_ABSENT, w6: float = W6,
            exclude_safe: bool = EXCLUDE_SAFE, p_safe: float = P_SAFE, w_f: float = 0.0) -> ForcedMix:
    """Install LAST (outermost), after every other hook.  Wraps algo_v2.dynamic_rollout and grpo_v2.train_step."""
    if _STATE["installed"]:
        raise RuntimeError("forced_mix installed twice")
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    fm = ForcedMix(seed, p_present, p_absent, w6, exclude_safe, Path(out) / "forced_mix_steps.jsonl", p_safe, w_f)
    from trainer_v2 import run_v2 as RV
    if "forced_mix" not in RV.STEP_EXTRA_KEYS:
        RV.STEP_EXTRA_KEYS = RV.STEP_EXTRA_KEYS + ("forced_mix",)
    dyn, ts = AL.dynamic_rollout, G.train_step

    def dynamic_rollout(items, images, backend, lcfg, refill, rollout, score, *a, **kw):
        return dyn(items, images, backend, lcfg, refill, fm.split_rollout(rollout), fm.wrap_score(score), *a, **kw)

    def train_step(*a, **kw):
        fm.t.clear(); fm.forced_ids.clear(); fm.forced_rows.clear()
        kept_items = {}; details = {}
        previous = kw.get("on_episode")
        def capture(ep, rec):
            key = id(ep.item)
            kept_items[key] = ep.item
            info = details.setdefault(key, {"row_id": ep.item.row_id, "forced": bool(getattr(ep, "forced", False)),
                "memory_required": bool(ep.item.target.get("memory_required")), "n_episodes": 0, "n_calls": 0, "reward_sum": 0.0})
            if info["forced"] != bool(getattr(ep, "forced", False)):
                raise RuntimeError("Mixed forced/on-demand episodes in one DAPO group")
            info["n_episodes"] += 1
            info["n_calls"] += int(ep.decision == "call_pdb")
            info["reward_sum"] += float(ep.ledger["total"])
            fbk = ep.ledger.get("fact_bonus")
            if fbk is not None:
                for kk, vv in (("fact_n_eligible", fbk["eligible"]), ("fact_n_fired", fbk["fired"]), ("fact_n_miss", fbk["miss"])):
                    info[kk] = info.get(kk, 0) + int(vv)
            if previous is not None:
                previous(ep, rec)
        res = ts(*a, **{**kw, "on_episode": capture})
        local = fm.report(list(kept_items.values()))
        local["kept_group_details"] = list(details.values())
        rep = _gather(local)
        if fm.w_f > 0:
            rep.update(fact_group_stats(rep.get("kept_group_details", [])))
        res["forced_mix"] = rep
        try:
            import torch.distributed as dist
            rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        except Exception:                               # noqa: BLE001
            rank0 = True
        if rank0:
            fm.log_path.parent.mkdir(parents=True, exist_ok=True)
            with fm.log_path.open("a") as fh:
                fh.write(json.dumps({"step": fm.nstep, **rep}) + "\n")
            fm.nstep += 1
        return res

    AL.dynamic_rollout, G.train_step = dynamic_rollout, train_step
    _STATE.update(installed=True, fm=fm, orig=(dyn, ts))
    return fm


def install_gt_clean(edits_jsonl: Path) -> dict:
    """Replace gold_answer of the edited training rows at release load time (release files and pins untouched)."""
    from eval_v1 import release_data as RD
    edits = {}
    for line in Path(edits_jsonl).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            edits[d["row_id"]] = d
    orig = RD.convert
    seen: Counter = Counter()

    def convert(row, root, arms, **kw):
        e = edits.get(row.get("sample_id"))
        if e is not None and row.get("split") == "train":
            if row.get("gold_answer") != e["before"]:
                raise RuntimeError(f"GT clean: {row['sample_id']} gold_answer differs from the recorded 'before'")
            row = {**row, "gold_answer": e["after"]}
            seen[row["sample_id"]] += 1
        return orig(row, root, arms, **kw)
    RD.convert = convert
    return {"n_edits": len(edits), "applied": seen}
