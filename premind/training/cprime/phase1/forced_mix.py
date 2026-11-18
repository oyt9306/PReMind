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
                  answer: str = "") -> dict:
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
    out["total"] = round(total, 10)
    return out


# ----------------------------------------------------------------------------- stateful hook
class ForcedMix:
    def __init__(self, seed: int = 0, p_present: float = P_PRESENT, p_absent: float = P_ABSENT, w6: float = W6,
                 exclude_safe: bool = EXCLUDE_SAFE, log_path: Optional[Path] = None, p_safe: float = P_SAFE):
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
            led2 = adjust_ledger(led, forced, self.w6, answer=answer_text(getattr(ep, "stage2_text", "")))
            if forced:
                self.t["scored_forced"] += 1
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
               "unforced_eligible_call_rate": (t["unforced_eligible_calls"] / t["unforced_eligible_eps"]
                                               if t.get("unforced_eligible_eps") else None),
               "kept_groups": None if used_items is None else len(used_items), "kept_forced_groups": kept_f,
               "p6_acute": t.get("p6_acute", 0), "p6_fired": t.get("p6_fired", 0), "forced_rows": list(self.forced_rows),
               "p_present": self.p_present, "p_absent": self.p_absent, "p_safe": self.p_safe, "w6": self.w6}
        n_f = sum(v for k, v in rep["drawn_items"].items() if k.startswith("f:"))
        n_e = sum(v for k, v in rep["drawn_items"].items() if k[:2] in ("f:", "u:"))
        rep["forced_share_of_eligible"] = (n_f / n_e) if n_e else None
        return rep


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
    out = copy.deepcopy(objs[0])
    for o in objs[1:]:
        out["forced_rows"] += o["forced_rows"]
        for k in ("forced_episodes", "scored_forced", "unforced_eligible_episodes", "p6_acute", "p6_fired"):
            out[k] += o[k]
        for k, v in o["drawn_items"].items():
            out["drawn_items"][k] = out["drawn_items"].get(k, 0) + v
        for k in ("kept_groups", "kept_forced_groups"):
            if out[k] is not None and o[k] is not None:
                out[k] += o[k]
    n_f = sum(v for k, v in out["drawn_items"].items() if k.startswith("f:"))
    n_e = sum(v for k, v in out["drawn_items"].items() if k[:2] in ("f:", "u:"))
    out["forced_share_of_eligible"] = (n_f / n_e) if n_e else None
    return out


_STATE: dict = {"installed": False}


def install(out: Path, *, seed: int = 0, p_present: float = P_PRESENT, p_absent: float = P_ABSENT, w6: float = W6,
            exclude_safe: bool = EXCLUDE_SAFE, p_safe: float = P_SAFE) -> ForcedMix:
    """Install LAST (outermost), after every other hook.  Wraps algo_v2.dynamic_rollout and grpo_v2.train_step."""
    if _STATE["installed"]:
        raise RuntimeError("forced_mix installed twice")
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import grpo_v2 as G
    fm = ForcedMix(seed, p_present, p_absent, w6, exclude_safe, Path(out) / "forced_mix_steps.jsonl", p_safe)
    dyn, ts = AL.dynamic_rollout, G.train_step

    def dynamic_rollout(items, images, backend, lcfg, refill, rollout, score, *a, **kw):
        return dyn(items, images, backend, lcfg, refill, fm.split_rollout(rollout), fm.wrap_score(score), *a, **kw)

    def train_step(*a, **kw):
        fm.t.clear(); fm.forced_ids.clear(); fm.forced_rows.clear()
        res = ts(*a, **kw)
        rep = _gather(fm.report(res.get("used_items")))
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
