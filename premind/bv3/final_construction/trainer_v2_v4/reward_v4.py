"""Withholding-fair reward presets for the PReMind v4 recipe (WP-B item 2). Pure post-processing of a reward_v5
ledger: trainer_v2/reward_v5.py is NOT edited (its sha256 is pinned by every existing PREREG and live eval jobs verify
those pins), and every existing preset scores exactly as before.

Audit (REPORT.md §2, ``withhold_audit.py``): two reward_v5 paths pay for withholding beyond what it should earn.
  W1 (reward_v5.py L269-276, ``_branch_target``): on a memory-required row that the model did NOT call, the branch
     target becomes a withhold target, so a clarify/abstain answer earns the correct-withhold credit (W 0.12) and an
     empty citation earns the full attribution term (set F1 of {} vs {} = 1 -> W 0.10), while the risk group is masked.
     No-call + withhold on a target-present row therefore scores 0.26 vs 0.06 for no-call + answer: the gradient
     teaches "ask without looking".
  W2 (``over_abstain`` -0.10 on non-withhold targets): on a called row with memories shown, the expected reward of
     withholding beats answering until the model is >= 56 % (>= 62 % with a look-alike fact) sure the target is shown,
     i.e. withholding is advantaged under uncertainty.
Fix ``v4wf`` (base preset ``main``):
  F1 not-called memory-required rows: ``withhold``, ``attribution`` and ``distractor`` become not applicable (nothing
     was shown, so there is nothing to withhold from or cite); the action term is kept (asking is still better than a
     guessed personal fact), so no-call + withhold earns -r + format + action = 0.04 and no-call + answer -0.04;
  F2 ``over_abstain`` x 1.5 (-0.15): on a called row the answer/withhold break-even confidence p* (class means of the
     training mixture, withhold_audit.break_even) falls from 0.52 to <= 0.50, so withholding is no longer advantaged
     under uncertainty (x 2.0 would push p* to 0.47, i.e. reward guessing over asking).
Everything else (decision +-r, hazard-first, risk, citation set F1 on shown branches, degenerate gate) is unchanged.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any

from trainer_v2 import reward_v5 as R5

NOCALL_GATED = ("withhold", "attribution", "distractor")


@dataclass(frozen=True)
class V4Flags:
    base: str = "main"                 # reward_v5 preset that scores the episode first
    nocall_credit: bool = True         # False = F1
    over_abstain_scale: float = 1.0    # 2.0 = F2
    name: str = "v4"


PRESETS = {"v4wf": V4Flags(nocall_credit=False, over_abstain_scale=1.5, name="v4wf"),
           # ablation of the fix: F1 only (shows that F1 alone does not remove the W2 bias)
           "v4wf_f1": V4Flags(nocall_credit=False, over_abstain_scale=1.0, name="v4wf_f1")}


def flags_dict(f: V4Flags) -> dict:
    return dataclasses.asdict(f)


def register() -> None:
    """Make the v4 names valid ``--reward-preset`` choices of eval_v1.main_run (in-process only): each name maps to
    a copy of its base reward_v5 flags; ``score`` applies the v4 adjustment on top."""
    for name, f in PRESETS.items():
        base = R5.PRESETS[f.base]
        if base.vef or base.withhold_match:
            raise ValueError("v4 presets need a base without VEF / withhold_match inputs")
        R5.PRESETS.setdefault(name, dataclasses.replace(base, name=name))


def adjust(ledger: dict, f: V4Flags) -> dict:
    """The v4 ledger: F1 / F2 applied to a reward_v5 ledger (not mutated). Degenerate episodes are unchanged."""
    out = copy.deepcopy(ledger)
    comps = out["components"]
    notes = []
    if not comps["degenerate"]["applicable"]:
        if not f.nocall_credit and out["branch"] == "not_called" and out["memory_required"]:
            for c in NOCALL_GATED:
                if comps[c]["applicable"]:
                    comps[c] = {"applicable": False, "value": None, "status": f"masked:no_memory_shown({f.name})"}
                    notes.append(f"gated:{c}")
        oa = comps["over_abstain"]
        if f.over_abstain_scale != 1.0 and oa["applicable"] and oa["value"] is not None:
            comps["over_abstain"] = {**oa, "value": oa["value"] * f.over_abstain_scale,
                                     "status": f"{oa['status']}:x{f.over_abstain_scale}({f.name})"}
            notes.append("over_abstain_scaled")
        total = sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None)
        out["total"] = round(total, 10)
        out["applicable_components"] = sorted(k for k, c in comps.items() if c["applicable"])
    out["reward_version"] = f"{ledger['reward_version']}+{f.name}"
    out["flags"] = {**ledger["flags"], "v4": flags_dict(f)}
    out["v4"] = {"base_total": ledger["total"], "notes": notes}
    return out


def score_target(target: dict, *, arm: str, stage1: str | None, stage2: str, n_tokens: int, r: float | None,
                 preset: str) -> dict:
    """Score one completion under a v4 preset (or any reward_v5 preset name, unchanged)."""
    if preset not in PRESETS:
        flags = R5.PRESETS[preset]
        if arm == "passive":
            return R5.reward_passive(target, stage2, n_tokens, flags=flags)
        return R5.reward_call(target, stage1, stage2, n_tokens, r=r, flags=flags)
    f = PRESETS[preset]
    base = dataclasses.replace(R5.PRESETS[f.base], name=preset)
    led = (R5.reward_passive(target, stage2, n_tokens, flags=base) if arm == "passive"
           else R5.reward_call(target, stage1, stage2, n_tokens, r=r, flags=base))
    return adjust(led, f)


def score_episode(ep: Any, cfg: Any) -> dict:
    """grpo_v2.score replacement body for a v4 preset."""
    arm = "passive" if ep.item.arm == "passive" else "call"
    return score_target(ep.item.target, arm=arm, stage1=ep.stage1_text, stage2=ep.stage2_text, n_tokens=ep.n_stage2,
                        r=cfg.r, preset=cfg.reward_preset)
