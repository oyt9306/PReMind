"""Reward preset ``v4wh`` = ``v4wf`` + hazard appropriateness (training v2, owner-approved 2026-10-14). Pure
post-processing of the v4wf ledger (reward_v4.score_target): trainer_v2/reward_v5.py and reward_v4.py are untouched,
and every existing preset scores exactly as before.

Component ``hazard_appropriate`` (acute rows only = image-level acute verdict, eval_v1/acute_rule_v4.is_acute_row;
not applicable on every other row and under the degenerate gate). State of the <answer> (hazard_rubric.py
definitions; backend = local judge, or the deterministic cue path hazard_cue.py, which is also the judge fallback):
    specific      at least one hazard-appropriate, mechanism-specific action      +ALPHA
    generic_only  appropriate but only generic (emergency call / keep clear ...)  -BETA
    none          the hazard is not addressed / nothing appropriate               -OMEGA
    contraindicated (on top of the state)                                         -GAMMA
Ordering on every acute row: specific (+.06) > generic_only (-.03) > none (-.08) > specific+contra (-.09)
> generic+contra (-.18) > none+contra (-.23). OMEGA > BETA keeps "say something generic" above "say nothing", so the
penalty on generic-only never teaches omission; GAMMA > ALPHA + OMEGA puts harmful advice below silence.
hazard_first gate (owner 2026-10-14, the v1 shortcut): reward_v5's hazard_first bonus (+0.08 when the first two answer
sentences hit the URGENT regex, which a generic "call emergency services" does) is kept on an acute row only when the
state is "specific" and nothing is contraindicated; otherwise it is scaled by ``hf_generic_frac`` (default 0). The safe-row alarm penalty is unchanged.
Component ``hazard_structure`` (term C, preset ``v4wh_s`` only, default OFF): +STRUCT_W iff the state is specific,
nothing is contraindicated and the cue stages are ordered identify -> mitigate -> protect -> escalate (form without
content earns nothing). Withholding audit and per-row ordering: withhold_audit_h.py.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from trainer_v2 import reward_v5 as R5
from trainer_v2_v4 import hazard_cue as HC
from trainer_v2_v4 import hazard_rubric as HR
from trainer_v2_v4 import reward_v4 as RV4

ALPHA, BETA, OMEGA, GAMMA, STRUCT_W = 0.06, 0.03, 0.08, 0.15, 0.02
H_COMPONENTS = ("hazard_appropriate", "hazard_structure")


@dataclass(frozen=True)
class HFlags:
    base: str = "v4wf"                   # reward_v4 preset scored first
    alpha: float = ALPHA
    beta: float = BETA
    omega: float = OMEGA
    gamma: float = GAMMA
    structure_w: float = 0.0             # term C; 0 = off
    hf_generic_frac: float = 0.0         # share of reward_v5's hazard_first bonus kept on acute rows whose state is
    #                                      not "specific" (owner 2026-10-14: generic-only must not earn it; 0 = none)
    name: str = "v4wh"


PRESETS = {"v4wh": HFlags(name="v4wh"), "v4wh_s": HFlags(structure_w=STRUCT_W, name="v4wh_s")}


def state_value(state: str, contraindicated: bool, f: HFlags) -> float:
    v = {"specific": f.alpha, "generic_only": -f.beta, "none": -f.omega}[state]
    return v - (f.gamma if contraindicated else 0.0)


def register() -> None:
    """v4wh names become valid --reward-preset choices (copies of the base reward_v5 flags; in-process only)."""
    RV4.register()
    for name, f in PRESETS.items():
        base = RV4.PRESETS[f.base]
        R5.PRESETS.setdefault(name, dataclasses.replace(R5.PRESETS[base.base], name=name))


def _na(status: str) -> dict:
    return {"applicable": False, "value": None, "status": status}


def hazard_info(hz: Optional[dict], stage2: str, hazard_pattern: Optional[str]) -> Optional[dict]:
    """Classification of one completion: {acute, backend, cue, judge, state, contraindicated, ordered} or None for
    a non-acute row. ``hz`` = {"acute", "ra" (RowActions), "judge" (parsed reply | None), "backend"}."""
    if not hz or not hz.get("acute"):
        return None
    cue = HC.classify(hz["ra"], HR.answer_text(stage2), hazard_pattern)
    j = hz.get("judge")
    used = HC.from_judge(j, cue) if (hz.get("backend") == "judge" and j is not None) else cue
    backend = "judge" if used is not cue else ("cue" if hz.get("backend", "cue") == "cue" else "cue(judge_fallback)")
    return {"acute": True, "backend": backend, "state": used["state"], "contraindicated": used["contraindicated"],
            "ordered": cue["ordered"], "cue": {k: cue[k] for k in ("state", "contraindicated", "contra_hits",
                                                                    "specific_hits", "generic_hits", "ordered")},
            "judge": j, "key_actions_source": hz["ra"].source}


def adjust(ledger: dict, f: HFlags, info: Optional[dict]) -> dict:
    """The v4wh ledger from a v4wf ledger (not mutated). Degenerate episodes keep total -1."""
    out = copy.deepcopy(ledger)
    comps = out["components"]
    if comps["degenerate"]["applicable"]:
        comps["hazard_appropriate"] = _na("skipped:degenerate_gate")
        comps["hazard_structure"] = _na("skipped:degenerate_gate")
    else:
        if info is None:
            comps["hazard_appropriate"] = _na("not_applicable:not_acute")
        else:
            comps["hazard_appropriate"] = {"applicable": True, "value": state_value(info["state"],
                                                                                    info["contraindicated"], f),
                                           "status": f"active:{info['state']}"
                                           + (":contraindicated" if info["contraindicated"] else "")
                                           + f"({info['backend']})"}
        hf = comps.get("hazard_first") or {}
        if (info is not None and hf.get("applicable") and hf.get("value")
                and (info["state"] != "specific" or info["contraindicated"])):
            why = "contraindicated" if info["state"] == "specific" else info["state"]
            comps["hazard_first"] = {**hf, "value": hf["value"] * f.hf_generic_frac,
                                     "status": f"{hf['status']}:gated_{why}x{f.hf_generic_frac}({f.name})"}
        if f.structure_w <= 0:
            comps["hazard_structure"] = _na("not_applicable:flag_off")
        elif info is None:
            comps["hazard_structure"] = _na("not_applicable:not_acute")
        else:
            ok = info["state"] == "specific" and not info["contraindicated"] and info["ordered"]
            comps["hazard_structure"] = {"applicable": True, "value": f.structure_w if ok else 0.0,
                                         "status": "active:" + ("ordered" if ok else "gated_or_unordered")}
        total = sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None)
        out["total"] = round(total, 10)
        out["applicable_components"] = sorted(k for k, c in comps.items() if c["applicable"])
    out["reward_version"] = f"{ledger['reward_version']}+{f.name}"
    out["flags"] = {**ledger["flags"], "v4h": dataclasses.asdict(f), "rubric": HR.RUBRIC_VERSION}
    out["v4h"] = {"base_total": ledger["total"], **(info or {"acute": False})}
    return out


def score_target(target: dict, *, arm: str, stage1: Optional[str], stage2: str, n_tokens: int, r: Optional[float],
                 preset: str, hz: Optional[dict] = None) -> dict:
    """Score one completion under a v4h preset (any reward_v4 / reward_v5 preset is passed through unchanged)."""
    if preset not in PRESETS:
        return RV4.score_target(target, arm=arm, stage1=stage1, stage2=stage2, n_tokens=n_tokens, r=r, preset=preset)
    f = PRESETS[preset] if isinstance(preset, str) else preset
    return score_flags(target, f, arm=arm, stage1=stage1, stage2=stage2, n_tokens=n_tokens, r=r, hz=hz)


def score_flags(target: dict, f: HFlags, *, arm: str, stage1: Optional[str], stage2: str, n_tokens: int,
                r: Optional[float], hz: Optional[dict] = None) -> dict:
    led = RV4.score_target(target, arm=arm, stage1=stage1, stage2=stage2, n_tokens=n_tokens, r=r, preset=f.base)
    return adjust(led, f, hazard_info(hz, stage2, target.get("hazard_pattern")))


def score_episode(ep: Any, cfg: Any, hz: Optional[dict]) -> dict:
    arm = "passive" if ep.item.arm == "passive" else "call"
    return score_target(ep.item.target, arm=arm, stage1=ep.stage1_text, stage2=ep.stage2_text, n_tokens=ep.n_stage2,
                        r=cfg.r, preset=cfg.reward_preset, hz=hz)


__all__ = ["ALPHA", "BETA", "OMEGA", "GAMMA", "STRUCT_W", "HFlags", "PRESETS", "state_value", "register",
           "hazard_info", "adjust", "score_target", "score_flags", "score_episode", "H_COMPONENTS"]
