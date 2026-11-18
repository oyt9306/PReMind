"""Reward preset "v52ws_ag" (owner 2026-10-10, warm-start Stage B, arm DAPO_ws_ag): episode Match input + step summary.

v52ws_ag = v52miss (missed needed call -2r) + (A) evidence-grounded withholding + (C) unneeded call -0.5r (reward_v5).
(A) needs the parsed Match part of the episode's <visual_evidence> (vef2.match_info: the rule-based parser of the VEF v2
schema, names / handles mapped to display indices through the row's own memory table; no judge, no Scene reward).
The arm renders its prompts with CONFIDANT_SCHEMA=vef2 (the model is asked for "Scene: ... / Match: ..."), so the
release is loaded with ``vef=True`` (``item.row["vef_src"]``) and every item's system prompt must carry the vef2
<visual_evidence> line (fail closed otherwise).
Step log (steps.jsonl ``ag``): grounded-withhold rate (correct withholds whose Match verdict is correct_none / all
correct withholds with memories shown), Match accuracy (correct / correct_none over memory-required episodes with
memories shown), unneeded-call rate (call_pdb on memory-not-required call-arm episodes).
"""
from __future__ import annotations

from typing import Any, Optional

from . import vef as VF
from . import vef2 as VF2

CORRECT = ("correct", "correct_none")


def check_prompt(item: Any) -> None:
    """Refuse (SystemExit) an item whose system prompt lacks the vef2 <visual_evidence> schema line."""
    sys_text = "".join(p.get("text", "") for p in item.messages[0]["content"])
    if VF2.prompt_line() not in sys_text:
        raise SystemExit(f"REFUSED: {item.row_id}: the prompt lacks the vef2 <visual_evidence> schema "
                         f"(the v52ws_ag preset needs CONFIDANT_SCHEMA={VF2.SCHEMA})")


def episode_match(item: Any, stage2_text: str, *, prompt_check: bool = True) -> dict:
    """vef2.match_info of one episode's <visual_evidence> against the item's rendered memory table."""
    src = item.row.get("vef_src") or {}
    if "match" not in src:
        raise ValueError(f"{item.row_id}: no vef_src.match (load the release with vef=True for v52ws_ag)")
    if prompt_check:
        check_prompt(item)
    ve, _ = VF.visual_evidence(stage2_text)
    return VF2.match_info(ve, VF2.table(src, item.target["rendered_keys"]))


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 6) if den else None


def summary(records: list[dict]) -> dict:
    """Step summary of the ``ag`` ledger blocks carried by the episode records (metrics_v9.episode_record)."""
    ags = [r["ag"] for r in records if r.get("ag")]
    cw = [a for a in ags if a["gate"] == "applied" and a["correct_withhold"]]
    m = [a for a in ags if a["match_verdict"]]
    wm = [a for a in m if a["gate"] == "applied"]
    nr = [a for a in ags if a["memory_required"] is False and a["unneeded_call"] is not None]
    return {"n": len(ags),
            "grounded_withhold_rate": _rate(sum(bool(a["grounded"]) for a in cw), len(cw)), "n_correct_withhold": len(cw),
            "match_acc": _rate(sum(a["match_verdict"] in CORRECT for a in m), len(m)), "n_match": len(m),
            "match_acc_withhold_target": _rate(sum(a["match_verdict"] == "correct_none" for a in wm), len(wm)),
            "n_match_withhold_target": len(wm),
            "match_lookalike_rate": _rate(sum(a["match_verdict"] == "lookalike" for a in m), len(m)),
            "match_missing_rate": _rate(sum(a["match_verdict"] in ("missing", "unparseable") for a in m), len(m)),
            "unneeded_call_rate": _rate(sum(bool(a["unneeded_call"]) for a in nr), len(nr)), "n_not_required": len(nr)}
