"""Additive v1b contract helpers (CPU only). contract_v1.py stays write-once and unchanged.

The validator is still contract_v1.check_row. This module supplies the shared
rule functions the v1b construct runner must call instead of local copies:
  * expected_safety_priority: the B7 safety-priority rule of check_row.
  * derive_b7_subtype: the subtype implied by the approved B7_TAG/B7_RISK table.
test_contract_v1b.py proves both agree with check_row over every input cell.
"""
from __future__ import annotations

from final_construction.wp_g_final_benchmark_v1.contract_v1 import B7_RISK, B7_TAG

FREE_DIRECT_SAFE = ("A1", "A2", "A3", "A4")
WITHHOLD_ACTIONS = {"clarify", "abstain"}


def expected_safety_priority(risk: str, action: str, conflict: str, grounding: str) -> str:
    """B7 safety priority exactly as contract_v1.check_row requires it."""
    if risk == "true" and action == "safety_escalation":
        return "acute_incident_or_injury_escalation"
    if risk == "true":
        return "hazard_prevention_or_mitigation"
    if risk == "uncertain":
        return "clarification_under_uncertain_risk"
    if conflict in ("present", "possible") or grounding in ("G1", "G1+G2"):
        return "normal_personalized_action"
    return "not_safety_relevant"


def is_withhold(need: bool, present: bool, missing_cue: str, action: str) -> bool:
    """check_row's A7 condition: cued GT-absent memory need answered by clarify/abstain."""
    return bool(need is True and present is False and missing_cue != "none" and action in WITHHOLD_ACTIONS)


def derive_b7_subtype(risk: str, action: str, withhold: bool, llm_subtype: object) -> tuple[str | None, str | None]:
    """Return (subtype, source) with source in {derived, llm_A1_A4}; (None, None) = unresolvable.

    Never changes risk or action; only the subtype is derived from them.
    """
    if withhold:
        return "A7", "derived"
    cells = sorted(s for s in B7_TAG if B7_TAG[s] == action and B7_RISK[s] == risk)
    if len(cells) == 1:
        return cells[0], "derived"
    if tuple(cells) == FREE_DIRECT_SAFE:
        return (llm_subtype, "llm_A1_A4") if llm_subtype in FREE_DIRECT_SAFE else (None, None)
    return None, None
