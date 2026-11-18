"""Episode -> pluggable reward function bridge with a per-component ledger (PROVISIONAL).

Reward semantics (DF-10, mirrored, r6 untouched): the passive arm uses the v3.3 lineage core
"minus the half-penalty and r_twin, with no renormalization" — the remaining positive weights are
NOT rescaled to sum to one; their documented positive sum 0.88 is not called a final total.
The memory-call arm adds R_dec = +r / -r (r required, no default; DF-01) and gates the
post-retrieval terms on the branch actually taken (MC prereg §2.4 table). Per-variant masks
(DF-07 / risk_mask_policy_version) come from the row; masked components are absent from the total.

GB-1 (owner ruling, Addendum 2, 2026-10-01): on a risk-masked row, `critical_miss` (the acute
critical-omission penalty baked into v3.3's `r_risk`) is split out into its own ledger component
and masked together with risk ONLY WHEN the row has no acute visual evidence
(`row["visual_acute_evidence"]` is False); if an acute hazard is evident in the image, critical_miss
stays active even though the rest of risk/false_alarm is masked. See `lineage_v3_3_core` and
`stage2_target`.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
_R6 = V3 / "rl_sweep" / "rlvr_v3_prep" / "r6"
for _p in (HERE, _R6):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rewards_v3_3 as R33  # noqa: E402  (read-only lineage import)
from adapter import ExclusionError, display_rank_map, parse_stage1  # noqa: E402

logger = logging.getLogger(__name__)

BRIDGE_VERSION = "train_interface_v1.reward_bridge_v1"
LINEAGE_VERSION = R33.VERSION
DOCUMENTED_POSITIVE_WEIGHT_SUM = 0.88  # DF-10: documented, not final, not renormalized
EXCLUDED_TERMS = ("r_twin", "half_penalty")
# lineage function name -> ledger component name
LINEAGE_COMPONENTS = {
    "r3_format": "format", "r_risk": "risk", "r_action": "action", "r_attribution": "attribution",
    "r_distractor": "distractor", "r3_uncertain_structure": "uncertain_structure",
    "r_false_alarm_v3_3": "false_alarm", "r_wrong_person_application_v3_3": "wrong_person",
    "r3_storage_content": "storage_content", "r_length": "length", "r_withhold": "withhold",
    "r_over_abstain": "over_abstain", "r_missed_warning": "missed_warning",
    "r_personal_content_v3_1": "personal_content",
}
# MC prereg §7.4: on a risk-masked row the risk term is removed and false_alarm suppressed.
RISK_MASK_GROUP = ("risk", "false_alarm")
ACTION_MASK_GROUP = ("action",)
# GB-1: critical_miss is split out of `risk` (see lineage_v3_3_core) so it can be masked
# independently. Masked together with RISK_MASK_GROUP UNLESS row["visual_acute_evidence"] is True.
CRITICAL_MISS_COMPONENT = "critical_miss"
# GB-1 extension (interpretation; see README "Open issues" -> "resolved by GB-1"): missed_warning
# only ever fires on NON-acute rows (safety_priority not in ACUTE; it is the non-acute counterpart
# of critical_miss), so GB-1's acute-visual carve-out structurally never applies to it -- it always
# depends on the dropped memory to know the true non-acute risk, so it is masked whenever risk is
# masked, with no exception. uncertain_structure is graded against the invariant risk_label and only
# rewards hedge+check language; it never penalizes the model for lacking the hidden record (hedging
# is the correct response to missing memory), so it is left ungated even on a masked row.
MEMORY_ONLY_RISK_COMPONENTS = ("missed_warning",)

# DF-10 guards: the lineage must carry neither r_twin nor the twin half-penalty carrier (r_ownership).
assert set(R33.CORE_COMPONENT_NAMES) == set(LINEAGE_COMPONENTS), "v3.3 lineage changed; re-audit"
assert "r_twin" not in R33.CORE_COMPONENT_NAMES and "r_ownership" not in R33.CORE_COMPONENT_NAMES
assert R33.POSITIVE_WEIGHT_SUM == DOCUMENTED_POSITIVE_WEIGHT_SUM
assert R33.R3.R2.PEN["critical_miss"] == -0.5  # GB-1 guard: value documented above and in the README

ComponentFn = Callable[..., dict]
JudgeFn = Callable[[str, dict], float]


def _critical_miss_terms(completions: list, *, safety_priority, withhold) -> list[float]:
    """Isolate the critical-omission penalty baked into v3 `r_risk`: PEN['critical_miss'] when the
    row is not withheld, safety_priority is acute, and the predicted risk is 'safe' -- the same
    guard rewards_v3.r_risk uses internally (GB-1 lets this be masked independently of `risk`)."""
    R2 = R33.R3.R2
    return [R2.PEN["critical_miss"] if (not wh and sp in R2.ACUTE and R33.R3._p(c)["risk"] == "safe") else 0.0
            for c, sp, wh in zip(completions, safety_priority, withhold)]


def lineage_v3_3_core(completions: list, **kwargs) -> dict[str, list[float]]:
    """Default pluggable component function: the v3.3 deterministic core, renamed.

    GB-1: the critical-omission penalty is split out of `risk` into its own `critical_miss`
    component (their sum equals the original undivided r_risk output) so the two can be masked
    independently on a risk-masked row; see `stage2_target`.
    """
    raw = R33.core_accounting(completions, **kwargs)
    out = {LINEAGE_COMPONENTS[name]: values for name, values in raw.items()}
    cm = _critical_miss_terms(completions, safety_priority=kwargs["safety_priority"], withhold=kwargs["withhold"])
    out["risk"] = [r - c for r, c in zip(out["risk"], cm)]
    out[CRITICAL_MISS_COMPONENT] = cm
    return out


def stage2_target(row: dict, branch: str) -> tuple[dict, dict[str, str], str]:
    """Reward kwargs (one completion), branch-gated components, and the GB-1 critical_miss branch
    that fired, for a Stage-2 turn.

    Branches: ``passive`` and ``called`` show the same exact-five context and share one target;
    ``not_called`` shows nothing: gt/decisive ranks and wrong-person records are emptied, and for
    R = true rows the frozen ``no_call_branch_targets`` (insufficient evidence) apply, with
    over_abstain not applied (MC prereg O-6).
    """
    if branch not in {"passive", "called", "not_called"}:
        raise ValueError(f"unknown branch {branch!r}")
    target = {"risk_label": row["risk_label"], "safety_priority": row["safety_priority"],
              "expected_action": row["expected_action"],
              "expected_memory_behavior": row["expected_memory_behavior"],
              "risk_reward_masked": row["risk_reward_masked"], "action_reward_masked": row["action_reward_masked"]}
    gated: dict[str, str] = {}
    shown = branch != "not_called"
    # Rows store retriever ranks; the model sees and cites display ranks (adapter.record_order, owner 2026-10-02).
    display = display_rank_map(row) if shown else {}
    if not shown and row["memory_required"]:
        target = dict(row["no_call_branch_targets"])
        gated["over_abstain"] = "not_applicable:required_no_call(O-6)"
    kwargs = {
        "risk_level": [target["risk_label"]], "safety_priority": [target["safety_priority"]],
        "exp_action": [target["expected_action"]],
        "withhold": [target["expected_memory_behavior"] == "clarify_or_abstain"],
        "gt_ranks": [[display[r] for r in row["gt_ranks"]] if shown else []],
        "decisive_ranks": [[display[r] for r in row["decisive_ranks"]] if shown else []],
        "decisive_fact": [row["decisive_fact"]], "scenario": [row["scenario"]],
        "storage_discard": [row["storage_discard"]], "risk_reward_masked": [target["risk_reward_masked"]],
        "action_reward_masked": [target["action_reward_masked"]],
        "wrong_person_records": [[{**w, "record_rank": display[w["record_rank"]]} for w in row["wrong_person_records"]]
                                 if shown else []],
        "label_policy_version": [row["label_policy_version"]],
        "adjudication_status": [row["adjudication_status"]],
    }
    critical_miss_branch = "risk_not_masked"
    if target["risk_reward_masked"]:
        reason = "masked:risk(" + str(row["risk_mask_reason"]) + ")"
        gated.update({c: reason for c in RISK_MASK_GROUP})
        gated.update({c: reason for c in MEMORY_ONLY_RISK_COMPONENTS})
        # GB-1: visual_acute_evidence is a row/image-level property (invariant across branches).
        if row["visual_acute_evidence"]:
            critical_miss_branch = "kept:acute_visual_evidence(GB-1)"
        else:
            gated[CRITICAL_MISS_COMPONENT] = reason
            critical_miss_branch = "masked:no_visual_evidence(GB-1)"
    if target["action_reward_masked"]:
        gated.update({c: "masked:action" for c in ACTION_MASK_GROUP})
    return kwargs, gated, critical_miss_branch


def validate_reward_inputs(row: dict, component_fn: ComponentFn = lineage_v3_3_core) -> None:
    """Admission-time dry evaluation on every branch; any lineage contract error excludes the row."""
    for branch in ("passive", "not_called"):
        kwargs, _, _ = stage2_target(row, branch)
        try:
            component_fn([""], **kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            raise ExclusionError(f"reward_inputs_invalid:{type(exc).__name__}:{exc}") from exc


@dataclass(frozen=True)
class Episode:
    """One rollout: the row (evaluator side), arm, Stage-1 output (memory-call only) and Stage-2 text."""
    row: dict
    arm: str
    stage1_output: Optional[str]
    completion: str


class RewardBridge:
    """Score episodes with a pluggable component function and write one ledger line per call."""

    def __init__(self, arm: str, *, r: Optional[float], contract_sha256: str,
                 component_fn: ComponentFn = lineage_v3_3_core, judge_fn: Optional[JudgeFn] = None,
                 ledger_path: Optional[Path] = None) -> None:
        if not isinstance(contract_sha256, str) or not contract_sha256:
            raise ValueError("contract_sha256 of the admitting upstream contract is required")
        if arm == "memory_call":
            if r is None:
                raise ValueError("memory-call arm requires an explicit decision magnitude r (DF-01: TBD, no default)")
            if not isinstance(r, (int, float)) or isinstance(r, bool) or r <= 0:
                raise ValueError("r must be a positive number")
        elif arm == "passive_rag":
            if r is not None:
                raise ValueError("passive arm has no decision term; pass r=None")
        else:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm, self.r, self.contract_sha256 = arm, r, contract_sha256
        self.component_fn, self.judge_fn, self.ledger_path = component_fn, judge_fn, ledger_path

    def score(self, ep: Episode) -> dict:
        """Return the ledger (with ``total``) for one episode and append it to the ledger file."""
        if ep.arm != self.arm:
            raise ValueError("episode arm does not match bridge arm")
        row = ep.row
        decision, decision_valid, branch = None, None, "passive"
        components: dict[str, dict] = {}
        if self.arm == "memory_call":
            decision = parse_stage1(ep.stage1_output)
            decision_valid = decision is not None
            branch = "called" if decision == "call_pdb" else "not_called"
            correct = decision == ("call_pdb" if row["memory_required"] else "direct_answer")
            components["decision"] = {"value": self.r if correct else -self.r, "status": "active"}
        elif ep.stage1_output is not None:
            raise ValueError("passive arm has no Stage-1 turn")
        kwargs, gated, critical_miss_branch = stage2_target(row, branch)
        values = self.component_fn([ep.completion], **kwargs)
        for name in sorted(values):
            status = gated.get(name, "active")
            components[name] = {"value": float(values[name][0]) if status == "active" else None, "status": status}
        if self.judge_fn is None:
            components["judge"] = {"value": None, "status": "not_evaluated:no_judge_fn"}
        else:
            components["judge"] = {"value": float(self.judge_fn(ep.completion, row)), "status": "active"}
        total = sum(c["value"] for c in components.values() if c["status"] == "active")
        ledger = {
            "bridge_version": BRIDGE_VERSION, "lineage_version": LINEAGE_VERSION,
            "contract_sha256": self.contract_sha256, "row_id": row["row_id"],
            "arm": self.arm, "branch": branch, "decision": decision, "decision_valid": decision_valid,
            "memory_required": row["memory_required"], "r": self.r, "components": components,
            "active_components": sorted(k for k, c in components.items() if c["status"] == "active"),
            "masked_components": sorted(k for k, c in components.items() if c["status"].startswith("masked")),
            "risk_mask_policy_version": row["risk_mask_policy_version"],
            "risk_mask_reason": row["risk_mask_reason"], "visual_acute_evidence": row["visual_acute_evidence"],
            "critical_miss_branch": critical_miss_branch, "excluded_terms": list(EXCLUDED_TERMS),
            "renormalized": False, "documented_positive_weight_sum": DOCUMENTED_POSITIVE_WEIGHT_SUM,
            "complete": components["judge"]["status"] == "active", "total": round(total, 10),
        }
        if self.ledger_path is not None:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ledger, sort_keys=True) + "\n")
        return ledger


def branch_balance(totals: dict[str, list[float]], *, margin_always: float, margin_never: float) -> dict:
    """DF-02 check: correct policy must beat always-call and never-call by explicit margins.

    Args:
        totals: ledger totals under keys ``correct``, ``always_call``, ``never_call`` (branch replay).
        margin_always, margin_never: TBD-PhaseC; required, no default.
    """
    if margin_always is None or margin_never is None:
        raise ValueError("branch-balance margins are TBD (DF-02) and must be given explicitly")
    if set(totals) != {"correct", "always_call", "never_call"} or any(not v for v in totals.values()):
        raise ValueError("totals must contain non-empty correct/always_call/never_call lists")
    mean = {k: sum(v) / len(v) for k, v in totals.items()}
    return {"mean": mean,
            "beats_always_call": mean["correct"] - mean["always_call"] >= margin_always,
            "beats_never_call": mean["correct"] - mean["never_call"] >= margin_never}
