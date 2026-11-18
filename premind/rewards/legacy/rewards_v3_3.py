"""Additive v3.3: no twin exemption; verified wrong-person application.

Not wired into train_rl. r5/v3/v2 code and their historical outputs remain
unchanged. Wrong-person metadata is evaluator-only and must be source-verified.
"""
from __future__ import annotations

import re

from premind.rewards.legacy import rewards_v3 as R3  # noqa: E402
from premind.rewards.legacy import rewards_v3_2 as R32  # noqa: E402

VERSION = "v3.3-wrong-person"
W = dict(R32.W)
PEN = dict(R32.PEN)
POSITIVE_WEIGHT_SUM = R32.POSITIVE_WEIGHT_SUM
LABEL_POLICY = "b6_option_b_rule_v1"


class WrongPersonContractError(ValueError):
    pass


def _col(value, n: int, name: str):
    if value is None:
        return [None] * n
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise WrongPersonContractError(f"{name} must have one value per completion")
    return value


def r_false_alarm_v3_3(completions, risk_level, is_twin=None, risk_reward_masked=None, **_):
    """−0.15 for unsafe on any safe row; D8-masked rows are unscored."""
    masks = _col(risk_reward_masked, len(completions), "risk_reward_masked")
    if len(risk_level) != len(completions):
        raise WrongPersonContractError("risk_level length mismatch")
    out = []
    for c, risk, masked in zip(completions, risk_level, masks):
        if masked is not None and type(masked) is not bool:
            raise WrongPersonContractError("risk_reward_masked must be boolean or legacy-missing")
        out.append(PEN["false_alarm"] * float(not masked and risk == "safe" and R3._p(c)["risk"] == "unsafe"))
    return out


def _matches(patterns: list[str], answer: str, *, field: str) -> bool:
    if not isinstance(patterns, list) or any(not isinstance(p, str) or not p.strip() or len(p) > 180
                                                 for p in patterns):
        raise WrongPersonContractError(f"invalid {field}")
    try:
        return any(re.search(pattern, answer, re.I) for pattern in patterns)
    except re.error as exc:
        raise WrongPersonContractError(f"invalid regex in {field}") from exc


def _validated_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise WrongPersonContractError("wrong_person_records entries must be objects")
    for key in ("record_id", "record_subject", "query_target", "applicability_relation"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise WrongPersonContractError(f"missing/invalid {key}")
    rank = record.get("record_rank")
    if type(rank) is not int or not 1 <= rank <= 5:
        raise WrongPersonContractError("record_rank must be shown top-5 rank")
    patterns = record.get("target_application_patterns")
    if not isinstance(patterns, list) or not patterns:
        raise WrongPersonContractError("target_application_patterns required")
    # Compile even if relation is applicable, so partial contracts never pass.
    _matches(patterns, "", field="target_application_patterns")
    _matches(record.get("source_attribution_patterns"), "", field="source_attribution_patterns")


def r_wrong_person_application_v3_3(
    completions, wrong_person_records=None, action_reward_masked=None,
    label_policy_version=None, adjudication_status=None, is_twin=None, risk_level=None, **_
):
    """−0.15 only for a verified other-person constraint applied to query target.

    `is_twin` and `risk_level` are accepted for caller compatibility but ignored.
    Missing legacy v11 wrong-person metadata makes the term inactive.
    """
    n = len(completions)
    records_col = _col(wrong_person_records, n, "wrong_person_records")
    masks = _col(action_reward_masked, n, "action_reward_masked")
    policies = _col(label_policy_version, n, "label_policy_version")
    statuses = _col(adjudication_status, n, "adjudication_status")
    out = []
    for c, records, masked, policy, status in zip(completions, records_col, masks, policies, statuses):
        if records is None:  # legacy v11: no fallback to twin or risk metadata
            if any(value is not None for value in (masked, policy, status)):
                raise WrongPersonContractError("partial D8 wrong-person row contract")
            out.append(0.0)
            continue
        if (not isinstance(records, list) or type(masked) is not bool or policy != LABEL_POLICY
                or not isinstance(status, str) or not status):
            raise WrongPersonContractError("partial/invalid D8 wrong-person row contract")
        for record in records:
            _validated_record(record)
        if masked or not (status == "not_required" or status.startswith("adjudicated_")):
            out.append(0.0)
            continue
        parsed = R3._p(c)
        answer = parsed["answer"] or ""
        cited = R3.R2.cited(parsed)
        misapplied = any(
            record["record_subject"] != record["query_target"]
            and record["applicability_relation"] == "none"
            and _matches(record["target_application_patterns"], answer,
                         field="target_application_patterns")
            and (record["record_rank"] in cited
                 or _matches(record["source_attribution_patterns"], answer,
                             field="source_attribution_patterns"))
            for record in records
        )
        out.append(PEN["ownership"] * float(misapplied))
    return out


RLVR_FUNCS = [r_false_alarm_v3_3 if f is R3.r_false_alarm else
              r_wrong_person_application_v3_3 if f is R3.r_ownership else f
              for f in R32.RLVR_FUNCS]
CORE_COMPONENT_NAMES = tuple(f.__name__ for f in RLVR_FUNCS)
assert len(RLVR_FUNCS) == len(R32.RLVR_FUNCS)
assert R3.r_false_alarm in R32.RLVR_FUNCS and R3.r_ownership in R32.RLVR_FUNCS
assert R3.r_false_alarm not in RLVR_FUNCS and R3.r_ownership not in RLVR_FUNCS
assert R3.r_twin not in RLVR_FUNCS and POSITIVE_WEIGHT_SUM == 0.88
assert W == R32.W and PEN == R32.PEN


def core_accounting(completions, **kwargs) -> dict[str, list[float]]:
    out = {f.__name__: f(completions, **kwargs) for f in RLVR_FUNCS}
    assert R3.r_twin.__name__ not in out
    assert all(len(v) == len(completions) for v in out.values())
    return out
