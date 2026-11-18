"""M1 additive reward v3.1: application-level personal content, no weight change.

Not wired to train_rl. Each decisive fact needs an owner-reviewed application
contract; the old keyphrases are retained for audit but never determine credit.
This is a conservative finite-pattern scorer, not a semantic judge. Uncovered
paraphrases must be reviewed and added versionedly, never guessed as success.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rl_sweep"))
import rewards_v3 as R3  # noqa: E402

VERSION = "v3.1-m1-fixture"
W = dict(R3.W)
assert W == R3.W
CATEGORIES = {"correct_use", "correct_paraphrase_without_condition_name", "mentioned_but_misapplied",
              "wrong_person", "ignored", "hallucinated"}


class ApplicationContractError(ValueError):
    pass


@dataclass(frozen=True)
class ApplicationResult:
    category: str
    cited_decisive: bool
    matched_rule: str | None


def _compile_contract(fact: dict) -> dict[str, list[re.Pattern]]:
    contract = fact.get("application_contract")
    if not isinstance(contract, dict) or contract.get("version") != "m1.v1":
        raise ApplicationContractError("owner-reviewed application_contract m1.v1 required")
    required = ("support_patterns", "contrary_patterns", "wrong_person_patterns",
                "hallucination_patterns", "condition_name_patterns")
    out = {}
    for key in required:
        patterns = contract.get(key)
        if not isinstance(patterns, list) or (key == "support_patterns" and not patterns):
            raise ApplicationContractError(f"invalid application contract: {key}")
        if any(not isinstance(p, str) or not p.strip() or len(p) > 180 for p in patterns):
            raise ApplicationContractError(f"invalid application pattern: {key}")
        try:
            out[key] = [re.compile(p, re.I) for p in patterns]
        except re.error as exc:
            raise ApplicationContractError(f"invalid regex in {key}") from exc
    return out


def classify_application(completion, fact: dict, decisive_ranks: list[int], withhold: bool = False) -> ApplicationResult:
    """Classify the *answer's application*, never negation near a condition name."""
    parsed = R3._p(completion)
    answer = parsed["answer"] or ""
    rules = _compile_contract(fact)
    cited = bool(R3.cited(parsed) & set(decisive_ranks))

    def match(key):
        for idx, pattern in enumerate(rules[key]):
            if pattern.search(answer):
                return f"{key}:{idx}"
        return None

    condition = match("condition_name_patterns")
    if withhold:
        return ApplicationResult("hallucinated" if condition or match("support_patterns") else "ignored", False, None)
    for key, category in (("wrong_person_patterns", "wrong_person"),
                          ("hallucination_patterns", "hallucinated"),
                          ("contrary_patterns", "mentioned_but_misapplied")):
        hit = match(key)
        if hit:
            return ApplicationResult(category, cited, hit)
    support = match("support_patterns")
    if support:
        return ApplicationResult("correct_use" if condition else "correct_paraphrase_without_condition_name",
                                 cited, support)
    return ApplicationResult("ignored", cited, condition)


def r_personal_content_v3_1(completions, withhold, decisive_fact, decisive_ranks, **_):
    """Same 0.12 weight and citation gate as v3; only the application test changes."""
    out = []
    for c, wh, fact, ranks in zip(completions, withhold, decisive_fact, decisive_ranks):
        if not fact:
            out.append(0.0)
            continue
        result = classify_application(c, fact, ranks or [], wh)
        ok = result.category in {"correct_use", "correct_paraphrase_without_condition_name"}
        out.append(W["personal_content"] * float(ok and result.cited_decisive and not wh))
    return out


RLVR_FUNCS = [r_personal_content_v3_1 if f is R3.r_personal_content else f for f in R3.RLVR_FUNCS]
assert len(RLVR_FUNCS) == len(R3.RLVR_FUNCS) and sum(f is r_personal_content_v3_1 for f in RLVR_FUNCS) == 1
