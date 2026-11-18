"""reward_v5.1 soft personal_content matcher (owner 2026-10-04)."""
from __future__ import annotations

import pytest

from premind.rewards import personal_match as PM
from premind.rewards import reward_v5 as R5


@pytest.mark.parametrize("pattern,answer,expected", [
    ("(?i)hand (?:the garden tools|them) to Simon", "Hand the tools to the man; that is Simon, cleared for the garden tools.", True),
    ("(?i)do not offer Jonah latex gloves", "If you mean Ezra, he may use latex gloves.", False),       # wrong name
    ("(?i)do not offer Jonah latex gloves", "Offer Jonah latex gloves.", False),                       # lost negation
    ("(?i)do not offer Jonah latex gloves", "Don't give Jonah latex gloves; he is allergic.", True),
    ("(?i)(?:help|assist) Dana (?:use|with) her prescribed rescue inhaler", "Call emergency services now.", False),
])
def test_soft_hit(pattern: str, answer: str, expected: bool) -> None:
    assert PM.hit(pattern, answer, "soft") is expected


def test_strict_is_v50_regex() -> None:
    p, a = "(?i)hand (?:the garden tools|them) to Simon", "Hand the tools to Simon."
    assert PM.hit(p, a, "strict") is False and PM.hit(p, a, "soft") is True
    assert PM.hit(p, "please hand them to Simon", "strict") is True
    with pytest.raises(ValueError):
        PM.hit(p, a, "fuzzy")


def test_presets() -> None:
    assert R5.PRESETS["main"].personal_match == "soft" and R5.PRESETS["v5.0"].personal_match == "strict"
    assert R5.REWARD_VERSION["call"].startswith("reward_v5.2")
