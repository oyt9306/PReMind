"""hazard_first / leading_alarm (owner 2026-10-04 C4 metric fix)."""
from __future__ import annotations

import pytest

from premind.rewards import hazard_match as HM


@pytest.mark.parametrize("pattern,answer,expected", [
    ("(?i)keep hands clear", "Stop working immediately. The hose can spray.", True),
    ("(?i)keep hands clear", "You should offer Simon nitrile gloves.", False),
    (r"(?i)\b(?:mug|floor)\b", "I don't have information about that mug on the floor.", False),   # naming is not acting
    ("(?i)first, help Abel steady himself", "First, help Abel steady himself, then move the bag.", True),
    ("(?i)keep hands clear", "This is an emergency. Call emergency services now.", True),          # alarm then action
    ("(?i)keep hands clear", "Yes, you should call for emergency help immediately.", True),
    (None, "anything", None),
])
def test_hazard_first(pattern, answer, expected) -> None:
    assert HM.hazard_first(pattern, answer) is expected


def test_answer_text_and_alarm() -> None:
    assert HM.answer_text("<decision>x</decision>\n<answer> Unplug it now. </answer>") == "Unplug it now."
    assert HM.leading_alarm("Unplug the heater first. Then put the vase away.")
    assert not HM.leading_alarm("Put the vase on the sideboard.")
