"""Pilot assemble fix (2026-10-03): I3 keyphrase presence and pattern-less fact application.

I3   The GPT author is told to refer to an object by its handle and to copy the keyphrase verbatim; in the pilot it
     often wrote the value phrase without the inserted designation ("leg is loose" in "A leg on the sofa from the
     flea-market find is loose"). The memory text keeps the fact sentence verbatim, so the keyphrase was never in the
     text to begin with (precheck did not check I3). Rule: the keyphrase tokens occur in order in the text, contiguous
     except for gaps made ONLY of the concept's own designation (handle, name, type words, articles/prepositions),
     <= MAX_GAP tokens in total. A paraphrase ("strap fixed by a cobbler" vs "a cobbler fixed ... strap") still fails
     and is an author failure (I3 is now a precheck check, so the delta repairs it).
A1b  every target's exported support pattern (its application_pattern) must match the row's own gold with
     re.search exactly as reward_v5 applies it: A1's keyphrase-overlap fallback and the re.I flag of the checks hid
     3/18 canary rows (two regexes that miss the gold wording, one capitalised "Print" vs a case-sensitive pattern).
A2/A3/A4 on facts WITHOUT an application_pattern (memory-text o1/o2/x facts): applied only when every distinctive
     keyphrase content word (>= 2) occurs in the answer - the 60% overlap rule matched generic words ("before
     leaving", "dog ... down", "footwear ... repaired") in golds that do not use the fact.
"""
from __future__ import annotations

import re

MAX_GAP = 8
GLUE = frozenset("the a an of on from in at my our this that its their his her s one user users".split())


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def keyphrase_present(keyphrase: str, text: str, allowed_gap: set[str]) -> bool:
    k, t = _tok(keyphrase), _tok(text)
    if not k:
        return False
    for start in (i for i, w in enumerate(t) if w == k[0]):
        j, gap, ok = start + 1, 0, True
        for w in k[1:]:
            while j < len(t) and t[j] != w and t[j] in allowed_gap and gap < MAX_GAP:
                j, gap = j + 1, gap + 1
            if j >= len(t) or t[j] != w:
                ok = False
                break
            j += 1
        if ok:
            return True
    return False


def patternless_applies(f: dict, answer: str, content, generic: frozenset) -> bool:  # noqa: ANN001
    kw = content(f.get("keyphrase", "")) - generic
    return len(kw) >= 2 and kw <= content(answer)


def support_match(pattern: str, text: str) -> bool:
    """A1b: reward_v5 semantics - re.search(pattern, text, re.I); reward_v5 matches case-insensitively since 2026-10-03."""
    try:
        return bool(pattern) and re.search(pattern, text or "", re.I) is not None
    except re.error:
        return False
