"""personal_content matcher for reward_v5.1 (owner 2026-10-04: the strict phrase regex misses paraphrases).

``hit(pattern, answer, mode)``: "strict" = the support regex itself (reward_v5.0). "soft" = strict OR a deterministic
content-word match: every proper name in the pattern appears in the answer, a negated pattern needs a negation in
the answer, and at least SOFT_COVERAGE of the pattern's content units appear (a unit is a content word outside any
group, or a (?:a|b) group satisfied by any one alternative). Words match by their first STEM_LEN letters.
"""
from __future__ import annotations

import re

SOFT_COVERAGE = 0.6
STEM_LEN = 5
STOP = frozenset("a an the to of and or for in on at with by be is are it its this that your you their his her him "
                 "from as do does did if so any all one ones them they she he we our was were has have had into onto "
                 "off out up use".split())
NEG = re.compile(r"\b(?:not|never|no|avoid|don't|do not|doesn't|mustn't|shouldn't|without)\b", re.I)
_GROUP = re.compile(r"\(\?:([^()]*)\)")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")


def _words(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text) if w.lower() not in STOP and len(w) > 2]


def _clean(pattern: str) -> str:
    p = re.sub(r"\(\?[a-z]+\)", " ", pattern)           # inline flags
    p = re.sub(r"\\[bBsSwWdD]|\\.", " ", p)              # escapes
    return re.sub(r"[\^$*+?{}\[\].]", " ", p)


def _units(pattern: str) -> tuple[list[str], list[list[list[str]]]]:
    """(content words outside groups, groups as lists of alternatives' content words)."""
    p = _clean(pattern)
    groups = []
    while True:
        m = _GROUP.search(p)
        if not m:
            break
        groups.append([_words(alt) for alt in m.group(1).split("|")])
        p = p[:m.start()] + " " + p[m.end():]
    p = re.sub(r"[()|]", " ", p)
    return _words(p), [g for g in groups if any(g)]


def _has(word: str, ans_stems: set[str]) -> bool:
    return word.lower()[:STEM_LEN] in ans_stems


def soft_hit(pattern: str, answer: str) -> bool:
    words, groups = _units(pattern)
    ans_stems = {w.lower()[:STEM_LEN] for w in _TOKEN.findall(answer)}
    names = [w for w in words if w[0].isupper()]
    if any(not _has(n, ans_stems) for n in names):
        return False
    if NEG.search(_clean(pattern).replace("'", "'")) and not NEG.search(answer):
        return False
    units = [_has(w, ans_stems) for w in words]
    units += [any(alt and all(_has(w, ans_stems) for w in alt) for alt in g) for g in groups]
    return bool(units) and sum(units) / len(units) >= SOFT_COVERAGE


def hit(pattern: str, answer: str, mode: str = "soft") -> bool:
    if re.search(pattern, answer, re.I):
        return True
    if mode == "strict":
        return False
    if mode != "soft":
        raise ValueError(f"personal_match mode {mode!r}")
    return soft_hit(pattern, answer)
