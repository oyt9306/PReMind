"""CPU phrase scorer v3 (2026-10-17): habit/contrast-aware successor of rebuild_v18_codex/qa_rules_v2.py.

Same score(answer, keys) API and the same negation scope / sentence / 'but'/'however' clause logic as v2
(imported, v2 is not edited). New: a lure key occurring only inside a habit/contrast clause
("This is different from its usual position, lying on its side"; "It is usually kept on the bedside table")
does NOT count as primed when the answer also states the gold. Normative/rule clauses ("should be in the
bottle rack", "his agreement requires him to sleep in his dog bed") and relocation directives
("move it to its proper place on the stand", "put it back in the storage box") are treated the same way. A lure asserted in a plain clause still primes,
and an answer that omits the gold and only mentions the lure inside a habit clause is still primed
(it substitutes the habit for the present).

Clause = v2 piece (split on . ! ? ; but however). A clause is a habit clause when it matches HABIT.
Gold hits are counted anywhere, exactly as in v2.
"""
import re
from .rebuild_v18_codex import qa_rules_v2 as V2
from .rebuild_v17_codex_r2.qa_rules import normalize

HABIT = re.compile(
    r"\b(?:usually|normally|typically|usual|tends? to|is kept|are kept|different from|instead of its|"
    r"not in its usual|habitually|customarily|used to|ordinarily|regular (?:spot|place|position)|"
    r"normal (?:spot|place|position)|according to (?:the |your )?memor(?:y|ies)|(?:the )?memory (?:says|states|notes|mentions)|"
    # normative/rule clauses describe where it ought to be, not where it is now
    r"should (?:be|sleep|stay|go|live|remain)|supposed to|belongs? (?:in|on|to)|required to|requir(?:e|es|ed|ing) (?:him|her|it|them|that)|"
    # relocation directives say where it should go, i.e. it is not there now
    r"(?:move|put|place|return|bring|take|guide|keep)\s+(?:it|them|him|her)\s+(?:back\s+)?(?:to|in|into|on|onto|inside)|"
    r"put (?:it |them )?(?:back|away)|(?:proper|designated) (?:place|spot))\b",
    re.I)


def _clauses(text):
    """v2 clause split; each entry = (normalized words, followed_by_but, is_habit_clause)."""
    pieces = re.split(r'([.!?;]|\bbut\b|\bhowever\b)', str(text), flags=re.I)
    return [(normalize(pieces[n]), n + 1 < len(pieces) and pieces[n + 1].lower() == 'but', bool(HABIT.search(pieces[n])))
            for n in range(0, len(pieces), 2)]


def lure_occurrences(text, keys):
    """key -> set of clause kinds ('current'/'habit') with a non-negated occurrence (v2 negation rules)."""
    out = {}
    clauses = _clauses(text)
    for key in keys:
        phrase = normalize(key)
        if not phrase:
            continue
        for words, before_but, habit in clauses:
            for i in range(len(words) - len(phrase) + 1):
                if words[i:i + len(phrase)] != phrase:
                    continue
                if (V2._negative(words[max(0, i - 6):i]) or (before_but and 'not' in words[:i])) and not V2._negative(phrase):
                    continue
                out.setdefault(key, set()).add('habit' if habit else 'current')
    return out


def score(answer, keys):
    """v2-compatible verdict. Extra fields: lure_hits_current, lure_hits_habit_only, primed_by_omission."""
    gold = list(V2.matches(answer, keys['gold_keys']))
    occ = lure_occurrences(answer, keys['lure_keys'])
    current = [k for k, s in occ.items() if 'current' in s]
    habit_only = [k for k, s in occ.items() if 'current' not in s]
    by_omission = bool(habit_only) and not gold and not current
    primed = bool(current) or by_omission
    return {'qa_correct': bool(gold) and not primed, 'qa_primed': primed, 'gold_hits': gold,
            'lure_hits': current + (habit_only if by_omission else []),
            'lure_hits_current': current, 'lure_hits_habit_only': habit_only, 'primed_by_omission': by_omission}


matches = V2.matches
extract = V2.extract
__all__ = ['HABIT', 'score', 'matches', 'extract', 'lure_occurrences']
