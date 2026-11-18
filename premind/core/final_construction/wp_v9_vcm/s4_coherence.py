"""P8 deterministic coherence guards for memory fact sentences (precheck, memtext validation, assemble).

dangling pronoun   "them" with no plausible antecedent before it in the same sentence (clauses split by ',' / ';' stay
                   in the sentence): "Eva prefers someone else to sort them for children's craft supplies" fails.
condition clash    a permission/ability stated as unrestricted BECAUSE OF a restricting condition ("Eva can kneel without
                   restriction because of a knee condition"), or a negated restriction justified by a condition.
M4 voice (production)  memory entries are third person to match the header ("The user's figure - ..."): no I / me /
                   my / mine / myself / I'm / I've / I'd / I'll in fact sentences (quoted speech excluded).
These are heuristics on top of review item 11 (memory_text_incoherent), which judges every shown memory text.
"""
from __future__ import annotations

import re

_SENT = re.compile(r"[^.!?\n]+")
_NOT_PLURAL = frozenset("its this his was has is us yes plus less unless always sometimes perhaps upstairs downstairs "
                        "always afterwards towards across".split())
_CLASH = (re.compile(r"\b(?:can|may|is allowed to|is free to|is fine to|is okay to)\b[^.]{0,60}\bwithout (?:any )?"
                     r"(?:restrictions?|limits?|limitations?|problems?)\b[^.]{0,40}\bbecause of\b[^.]{0,40}"
                     r"\b(?:condition|injur\w*|allerg\w*|surgery|illness|disease|pain|fracture)\b", re.I),
          re.compile(r"\b(?:no|never|without)\b[^.]{0,30}\brestrictions?\b[^.]{0,40}\bbecause of\b[^.]{0,40}"
                     r"\b(?:condition|injur\w*|allerg\w*|surgery|illness|disease|pain|fracture)\b", re.I))


MASS_NOUNS = frozenset("footwear clothing clothes luggage supplies equipment groceries glasses shoes people children kids "
                       "toys dishes cutlery papers keys books paramedics staff crew family".split())
QUANTIFIERS = frozenset("both all these those several".split())
PERSON_NOUNS = frozenset("coworker colleague friend neighbour neighbor classmate guest visitor child kid person owner nurse "
                         "doctor teacher relative cousin roommate partner sister brother parent mother father son "
                         "daughter niece nephew vet driver courier".split())       # singular "they"
SUBJECTS = frozenset("he she it who which that one".split())
NOT_NAMES = frozenset("the this that my our his her their its these those a an some all both".split())


def _referent_before(toks: list[str]) -> bool:
    """A plausible antecedent for "them" precedes it (pilot fix: 8/8 first-pass M3 flags were false positives).
    Counts: mass/collective nouns, quantifiers, person nouns (singular they), and -s words that are not a verb right
    after a name or subject pronoun ("Eva prefers ..." is a verb; "library books", "toy-house screws" are nouns)."""
    for i, t in enumerate(toks):
        w = t.casefold()
        if w in MASS_NOUNS or w in QUANTIFIERS or w in PERSON_NOUNS:
            return True
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "'s")) and w not in _NOT_PLURAL:
            prev = toks[i - 1] if i else ""
            if (prev[:1].isupper() and prev.casefold() not in NOT_NAMES) or prev.casefold() in SUBJECTS:
                continue
            return True
    return False


def sentence_problems(text: str) -> list[str]:
    out, before = [], []                                   # earlier sentences of the same fact count (pilot fix)
    for sent in _SENT.findall(text or ""):
        toks = re.findall(r"[A-Za-z']+", sent)
        low = [t.casefold() for t in toks]
        if "them" in low and not (_referent_before(toks[:low.index("them")]) or (before and _referent_before(before))):
            out.append(f"dangling 'them': {sent.strip()[:80]!r}")
        if any(p.search(sent) for p in _CLASH):
            out.append(f"condition clash: {sent.strip()[:80]!r}")
        before += toks
    return out


def incoherent_facts(facts: dict) -> list[list]:
    """[concept, slot, problem] for every fact sentence failing a guard."""
    return [[k, s, p] for k, fs in sorted(facts.items()) for s, f in sorted(fs.items())
            for p in sentence_problems(f.get("text", ""))]


_FIRST = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|I\u2019m|I\u2019ve|I\u2019d|I\u2019ll)\b|\b(?:me|my|mine|myself)\b", re.I)
_QUOTED = re.compile(r"\"[^\"]*\"|\u201c[^\u201d]*\u201d")


def first_person(text: str) -> list[str]:
    """First-person tokens outside quoted speech ("I" must be capitalised: the pronoun)."""
    t = _QUOTED.sub(" ", text or "")
    return sorted({m.group(0) for m in _FIRST.finditer(t) if not m.group(0).startswith("i") or len(m.group(0)) > 1
                   and m.group(0).casefold() in ("me", "my", "mine", "myself")})


def voice_problems(facts: dict, texts: dict) -> list[list]:
    """M4: [concept, slot | "entry", tokens] for first-person fact sentences / entry texts."""
    out = [[k, s, hits] for k, fs in sorted(facts.items()) for s, f in sorted(fs.items())
           if (hits := first_person(f.get("text", "")))]
    bad = {k for k, _, _ in out}
    out += [[k, "entry", hits] for k, t in sorted(texts.items()) if k not in bad and (hits := first_person(t))]
    return out
