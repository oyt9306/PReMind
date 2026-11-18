"""Chunk-1 design fix (2026-10-03, production): per-concept consistency of the stored facts.

M5 ownership/header  a concept whose facts say someone else owns it (belongs to / owner / on loan / due back / borrowed
                     from) is shown with a minding header by construction (s2_memtext.assemble_text: "A plush toy the
                     user is looking after - <handle>."); the check fails when facts of ONE concept disagree (another
                     owner AND "the user bought/owns it") or an entry header still claims "the user's" for it.
M6 action conflict   instruction clauses (a placement verb - put/place/leave/keep/return/store/go/... - followed by a
                     destination "in/into/on/to/at <place>", negated and "rather than / instead of" parts dropped) from
                     two DIFFERENT facts of one concept with different destinations ("after play, put it in the dog
                     basket" vs "keep it away from the dog in a repair container"). Clauses inside one fact may be a
                     sequence and are not compared.
"""
from __future__ import annotations

import re

OTHER_OWNER = re.compile(
    r"\b(?:belongs? to (?!the user\b)|owned by (?!the user\b)|(?:its|his|her|their|[A-Z][a-z]+'s|[A-Z][a-z]+’s) "
    r"owner\b|on loan\b|(?:loaned|lent) to the user\b|borrowed (?:\w+ )?from\b|due back\b|is (?:only )?minding\b|"
    r"(?:minding|looking after) (?:it|her|him) for\b|not the user's\b)", re.I)
USER_OWNER = re.compile(r"\b(?:the user (?:bought|purchased|owns|inherited|won|received)\b|belongs to the user\b|"
                        r"the user's own\b|(?:a )?(?:gift|given) to the user\b)", re.I)
BASE = frozenset("put place leave keep return store go move seal hang pack stow lock set bring take".split())
HABIT = frozenset("puts places leaves keeps returns stores goes moves seals hangs packs stows locks brings takes".split())
PARTICIPLE = frozenset("put placed left kept returned stored moved sealed hung packed stowed locked set brought "
                       "taken".split())
MODALS = frozenset("must should can may will needs has is are ought".split())
_TRIGGER = re.compile(r"\b(?:after|until|once|whenever|rather than|instead of)\b", re.I)
_PREP = frozenset("in into inside on onto to at".split())
_STOP = frozenset("after before until when while so for because and or but rather instead pending once if during "
                  "away from with by near beside than of between".split())
_DET = frozenset("the a an its his her their this that separate regular usual marked own some".split())
_NEG = frozenset("not never no n't cannot".split())
_CUT = re.compile(r"\b(?:rather than|instead of|not in|not on)\b.*", re.I)


def _instruction_verb(low: list[str], i: int, trigger: bool) -> bool:
    """low[i] starts an instruction: imperative, modal (+ be + participle), requested "to <verb>", or a habitual
    present verb in a sentence with an after/until/rather-than trigger. Negated and past-tense uses do not."""
    w, prev = low[i], low[max(0, i - 3):i]
    if set(prev) & _NEG:
        return False
    if w in BASE and (i == 0 or low[i - 1] in MODALS or low[i - 1] == "to" and set(low[max(0, i - 6):i])
                      & {"asked", "told", "requested", "ask", "asks"} or low[i - 1] in ("and", "then")):
        return True
    if w in PARTICIPLE and i >= 2 and low[i - 1] == "be" and low[i - 2] in MODALS:
        return True
    return w in HABIT and trigger


def destinations(sentence: str) -> list[tuple[str, frozenset]]:
    """("put", destination content words) of every instruction clause in one fact sentence."""
    out = []
    trigger = bool(_TRIGGER.search(sentence))
    for clause in re.split(r"[;,:]|\band then\b", sentence):
        clause = _CUT.sub("", clause)
        low = [t.casefold() for t in re.findall(r"[A-Za-z'-]+", clause)]
        for i, w in enumerate(low):
            if not (w in BASE | HABIT | PARTICIPLE and _instruction_verb(low, i, trigger)):
                continue
            j = next((j for j in range(i + 1, min(len(low), i + 9)) if low[j] in _PREP), None)
            if j is None:
                continue
            dest = []
            for t in low[j + 1:j + 6]:
                if t in _STOP or t in _PREP:
                    break
                if t not in _DET:
                    dest.append(t)
            if dest:
                out.append(("put", frozenset(dest[:3])))
            break
    return out


def action_conflicts(facts: dict) -> list[list]:
    """M6: [key, slot_a, key, slot_b, "dest_a | dest_b"] per concept with conflicting placement instructions."""
    out = []
    for k, fs in sorted(facts.items()):
        clauses = [(s, v, d) for s, f in sorted(fs.items()) for v, d in destinations(f.get("text", ""))]
        hit = next(((a, b) for i, a in enumerate(clauses) for b in clauses[i + 1:]
                    if a[0] != b[0] and a[1] == b[1] and not (a[2] <= b[2] or b[2] <= a[2])), None)
        if hit:
            a, b = hit
            out.append([k, a[0], k, b[0], f"{' '.join(sorted(a[2]))} | {' '.join(sorted(b[2]))}"])
    return out


def other_owner(texts: list[str]) -> bool:
    return any(OTHER_OWNER.search(t or "") for t in texts)


def ownership_problems(facts: dict, texts: dict, kinds: dict | None = None) -> list[list]:
    """M5: [key, slot_other, key, slot_user] for facts disagreeing on ownership; [key, "header"] for an entry whose
    header claims the user's ownership while its facts name another owner."""
    out = []
    for k, fs in sorted(facts.items()):
        if (kinds or {}).get(k) == "person":                      # people are not owned ("belongs to a choir")
            continue
        oth = [s for s, f in sorted(fs.items()) if OTHER_OWNER.search(f.get("text", ""))]
        usr = [s for s, f in sorted(fs.items()) if USER_OWNER.search(f.get("text", ""))]
        if oth and usr:
            out.append([k, oth[0], k, usr[0]])
        head = (texts.get(k) or "").split("\n")[0].split(".")[0]
        if oth and re.search(r"\bthe user's\b", head, re.I):
            out.append([k, "header"])
    return out


def instruction_errors(sentence: str) -> bool:
    """Memtext background facts must not instruct a placement (production memtext validation)."""
    return bool(destinations(sentence))


def production_checks(fam: dict, facts: dict, texts: dict) -> dict:
    from final_construction.wp_v9_vcm import s4_coherence
    kinds = {k: c["kind"] for k, c in fam["concepts"].items()}
    voice, act = s4_coherence.voice_problems(facts, texts), action_conflicts(facts)
    own = ownership_problems(facts, texts, kinds)
    return {"M4_voice": {"pass": not voice, "detail": voice[:8]},
            "M5_ownership_header": {"pass": not own, "detail": own[:8]},
            "M6_action_conflict": {"pass": not act, "detail": act[:8]}}


def header(concept: dict, readable_type: str, sentences: list[str]) -> str:
    """Production entry header (M5 by construction): "the user's" unless a fact names another owner / a loan."""
    minded = other_owner(sentences)
    if concept.get("handle"):
        return (f"A {readable_type} the user is looking after \u2014 {concept['handle']}." if minded
                else f"The user's {readable_type} \u2014 {concept['handle']}.")
    rel = concept["relation"] or readable_type
    if minded and concept.get("kind") == "pet":
        return f"{concept['name']}: a {rel} the user is looking after."
    return f"{concept['name']}: the user's {rel}."
