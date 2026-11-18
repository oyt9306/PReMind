"""P6 naming checks (active only when concept labels v2 are applied, i.e. objects carry a `handle`).

N1 object naming   an object concept's memory text / facts use no proper-name-like token AS the object's designation
                   (pilot fix 2026-10-03, see `designations`); gold answers: no assigned-name token that is not a
                   person/pet of this family. Whitelist: the family's person and pet names, people's names in
                   possessives ("Jisu's"), capitalized tokens of the family's handles, I, months and weekdays.
N2 handle rules    handles unique within the family; no colour/pattern/material/location word, <= 6 words after the
                   type noun (s1_labels.handle_problems); no overlap with the concept's DECISIVE (s1/s2)
                   keyphrases (content-word Jaccard >= 0.5).
I1 (extension)     a query must not contain an object's handle (>= 2 of its content words) - it would pinpoint it.
"""
from __future__ import annotations

import re

from final_construction.wp_v9_vcm import taxonomy as tx

ASSIGNED = frozenset(n for lst in tx.PERSON_NAMES.values() for n in lst) | frozenset(tx.PET_NAMES) | \
    frozenset(tx.OBJECT_NAMES)
CALENDAR = frozenset("January February March April May June July August September October November December Monday "
                     "Tuesday Wednesday Thursday Friday Saturday Sunday".split())
SUBJECT_VERB = re.compile(r"(?:^|(?<=[.!?;:\n-])\s*)([A-Z][a-z]+)\s+(?:is|was|has|had|sits|stays|lives|belongs|needs|"
                          r"must|can|cannot|should|gets|goes|comes|holds|keeps)\b")
STOP = frozenset("the one a an from of my at".split())
DECISIVE_SLOTS = ("s1", "s2")
DETERMINERS = ("This", "That", "It", "The", "My", "Our", "Each", "Every", "Its", "A", "An", "After", "Before", "When",
               "If", "Since", "Because", "During", "Once", "While")
DESIGNATION = re.compile(r"\b(?:(?:named|called|nicknamed)(?:\s+it)?|calls?\s+it)\s+([A-Z][a-z]+)\b|^([A-Z][a-z]+):|"
                         r"\b([A-Z][a-z]+) is (?:my|our|the user's|their|his|her)\b", re.M)
PLACE_BEFORE = re.compile(r"\b(?:in|to|from|at|near|of|visit|visited|visiting|via|around|across|through|for)\s*$")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").casefold()))


def active(fam: dict) -> bool:
    return any(c.get("handle") for c in fam["concepts"].values())


def allowed_caps(fam: dict) -> set[str]:
    ok = {c["name"] for c in fam["concepts"].values() if c["name"] and c["kind"] in ("person", "pet")}
    ok |= {w.removesuffix("'s") for c in fam["concepts"].values() for w in re.findall(r"[A-Z][a-z]+(?:'s)?",
                                                                                       c.get("handle", ""))}
    return ok | CALENDAR | {"I"}


def name_like(text: str, allowed: set[str], strict: bool = True) -> list[str]:
    allowed = allowed | {d + "s" for d in CALENDAR}                         # "on Fridays"
    hits = [t for t in re.findall(r"\b([A-Z][a-z]+)\b", text or "") if t in ASSIGNED and t not in allowed]
    if strict:
        for m in re.finditer(r"(?<=[a-z,;]\s)([A-Z][a-z]+)(?!'s|’s)\b", text or ""):
            if m.group(1) not in allowed:
                hits.append(m.group(1))
        hits += [m.group(1) for m in SUBJECT_VERB.finditer(text or "") if m.group(1) not in allowed
                 and m.group(1) not in ("This", "That", "It", "The", "My", "Our", "Each", "Every", "Its")]
    return sorted(set(hits))


def designations(sentences: list[str], allowed: set[str]) -> list[str]:
    """Pilot fix: proper-name-like tokens used AS the object's designation - a legacy object name anywhere, a
    capitalised subject ("Lulu sits ..."), a line header ("Lulu: ..."), "named/called Lulu", "Lulu is my ...", or a
    capitalised token repeated in >= 2 sentences outside a place phrase. Place names ("in Japan"), months, weekdays
    and brand words inside a fact are not object names."""
    allowed = allowed | {d + "s" for d in CALENDAR}
    hits = set()
    for sent in sentences:
        hits |= {t for t in re.findall(r"\b([A-Z][a-z]+)\b", sent) if t in tx.OBJECT_NAMES}
        hits |= {m.group(1) for m in SUBJECT_VERB.finditer(sent) if m.group(1) not in DETERMINERS}
        hits |= {next(g for g in m.groups() if g) for m in DESIGNATION.finditer(sent)}
    seen: dict[str, int] = {}
    for sent in sentences:
        for m in re.finditer(r"(?<=[a-z,;]\s)([A-Z][a-z]+)(?!'s|’s)\b", sent):
            if not PLACE_BEFORE.search(sent[:m.start()]):
                seen[m.group(1)] = seen.get(m.group(1), 0) + 1
    hits |= {t for t, n in seen.items() if n >= 2}
    return sorted(h for h in hits if h not in allowed)


def object_naming(fam: dict, facts: dict, texts: dict) -> list[list]:
    allowed = allowed_caps(fam)
    out = []
    for k, c in fam["concepts"].items():
        if c["kind"] != "object":
            continue
        body = [ln for ln in (texts.get(k, "") or "").split("\n")]
        sents = {x.strip(" -") for b in body + [f.get("text", "") for f in facts.get(k, {}).values()]
                 for x in re.split(r"(?<=[.!?])\s+", b) if x.strip(" -")}
        hits = designations(sorted(sents), allowed)
        if hits:
            out.append([k, hits])
    return out


def gold_names(fam: dict, gold: str) -> list[str]:
    """Assigned-name tokens in a gold answer that are not a person/pet of this family (objects never get names)."""
    return name_like(gold, allowed_caps(fam), strict=False)


def handle_rules(fam: dict, facts: dict) -> list[list]:
    from final_construction.wp_v9_vcm import s1_labels
    out, seen = [], {}
    for k, c in sorted(fam["concepts"].items()):
        h = c.get("handle", "")
        if c["kind"] != "object":
            continue
        if not h:
            out.append([k, "missing_handle"])
            continue
        if h in seen:
            out.append([k, f"duplicate_of:{seen[h]}"])
        seen.setdefault(h, k)
        out += [[k, p] for p in s1_labels.handle_problems(h)]
        hw = _words(h) - STOP
        for slot, f in facts.get(k, {}).items():
            if slot not in DECISIVE_SLOTS:          # o1/o2/x extra facts may share words with the handle
                continue
            kw = _words(f.get("keyphrase", "")) - STOP
            if hw and kw and len(hw & kw) / len(hw | kw) >= 0.5:
                out.append([k, f"handle_overlaps_keyphrase:{slot}"])
    return out


def handle_in_query(fam: dict, query: str) -> list[str]:
    """>= 2 handle content words in the query; a one-content-word handle ("the one from the move") counts only as its
    phrase ("from the move") or a capitalized word ("Lisbon") - the bare word is often a verb ("may I move ...")."""
    qw, q = _words(query), " ".join(re.findall(r"[a-z]+", (query or "").casefold()))
    out = []
    for c in fam["concepts"].values():
        h = c.get("handle", "")
        cw = _words(h) - STOP
        if not h or not cw:
            continue
        phrase = " ".join(re.findall(r"[a-z]+", re.sub(r"^the (?:one )?", "", h.casefold())))
        caps = {w.casefold() for w in re.findall(r"\b[A-Z][a-z]+", h)}
        if len(cw & qw) >= 2 or (len(cw) == 1 and (f" {phrase} " in f" {q} " or cw & caps & qw)):
            out.append(h)
    return out
