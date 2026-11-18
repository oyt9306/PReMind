"""M7 scene anchor (owner spot-check of chunk 2, 2026-10-03; production): real memories are scene-independent.

Scene vocabulary of a family = from its queries (G1_visual -> slot s1, G1G2_visual -> slot s2, G0_closed -> all):
the action verbs (after "can/may/should I", "to", or a temporal gerund "after tidying"), the place words (heads of
in/on/at/into/near/... phrases) and the time-clause words (after/before/while/when/during/until ...), all stemmed,
minus every concept's type words, handle, name and the category nouns (the objects themselves are not the scene).
Rules (per sentence):
  non-decisive facts (o1/o2/x of every concept = all facts of non-look-alike concepts and every memtext extra):
      fail on a temporal clause with a query time word ("... after outings", "during toddler playtime") or with a
      scene place while the sentence uses a scene action, or on a scene action together with a scene place;
  decisive facts (s1/s2 of targets and look-alikes, against their own slot's query): fail on a moment-specific anchor
      - a temporal clause with a definite scene noun, a scene noun + passive ("after the counter is cleared", "while
      the room is aired") or now/today/tonight - and on scene wording in MORE than one sentence of the fact.
      General conditions ("after use", "after play", "after outings") pass.
"""
from __future__ import annotations

import re

STOP = frozenset("the a an this that these those it its them they their his her my your our i we you he she to of and "
                 "or for with now here there each every some any all be is are was were can may should could would will "
                 "do does did have has had not no yes where what which who whom whose how why if then than so as at "
                 "in on into onto by near from under up out over off before after while when during until once one "
                 "both either user users item items thing things".split())
_PREP = r"(?:in|on|at|into|onto|near|by|beside|under|inside|around|across|through|from|to|next to)"
_DET = r"(?:the|a|an|this|that|these|those|its|their|his|her|my|your|our)"
_TEMPORAL = re.compile(r"\b(?:after|before|while|when|whenever|during|until|once|as soon as)\b([^.;,:!?]*)", re.I)
_PASSIVE = re.compile(r"\b(?:the|this|that|these|those)\s+([a-z'-]+(?:\s+[a-z'-]+)?)\s+(?:is|are|gets|get|has been|"
                      r"have been|was|were)\s+[a-z]+(?:ed|en)\b", re.I)
_NOW = re.compile(r"\b(?:tonight|today|right now|this (?:morning|afternoon|evening|weekend))\b", re.I)
_ACTION = re.compile(rf"\b(?:can|may|should|could|shall|must)\s+(?:i|we|they|he|she|{_DET}\s+[a-z-]+)\s+([a-z-]+)|"
                     r"\bto\s+([a-z-]+)|\b(?:after|before|while|when)\s+([a-z-]+ing)\b", re.I)
_PLACE = re.compile(rf"\b{_PREP}\s+(?:{_DET}\s+)?([a-z'-]+(?:\s+[a-z'-]+)?)", re.I)


def stem(w: str) -> str:
    w = w.casefold().strip("'")
    for _ in range(2):
        w = re.sub(r"(?:ing|ed|es|s)$", "", w) if len(w) > 4 else w
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def _stems(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z'-]+", (text or "").casefold()) if w not in STOP and len(w) > 2}


def own_words(fam: dict, category_nouns: list[str]) -> set[str]:
    from final_construction.wp_v9_vcm import s2_memtext
    from final_construction.wp_v9_vcm import taxonomy as tx
    out = set()
    for c in fam["concepts"].values():
        ft = c.get("fine_type", "")
        out |= _stems(" ".join([ft.replace("_", " "), s2_memtext.readable(ft), c.get("handle", ""), c.get("name", ""),
                                c.get("relation", "")] + sorted(tx.CATEGORY_LEXICON.get(ft, frozenset()))))
    return out | _stems(" ".join(category_nouns))


def scene_of(fam: dict, items: list[dict]) -> dict[str, dict[str, set[str]]]:
    """slot ("s1" | "s2" | "all") -> {"action", "place", "time"} stems from the family's queries."""
    own = own_words(fam, [n for it in items for n in it.get("category_nouns", [])])
    out = {s: {"action": set(), "place": set(), "time": set()} for s in ("s1", "s2", "all")}
    for it in items:
        q = it.get("query") or ""
        slot = {"G1_visual": "s1", "G1G2_visual": "s2"}.get(it.get("row_type"))
        found = {"action": {stem(w) for m in _ACTION.finditer(q) for w in m.groups() if w} - own,
                 "place": {s for m in _PLACE.finditer(q) for s in _stems(m.group(1))} - own,
                 "time": {s for m in _TEMPORAL.finditer(q) for s in _stems(m.group(1))} - own}
        for k, v in found.items():
            out["all"][k] |= v
            if slot:
                out[slot][k] |= v
    return out


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text or "") if s.strip(" -")]


def _scene_sentence(sent: str, sc: dict[str, set[str]]) -> str | None:
    words = _stems(sent)
    for m in _TEMPORAL.finditer(sent):
        clause = _stems(m.group(1))
        if clause & sc["time"] or clause & sc["place"] and words & sc["action"]:
            return f"scene time clause {m.group(0).strip()!r}"
    if words & sc["action"] and words & sc["place"]:
        return f"scene action + place {sorted(words & (sc['action'] | sc['place']))}"
    return None


def _moment(sent: str, sc: dict[str, set[str]]) -> str | None:
    if _NOW.search(sent):
        return f"moment-specific anchor {_NOW.search(sent).group(0)!r}"
    for m in _TEMPORAL.finditer(sent):
        clause = m.group(0)
        scene = sc["place"] | sc["time"] | sc["action"]
        definite = {stem(w) for w in re.findall(r"\b(?:the|this|that|these|those)\s+([a-z'-]+)", clause, re.I)}
        passive = {s for p in _PASSIVE.finditer(clause) for s in _stems(p.group(1))}
        if _NOW.search(clause) or passive & scene or definite & (sc["place"] | sc["time"]):
            return f"moment-specific anchor {clause.strip()!r}"
    return None


def scene_anchor_problems(fam: dict, facts: dict, items: list[dict]) -> list[list]:
    """M7: [key, slot, reason] per offending fact."""
    from final_construction.wp_v9_vcm import s2_author
    sc = scene_of(fam, items)
    group = set(s2_author.target_group(fam))
    out = []
    for k, fs in sorted(facts.items()):
        for slot, f in sorted(fs.items()):
            sents = _sentences(f.get("text", ""))
            if k in group and slot in ("s1", "s2"):
                own = sc[slot]
                bad = next((r for s in sents if (r := _moment(s, own))), None)
                if not bad and sum(bool(_scene_sentence(s, own)) for s in sents) > 1:
                    bad = "scene wording in more than one sentence"
            else:
                bad = next((r for s in sents if (r := _scene_sentence(s, sc["all"]))), None)
            if bad:
                out.append([k, slot, bad])
    return out


def memtext_errors(fam: dict, got: dict, items: list[dict]) -> list[str]:
    """Memtext validation: extras must not mention the scene (retry)."""
    facts = {k: {f["slot"]: {"text": f["sentence"]} for f in fs} for k, fs in got.items()}
    return [f"{k}/{s}: {r}; write a scene-independent background fact (history, condition, routine)"
            for k, s, r in scene_anchor_problems(fam, facts, items)]
