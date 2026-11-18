"""Stage 4b (CPU): deterministic checks I1-I9 + A1/A2/A4 + leak + lexicon (a row failing any check is excluded,
never patched). Every check returns {"pass": bool, "detail": ...}.

I1/I2  query (owner feedback: precise queries): no assigned name, no possessive/deictic pinpointing of one instance
       ("my/this/that <category noun>"; the shared category noun of the look-alike group and generic terms are allowed),
       no colour/pattern word, no brand text of a target, no target/look-alike fact keyphrase (F5: verbatim, or all of
       its >= 2 content words present - precise-action queries legitimately name the constrained action).
I2b    text shortcut (F5, "use" rows): the query must not overlap the target's used fact more than every look-alike's
       same-slot fact (overlap >= 2 content words) - otherwise the text alone identifies the target.
Q1     length floor: action rows >= 8 words; closed rows (G0, general rule) >= 5 words, exempt from action precision (F3).
M2     no copy-pasted memory facts: no two facts of a family near-identical once names are masked (token Jaccard >= 0.6
       or identical), and every non-target concept carries >= 1 instance-specific fact (owner, P5).
C1     commonsense: no stored fact and no gold may license a denylisted risky action (pet-toxic food permitted,
       fragile/antique/crystal/wooden item declared dishwasher-fine).
Gate (F2): deterministic checks + the GPT author's self-check fields only; the local Qwen self-check fields are stored
       as diagnostics (row["selfcheck_local"]) and are never conjuncts.
I3     every fact keyphrase verbatim in its concept's memory text.
I4     look-alikes share the target's fine type (or relaxed stratum flagged); matched s1/s2 contrast facts with a
       different action implication; swap proxy (F4, independent of A1): no look-alike's same-slot pattern matches.
I5     exactly K entries, distinct, one seeded display permutation per family (shared by every pair member).
I6     labels re-derived from the stored context of each arm equal the stored labels.
I7     no identity id / dataset concept name / path / role marker / person or row id / concept key in model text.
I8     train/dev split only, no holdout path.   I9  Option-B risk (no 'uncertain'), legal B7 cell, A-2.
A1     gold applies every target (2-concept: both).   A2  only the used slot is applied, nothing from other
       concepts; acute hazard addressed first.   A4  GT-absent reply clarifies, guesses no stored fact.
"""
from __future__ import annotations

import re

from final_construction.wp_v9_vcm import taxonomy as tx

STOP = frozenset("the a an my your his her their of to and or in on at for with each every is are be this that it "
                 "only not no should must can may might will would could shall does did done has have had was were "
                 "been being never always".split())
_ID_PAT = re.compile(r"(?<![\w:])(?:[A-Za-z][A-Za-z0-9]*[_/][A-Za-z0-9_/.-]+|[A-Za-z]+:[A-Za-z][\w-]+)")
_HEX = re.compile(r"\b[0-9a-f]{16,}\b")
_FILE = re.compile(r"\.(?:png|jpe?g|webp|json|py)\b", re.I)
_PERSON_ID = re.compile(r"\bs2[a-z]\d{4}\b|\bperson:")
_KEY = re.compile(r"\bC\d{2}\b")
ALLOWED_SLASH = frozenset({"and/or", "either/or", "on/off", "w/o"})
POINTERS = r"(?:my|our|your|his|her|their|this|that)"
MIN_QUERY_WORDS = 8
MIN_CLOSED_QUERY_WORDS = 5
CLOSED_ROWS = ("G0_closed", "general_rule_closed")
_PET_TOXIC = r"(?:chocolate|grapes?|raisins?|onions?|garlic|xylitol|macadamia|avocado|alcohol|coffee|caffeine|chives)"
_PERMIT = r"(?:can (?:have|eat)|may (?:have|eat)|is allowed|allowed to (?:have|eat)|fine to (?:eat|have)|safe to eat|" \
          r"loves eating|gets? (?:a |some )?(?:treat|piece|bite)s? of)"
_DISHWASHER_OK = r"(?:dishwasher[- ]safe|fine in the dishwasher|can go (?:in|into) the dishwasher|" \
                 r"put (?:it|them) (?:in|into) the dishwasher)"
_FRAGILE = r"(?:crystal|hand[- ]painted|antique|gold[- ]rim\w*|cast[- ]iron|wooden|lacquer\w*|porcelain heirloom)"
DENYLIST = (re.compile(rf"{_PERMIT}[^.]{{0,60}}{_PET_TOXIC}|{_PET_TOXIC}[^.]{{0,40}}(?:is|are) (?:fine|safe|ok|okay)", re.I),
            re.compile(rf"{_FRAGILE}[^.]{{0,80}}{_DISHWASHER_OK}|{_DISHWASHER_OK}[^.]{{0,80}}{_FRAGILE}", re.I))


from final_construction.wp_v9_vcm.s4_critical import _CRITICAL, _SOFT, decision_critical  # noqa: E402,F401  (R1)
from final_construction.wp_v9_vcm.s4_keyphrase import support_match  # noqa: E402  (A1b, reward_v5 semantics)
# R4: templated, non-natural fact phrasing ("Vera's rule for moving the chairs without asking is: you may move them.")
_TEMPLATED = (re.compile(r"(?:'s|\u2019s) (?:rule|policy|instruction|status|setting|guideline|answer|permission)s? "
                         r"(?:for|on|about|regarding) [^.:]{1,80}?(?: is)?\s*:", re.I),
              re.compile(r"\b(?:rule|policy|instruction|answer) (?:for|on) [^.]{1,80}? is\s*:", re.I),
              re.compile(r":\s*(?:you|they|it) (?:may|must|can|should)\b", re.I))


def templated_hits(text: str) -> list[str]:
    return [m.group(0) for p in _TEMPLATED for m in p.finditer(text or "")]


def ok(passed: bool, detail: object = None) -> dict:
    return {"pass": bool(passed), "detail": detail}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def content(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 2 and w not in STOP}


def rx(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(pattern, re.I) if len(pattern.strip()) >= 4 else None
    except re.error:
        return None


def matches(pattern: str, text: str) -> bool:
    p = rx(pattern)
    return bool(p and p.search(text))


def overlap_applies(f: dict, gold: str) -> bool:
    """>= 60% of the fact keyphrase's content words occur in the gold."""
    kw = content(f.get("keyphrase", ""))
    return bool(kw) and len(kw & content(gold)) >= 0.6 * len(kw)


def applies(f: dict, gold: str) -> bool:
    """A fact is applied by the gold: its regex matches (GPT facts), else (memory-text facts, no regex) every
    distinctive keyphrase content word occurs (s4_keyphrase; the 60% overlap matched generic words)."""
    if f.get("application_pattern"):
        return matches(f["application_pattern"], gold)
    from final_construction.wp_v9_vcm import s4_keyphrase
    return s4_keyphrase.patternless_applies(f, gold, content, GENERIC_WORDS)


def family_checks(fam: dict, facts: dict, texts: dict) -> dict:
    from final_construction.wp_v9_vcm import s2_local
    res = {}
    from final_construction.wp_v9_vcm import s4_keyphrase                 # pilot fix: designation gaps allowed
    missing = s4_keyphrase.missing_keyphrases(fam, facts, texts)
    res["I3_keyphrase_verbatim"] = ok(not missing, missing)
    struct = [(k, sorted(facts.get(k, {}))) for k in fam["concepts"]
              if sorted(facts.get(k, {})) != sorted(s2_local.slots_for(fam, k)) or not texts.get(k, "").strip()]
    res["M1_matched_fact_structure"] = ok(not struct, struct)
    visual = [k for k, c in fam["concepts"].items() if c["role"] in ("target", "negative", "replacement")
              and set(words(texts.get(k, ""))) & (tx.COLOR_WORDS | tx.PATTERN_WORDS)]
    res["memory_text_no_visual_words"] = ok(not visual, visual)
    i4 = []
    for t, info in fam["target_info"].items():
        for n in info["negatives"] + ([info["replacement"]] if info["replacement"] else []):
            same = fam["concepts"][n]["fine_type"] == fam["concepts"][t]["fine_type"] or (
                fam["concepts"][t]["kind"] == "person" and fam["concepts"][n]["kind"] == "person")
            if not same and not info["relaxed"]:
                i4.append((n, "type_mismatch_not_flagged"))
            for slot in ("s1", "s2"):
                a, b = facts.get(t, {}).get(slot), facts.get(n, {}).get(slot)
                if not a or not b or " ".join(words(a["action_implication"])) == " ".join(words(b["action_implication"])):
                    i4.append((n, slot, "no_contrasting_fact"))
    res["I4_lookalike_contrast"] = ok(not i4, i4)
    bad_rx = [(k, s) for k, fs in facts.items() for s, f in fs.items() if s in ("s1", "s2")
              and rx(f.get("application_pattern", "")) is None]
    res["patterns_compile"] = ok(not bad_rx, bad_rx)
    res["I5_display_perm_shared"] = ok(sorted(fam["display_perm"]) == list(range(fam["k"])))
    risky = [(k, s, commonsense_flags(f.get("text", "") + " " + texts.get(k, ""))) for k, fs in facts.items()
             for s, f in fs.items() if commonsense_flags(f.get("text", "") + " " + texts.get(k, ""))]
    res["C1_facts_commonsense"] = ok(not risky, risky[:5])
    templ = [(k, s, templated_hits(f.get("text", ""))) for k, fs in facts.items() for s, f in fs.items()
             if templated_hits(f.get("text", ""))]
    res["R4_natural_phrasing"] = ok(not templ, templ[:5])
    clones, filler = cloned_facts(fam, facts), filler_concepts(fam, facts)
    res["M2_no_cloned_facts"] = ok(not clones and not filler, (clones + filler)[:8])
    from final_construction.wp_v9_vcm import s4_coherence, s4_naming
    inco = s4_coherence.incoherent_facts(facts)                   # P8: dangling pronoun / condition clash
    res["M3_coherent_facts"] = ok(not inco, inco[:8])
    if s4_naming.active(fam):                                     # P6: objects have handles, never proper names
        names = s4_naming.object_naming(fam, facts, texts)
        res["N1_object_naming"] = ok(not names, names[:8])
        rules = s4_naming.handle_rules(fam, facts)
        res["N2_handle_rules"] = ok(not rules, rules[:8])
        from final_construction.wp_v9_vcm import s4_consistency   # M4 voice, M5 ownership/header, M6 action conflict
        res.update(s4_consistency.production_checks(fam, facts, texts))
    from final_construction.wp_v9_vcm import s1_relations
    implaus = s1_relations.implausible(fam)
    res["R3_relation_age_plausible"] = ok(not implaus, implaus)
    return res


CLONE_JACCARD = 0.6
GENERIC_WORDS = frozenset(
    "user users uses use used using owns own owned has have keeps keep kept item items thing things belonging belongs one "
    "ones likes like liked drinks drink drinking eating eat play plays home house everyday daily regular usual usually "
    "often sometimes stuff object stored store stores".split())


def _masked_tokens(text: str, names: set[str]) -> set[str]:
    return {w for w in words(text) if w not in names}


def cloned_facts(fam: dict, facts: dict) -> list[list]:
    """M2 (owner: no copy-paste): fact pairs of one family that are identical or near-identical once concept names are
    masked (token Jaccard >= CLONE_JACCARD). Detail rows: [concept_a, slot_a, concept_b, slot_b, jaccard]."""
    names = {c["name"].casefold() for c in fam["concepts"].values() if c["name"]}
    names |= {w for c in fam["concepts"].values() for w in words(c.get("handle", ""))}
    items = sorted((k, s, _masked_tokens(f.get("text", ""), names)) for k, fs in facts.items() for s, f in fs.items())
    out = []
    for i, (ka, sa, ta) in enumerate(items):
        for kb, sb, tb in items[i + 1:]:
            if ta and tb:
                jac = len(ta & tb) / len(ta | tb)
                if jac >= CLONE_JACCARD:
                    out.append([ka, sa, kb, sb, round(jac, 3)])
    return out


def filler_concepts(fam: dict, facts: dict) -> list[list]:
    """M2: non-target concepts none of whose facts carries an instance-specific content word beyond the concept's
    name, type, relation and generic usage words ("Courtney is a mug the user uses for drinks")."""
    out = []
    for k, c in fam["concepts"].items():
        if c["role"] == "target" or not facts.get(k):
            continue
        own = {c["name"].casefold(), *words(c["fine_type"].replace("_", " ")), *words(c.get("relation", "")),
               *words(c.get("handle", ""))}
        own |= tx.CATEGORY_LEXICON.get(c["fine_type"], frozenset())
        if not any(content(f.get("text", "")) - own - GENERIC_WORDS for f in facts[k].values()):
            out.append([k, "filler"])
    return out


def assertions(text: str) -> str:
    """The asserted part of a reply: question sentences and "whether ..." clauses dropped (they ask, not claim)."""
    sents = re.findall(r"[^.!?]+[.!?]*", text or "")
    return " ".join(re.split(r"\bwhether\b", x, 1, flags=re.I)[0] for x in sents if not x.strip().endswith("?"))


def commonsense_flags(text: str) -> list[str]:
    return [m.group(0) for p in DENYLIST for m in p.finditer(assertions(text))]


def pointer_hits(query: str, lex: frozenset[str] | set[str]) -> list[str]:
    """Possessive/deictic pinpointing of one instance: my/our/.../this/that + (<= 2 words) + a category noun."""
    hits = []
    for m in re.finditer(rf"\b{POINTERS}\s+((?:[a-z'-]+\s+){{0,2}}[a-z'-]+)", query.casefold()):
        if set(m.group(1).split()) & set(lex):
            hits.append(m.group(0))
    return hits


def leak_hits(text: str, banned: set[str]) -> list[str]:
    hits = [m.group(0) for m in _ID_PAT.finditer(text) if m.group(0).casefold() not in ALLOWED_SLASH] + _HEX.findall(text) + _FILE.findall(text)
    hits += _PERSON_ID.findall(text) + _KEY.findall(text)
    low = text.casefold()
    hits += [m for m in tx.ROLE_MARKERS if re.search(r"\b" + re.escape(m) + r"\b", low)]
    hits += [b for b in banned if re.search(r"(?<![\w])" + re.escape(b) + r"(?![\w])", low)]
    return hits


def banned_tokens(fam: dict, pool_ids: dict) -> set[str]:
    """Identity ids, their dataset concept names (>= 6 chars with _ or -) and underscore-style source categories."""
    out = set()
    for c in fam["concepts"].values():
        iid = c["iid"]
        out.add(iid.casefold())
        tail = iid.split(":")[-1]
        if len(tail) >= 6 and re.search(r"[_\-]", tail):
            out.add(tail.casefold())
        cat = str(pool_ids.get(iid, {}).get("category", ""))
        if "_" in cat or ":" in cat:
            out.add(cat.casefold())
    return out


def lookalikes_of(fam: dict, t: str) -> list[str]:
    info = fam["target_info"][t]
    return info["negatives"] + ([info["replacement"]] if info["replacement"] else [])


def fact_leaks(fam: dict, facts: dict, query: str) -> list[str]:
    """F5: a target-group fact keyphrase in the query verbatim, or all of its (>= 2) content words."""
    q, qc = re.sub(r"\s+", " ", query.casefold()), content(query)
    group = [k for k, c in fam["concepts"].items() if c["role"] in ("target", "negative", "replacement")]
    out = []
    for k in group:
        for s in ("s1", "s2"):
            kp = (facts.get(k, {}).get(s) or {}).get("keyphrase", "")
            kc = content(kp)
            if kp.strip() and (re.sub(r"\s+", " ", kp.casefold()) in q or (len(kc) >= 2 and kc <= qc)):
                out.append(kp)
    return out


I2B_MIN_MARGIN, I2B_MIN_OVERLAP = 2, 3


def text_shortcut(fam: dict, facts: dict, query: str, slot: str) -> list[dict]:
    """I2b (pilot fix): a usable text shortcut = the target's used fact overlaps the query by >= I2B_MIN_OVERLAP
    content words AND by >= I2B_MIN_MARGIN more than every look-alike's same-slot fact. Words shared by the query and
    ALL same-slot facts of the group (the action vocabulary I2b itself requires) are excluded first."""
    qc, hits = content(query), []
    for t in fam["targets"]:
        group = [t] + lookalikes_of(fam, t)
        sets = {k: content((facts.get(k, {}).get(slot) or {}).get("text", "")) for k in group}
        common = set.intersection(*sets.values()) if all(sets.values()) else set()
        ov = {k: len((qc & v) - common) for k, v in sets.items()}
        best_look = max((ov[n] for n in group[1:]), default=0)
        if ov[t] >= I2B_MIN_OVERLAP and ov[t] - best_look >= I2B_MIN_MARGIN:
            hits.append({"target": t, "overlap": ov[t], "best_lookalike_overlap": best_look,
                         "words": sorted((qc & sets[t]) - common)})
    return hits


def query_checks(fam: dict, facts: dict, query: str, category_nouns: list[str], row_type: str) -> dict:
    qw = set(words(query))
    names = {c["name"].casefold() for c in fam["concepts"].values() if c["name"]}
    types = {c["fine_type"] for c in fam["concepts"].values() if c["role"] in ("target", "negative", "replacement")}
    lex = tx.lexicon_for(types) | {w for n in category_nouns for w in words(n)}
    brand = {w for t in fam["targets"] for w in words(fam["concepts"][t]["visible_text"]) if len(w) >= 3}
    attrs = qw & (tx.COLOR_WORDS | tx.PATTERN_WORDS)
    pointers, leaks = pointer_hits(query, lex), fact_leaks(fam, facts, query)
    from final_construction.wp_v9_vcm import s4_naming
    handles = s4_naming.handle_in_query(fam, query)
    res = {"I1_I2_query_implicit": ok(not (qw & names) and not pointers and not attrs and not (qw & brand) and not leaks
                                      and not handles,
                                      {"names": sorted(qw & names), "pointers": pointers, "attributes": sorted(attrs),
                                       "brand": sorted(qw & brand), "facts": leaks, "handles": handles})}
    floor = MIN_CLOSED_QUERY_WORDS if row_type in CLOSED_ROWS else MIN_QUERY_WORDS
    res["Q1_action_precise"] = ok(len(words(query)) >= floor, {"words": len(words(query)), "floor": floor})
    if row_type.startswith(("G1_visual", "G1G2_visual")):
        hits = text_shortcut(fam, facts, query, "s1" if row_type.startswith("G1_") else "s2")
        res["I2b_text_shortcut"] = ok(not hits, hits)
    return res


def gold_checks(fam: dict, facts: dict, behavior: str, row_type: str, gold: str, sc: dict, acute: dict,
                used_fact_slot: str) -> dict:
    """Gold-side checks; `behavior` is use | clarify_or_abstain | ignore. Uses only the GPT author's self-checks."""
    res = {"C1_gold_commonsense": ok(not commonsense_flags(gold), commonsense_flags(gold))}
    from final_construction.wp_v9_vcm import s4_naming
    if s4_naming.active(fam):
        res["N1_gold_no_object_names"] = ok(not s4_naming.gold_names(fam, gold), s4_naming.gold_names(fam, gold))
    slot = "s1" if row_type.startswith("G1_") else "s2"
    if behavior == "use":
        # regex OR keyphrase overlap: GPT regexes alone proved brittle on correct golds (canary 2026-10-02)
        tgt = [bool(facts.get(t, {}).get(slot)) and (matches(facts[t][slot]["application_pattern"], gold) or
               overlap_applies(facts[t][slot], gold)) for t in fam["targets"]]
        res["A1_gold_applies_every_target"] = ok(all(tgt) and sc.get("a1_gold_applies_every_target", False), tgt)
        # A1b: the exported support pattern must match the row's own gold EXACTLY as reward_v5 applies it (re.search,
        # no flags): A1 can pass by keyphrase overlap or by re.I while the reward regex fails (canary: 3/18 rows)
        miss = [t for t in fam["targets"] if not support_match((facts.get(t, {}).get(slot) or {}).get(
            "application_pattern", ""), gold)]
        res["A1b_gold_matches_support_pattern"] = ok(not miss, miss)
        swap = [(n, slot) for t in fam["targets"] for n in lookalikes_of(fam, t)
                if (facts.get(n, {}).get(slot) or {}).get("application_pattern")
                and matches(facts[n][slot]["application_pattern"], gold)]
        res["I4_swap_proxy"] = ok(not swap, swap)
        look = {k for k, c in fam["concepts"].items() if c["role"] in ("negative", "replacement")}
        extra = [(k, s) for k, fs in facts.items() for s, f in fs.items()
                 if s != slot or (k not in fam["targets"] and k not in look)
                 if applies(f, gold)]
        res["A2_one_fact_per_target"] = ok(not extra, extra)
    if behavior == "clarify_or_abstain":
        said = assertions(gold)                                # asking about the action is not guessing a fact
        guess = [k for k in fam["concepts"] if slot in facts.get(k, {}) and applies(facts[k][slot], said)]
        res["A4_absent_clarifies"] = ok(sc.get("a4_missing_fact_required", False) and "?" in gold and not guess, guess)
    if behavior == "ignore":
        used = [(k, s) for k, fs in facts.items() for s, f in fs.items() if applies(f, gold)]
        res["A3_closed_no_memory"] = ok(sc.get("closed_impersonal_image_answerable", False) and used_fact_slot == "none"
                                        and not used, used)
    if acute.get("present") and behavior == "clarify_or_abstain":
        # pilot fix: the gold's acute pattern names concepts the GT-absent reply cannot know; the reply must address
        # the hazard first, i.e. its first sentence is a statement/instruction and any question comes after it
        q, stops = gold.find("?"), [i for i in (gold.find("."), gold.find("!")) if i >= 0]
        first = min(stops) if stops else -1
        res["A2_acute_first"] = ok(first >= 0 and (q < 0 or first < q), {"first_stop": first, "first_question": q})
    elif acute.get("present"):
        p = rx(acute.get("gold_pattern", ""))
        m = p.search(gold) if p else None
        starts = []
        for t in fam["targets"]:
            for f in facts.get(t, {}).values():
                pt = rx(f.get("application_pattern", ""))
                hit = pt.search(gold) if pt else None
                if hit:
                    starts.append(hit.start())
        res["A2_acute_first"] = ok(m is not None and (not starts or m.start() <= min(starts)),
                                   {"pattern_found": m is not None, "pattern_at": m.start() if m else None,
                                    "first_fact_at": min(starts) if starts else None})
    return res


def b7_legal(risk: str, act: str, sub: str, condition: str | None) -> dict:
    from final_construction.wp_g_final_benchmark_v1.contract_v1 import B7_RISK, B7_TAG
    legal = (sub == "A7" and act in ("clarify", "abstain") and condition == "gt_absent") or (
        sub in B7_TAG and B7_TAG[sub] == act and B7_RISK[sub] == risk)
    return ok(risk in ("true", "false") and legal, {"risk": risk, "action": act, "subtype": sub})


def row_checks(fam: dict, facts: dict, row: dict, fam_checks: dict, pool_ids: dict) -> dict:
    from final_construction.wp_v9_vcm import s1_plan
    res = dict(fam_checks)
    q, gold = row["query"], row["gold_answer"] or ""
    res.update(query_checks(fam, facts, q, row["category_nouns"], row["row_type"]))
    res.update(gold_checks(fam, facts, row["expected_memory_behavior"], row["row_type"], gold, row["self_check"],
                           row["acute_visual_hazard"], row["used_fact_slot"]))
    entries = row["entries"]
    res["I5_exact_k_entries"] = ok(row["excluded"] == "context_size_mismatch" or (
        len(entries) == row["k"] and len({e["key"] for e in entries}) == row["k"]))
    i6 = []
    for arm, lab in row["arms"].items():
        if lab.get("excluded") == "context_size_mismatch":
            continue
        keys = [e["key"] for e in lab["ranked"]]
        if lab["condition"] != s1_plan.condition(keys, fam["targets"]):
            i6.append((arm, "condition"))
        want = sorted(j + 1 for j, k in enumerate(lab["display_keys"]) if k in fam["targets"]) \
            if lab.get("expected_memory_behavior") == "use" else []
        if lab.get("excluded") is None and lab["gold_citation_set"] != want:
            i6.append((arm, "citation"))
    res["I6_labels_by_rule"] = ok(not i6, i6)
    banned = banned_tokens(fam, pool_ids)
    leaks = leak_hits(q, banned) + [h for e in entries for h in leak_hits(e["text"], banned)]
    leaks += [e["image"] for e in entries if not re.fullmatch(r"entry_[0-9a-f]{20}\.png", e["image"].rsplit("/", 1)[-1])]
    res["I7_no_leak"] = ok(not leaks, leaks[:10])
    res["I8_split"] = ok(row["split"] == "holdout" or row["split"] in ("train", "dev") and "holdout" not in
                         row["image_path"].lower())                  # sealed holdout rows come only from that profile
    res["I9_option_b_b7"] = b7_legal(row["physical_safety_or_health_risk"], row["expected_model_action"],
                                     row["b7_subtype"], row["condition"])
    return res
