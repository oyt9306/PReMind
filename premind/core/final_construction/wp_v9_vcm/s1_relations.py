"""Stage 1c (R3, CPU): per-concept, age-plausible person relations without touching plan.json.

The plan gave every person entry of a family one shared relation (R2 B2), so a newborn could become "the user's
friend". Rule over the concept's Qwen description and age group: child descriptors (baby, newborn, infant, toddler,
child, kid, boy, girl; age group child) get a child relation consistent with the family relation (sister/brother ->
niece/nephew, friend -> friend's child, colleague -> colleague's child, neighbor -> neighbor's child; child relations
such as son/daughter stay; otherwise family friend's child); adults keep the family relation. Output:
s1/relations_v2.json (write-once, bound to the plan hash); common.plan_raw applies it for every downstream stage.
"""
from __future__ import annotations

import re

from final_construction.wp_v9_vcm import common as cm

CHILD_WORDS = re.compile(r"\b(?:baby|babies|newborn|infant|toddler|child|children|kid|kids|boy|girl)\b", re.I)
CHILD_RELATIONS = frozenset({"son", "daughter", "nephew", "niece", "grandson", "granddaughter", "friend's child",
                             "colleague's child", "neighbor's child", "family friend's child", "sibling's child"})
ADULT_ONLY = frozenset({"friend", "colleague", "neighbor", "brother", "sister", "father", "mother", "grandfather",
                        "grandmother", "husband", "wife"})


def is_child(concept: dict) -> bool:
    return concept.get("age_group") == "child" or bool(CHILD_WORDS.search(concept.get("description") or ""))


def child_relation(family_relation: str, gender: str) -> str:
    if family_relation in CHILD_RELATIONS:
        return family_relation
    if family_relation in ("sister", "brother"):
        return {"male": "nephew", "female": "niece"}.get(gender, "sibling's child")
    if family_relation in ("friend", "colleague", "neighbor"):
        return f"{family_relation}'s child"
    return "family friend's child"


def implausible(fam: dict) -> list[str]:
    """Person concepts whose description says child but whose relation is adult-only (R3 check)."""
    return [k for k, c in fam["concepts"].items()
            if c["kind"] == "person" and is_child(c) and c["relation"] in ADULT_ONLY]


def run_relations(cfg: cm.Cfg) -> dict:
    out = cfg.p("s1", "relations_v2.json")
    if out.exists():
        return {"status": "exists"}
    plan = cm.read(cfg.p("s1", "plan.json"))
    fams, changed = {}, 0
    for fam in plan["families"]:
        rel = {}
        for k, c in sorted(fam["concepts"].items()):
            if c["kind"] != "person":
                continue
            new = child_relation(c["relation"], c.get("gender", "")) if is_child(c) else c["relation"]
            rel[k] = {"relation": new, "family_relation": c["relation"], "child": is_child(c)}
            changed += new != c["relation"]
        fams[fam["family_id"]] = rel
    cm.write_once(out, {"plan_sha256": cm.sha_file(cfg.p("s1", "plan.json")), "families": fams,
                        "rule": "child descriptor or age group child -> child relation consistent with the family relation"})
    return {"families": len(fams), "persons": sum(len(v) for v in fams.values()), "changed": changed}
