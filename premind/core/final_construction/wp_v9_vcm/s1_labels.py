"""Stage 1c' (P6, CPU, production): concept labels v2 - how the user refers to each concept (plan.json untouched).

Persons keep their name + an age-plausible relation (R3 rule, s1_relations); pets keep their name; OBJECTS get no
proper name ("people rarely name a bag, a mug or a power strip"): a short natural handle the way people refer to
their things - origin, ownership, gift or occasion ("the one from the Kyoto trip", "Jisu's old one", "the work one").
Handles are drawn (seeded) from HANDLES, unique within the family (owner-less look-alikes included), <= 6 words after
the type noun, and carry no colour, pattern, brand, material, visible attribute or room/location word. The memory
header becomes "The user's <type> - <handle>."; facts and golds refer to the object by its handle, "it" or
"this <type>". Output: s1/concept_labels_v2.json (write-once, bound to the plan hash); common.plan_raw applies it.
"""
from __future__ import annotations

import re

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s1_relations
from final_construction.wp_v9_vcm import taxonomy as tx

# Production v3 (2026-10-03, pilot spot-check): neutral occasion / origin handles only - no ownership-transfer verb
# ("gave me" next to an "on loan" fact contradicts it), no first person (entries are third person, M4), no possessive.
HANDLES = ("the one from the team party", "the one from the Kyoto trip", "the work one", "the flea-market one",
           "the birthday one", "the one from the fair", "the one from the family visit", "the airport-shop one",
           "the spare one", "the one from the hardware sale", "the one from the school fete", "the housewarming one",
           "the one from the camping trip", "the one from Lisbon", "the charity-shop one", "the anniversary one",
           "the one from the library sale", "the one from the first job", "the one from the neighbour",
           "the one from the wedding", "the holiday one", "the one from the conference", "the Seoul one",
           "the Madrid one", "the one from the move", "the one from the night market", "the graduation one",
           "the one from the street festival")
# the pilot's v2 handles (concept_labels_v2.json; kept for the record - several break the v3 rules)
HANDLES_V2 = ("the one from the team party", "the one from the Kyoto trip", "the work one", "the flea-market find",
              "the birthday one", "the one I won at the fair", "the hand-me-down from my aunt", "the airport-shop one",
              "the spare one", "the one from the hardware sale", "the one from the school fete",
              "the housewarming gift", "the one from the camping trip", "the one from Lisbon", "the charity-shop find",
              "the anniversary gift", "the one from the library sale", "the one from my first job",
              "the one my neighbour gave me", "the one from the wedding", "the holiday one", "the one from the conference",
              "Jisu's old one", "Mateo's spare one", "the one from the move", "the one from the night market",
              "the graduation gift", "the one from the swap meet")
TRANSFER_WORDS = frozenset("gave give given gives gift gifts gifted bought buy buys purchased purchase lent lend lends loan "
                           "loaned borrowed borrow won win inherited hand sold sell swap swapped traded trade owned owner".split())
FIRST_PERSON = frozenset("i me my mine myself".split())
MATERIAL_WORDS = frozenset("wooden wood metal metallic glass plastic ceramic leather cotton wool woolen steel silver gold "
                           "golden paper stone porcelain rubber canvas velvet linen silk bamboo wicker denim".split())
LOCATION_WORDS = frozenset("kitchen bedroom bathroom hall hallway garden balcony sofa couch desk table shelf counter sink "
                           "garage porch window door floor bed closet cupboard drawer office living room yard".split())
MAX_HANDLE_WORDS = 6


def handle_problems(handle: str) -> list[str]:
    w = set(re.findall(r"[a-z]+", handle.casefold()))
    bad = []
    if len(handle.split()) > MAX_HANDLE_WORDS + 1:                     # "+1": the leading article
        bad.append("too_long")
    if re.search(r"\w(?:'|\u2019)s\b", handle):
        bad.append("possessive")
    for name, vocab in (("colour", tx.COLOR_WORDS), ("pattern", tx.PATTERN_WORDS), ("material", MATERIAL_WORDS),
                        ("location", LOCATION_WORDS), ("transfer", TRANSFER_WORDS), ("first_person", FIRST_PERSON)):
        if w & vocab:
            bad.append(f"{name}:{sorted(w & vocab)}")
    return bad


assert not any(handle_problems(x) for x in HANDLES), "a curated handle breaks the handle rules"


def labels_for(fam: dict, curated: tuple[str, ...] = HANDLES) -> dict[str, dict]:
    objects = sorted(k for k, c in fam["concepts"].items() if c["kind"] == "object")
    handles = cm.seeded_order(list(curated), f"handles:{fam['family_id']}")
    if len(objects) > len(handles):
        raise RuntimeError(f"{fam['family_id']}: more objects than curated handles")
    out = {}
    for k, c in sorted(fam["concepts"].items()):
        if c["kind"] == "person":
            rel = s1_relations.child_relation(c["relation"], c.get("gender", "")) if s1_relations.is_child(c) \
                else c["relation"]
            out[k] = {"name": c["name"], "relation": rel, "handle": ""}
        elif c["kind"] == "pet":
            out[k] = {"name": c["name"], "relation": "", "handle": ""}
        else:
            out[k] = {"name": "", "relation": "", "handle": handles[objects.index(k)]}
    return out


def run_labels(cfg: cm.Cfg, version: int = 2, first: int | None = None) -> dict:
    """v2 (pilot, HANDLES_V2): every family. v3 (production, HANDLES): an overlay for the families from `first`
    (default: after the pilot); common.plan_raw applies v2 then v3."""
    out = cfg.p("s1", f"concept_labels_v{version}.json")
    if out.exists():
        return {"status": "exists"}
    plan = cm.read(cfg.p("s1", "plan.json"))
    todo = plan["families"] if version == 2 else plan["families"][cm.PROD_PILOT_N if first is None else first:]
    fams = {f["family_id"]: labels_for(f, HANDLES_V2 if version == 2 else HANDLES) for f in todo}
    cm.write_once(out, {"plan_sha256": cm.sha_file(cfg.p("s1", "plan.json")), "families": fams,
                        "policy": "persons: name + age-plausible relation; pets: name; objects: no name, unique handle"})
    return {"families": len(fams), "objects": sum(1 for f in fams.values() for v in f.values() if v["handle"])}


def display(c: dict) -> str:
    """How a concept is shown in prompts/headers: name for persons/pets, "the user's <type> - <handle>" for objects."""
    if c.get("handle"):
        from final_construction.wp_v9_vcm import s2_memtext
        return f"the user's {s2_memtext.readable(c['fine_type'])} - {c['handle']}"
    return c["name"]
