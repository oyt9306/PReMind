"""Stage 2b (F1, paid gpt-6-sol, text-only; owner hybrid rule: memory text is quality-critical) + code assembly.

One call per family ("memtext_family") gets the concepts (key, name, relation, kind, type, role, the GPT fact sentences
already stored) and the family's normalised v1.0.1 records, and returns per concept: the open slots filled with ONE
sentence + keyphrase each, either a minimally adapted record (<= 1 record per concept, each record used once) or a new
lasting fact. Code validates every sentence (keyphrase verbatim, record keyphrase kept, no colour/pattern words, no
commonsense denylist hit, no id/role leak, exact slot coverage); on failure ONE retry with the error list, otherwise the
family is invalid. Code (never an LLM) assembles the entry text: "<Name>: the user's <relation | type>." + the fact
sentences in a seeded order; style note = one paragraph, record_list = "- " lines (short_dialogue -> note: dialogue is
optional). The GPT target-group facts stay verbatim, so I3 holds by construction (the check is kept).
Outputs: s2/memtext_v2/<family>.json (write-once; the earlier Qwen s2/memtext/ outputs are never touched).
"""
from __future__ import annotations

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import gpt_client as gc
from final_construction.wp_v9_vcm import llm
from final_construction.wp_v9_vcm import s2_author, s2_local
from final_construction.wp_v9_vcm import s4_checks as ck
from final_construction.wp_v9_vcm import taxonomy as tx
from final_construction.wp_v9_vcm.llm import S, arr, enum, obj

MEMTEXT_DIR = "memtext_v2"
READABLE = {"cup_or_mug": "mug", "glasses_or_sunglasses": "glasses", "teapot_or_kettle": "teapot", "bowl_or_dish": "dish",
            "handbag_or_purse": "handbag", "hat_or_helmet": "helmet", "phone_or_tablet": "phone", "toy_figure": "figure",
            "figurine_or_action_figure": "figurine", "other_object": "belonging", "other_animal": "pet",
            "outdoor_scene": "place", "furniture_other": "piece of furniture", "electronics_other": "device"}
MEMTEXT_PROMPT = """You write the remaining saved facts of ONE user's personal memory (text only, no images).
Each concept is one of the user's people, pets or belongings; `existing_facts` are already stored for it.
1. Records: for each concept choose at most ONE of the user's saved records that naturally and commonsensically belongs\
 to it (the concept whose use, care or situation it is about), or none; each record goes to at most one concept.\
 Adapt a chosen record minimally into ONE third-person sentence using the concept's name (or "the user"), keeping the\
 record's keyphrase word for word; it fills one open slot (source_record_id = its record_id).
2. Every other open slot: ONE new lasting fact sentence about that concept (source_record_id "").
Objects have NO proper name: refer to an object as "it", "this <type>" or by its `handle`, never by an invented name\
 (bad: "Courtney is the mug I use for tea."; good: "This mug was the farewell gift from my first team, so it stays at\
 home."). Persons and pets keep their names.
All sentences: about that instance; never visual (no colours, patterns, brand text, appearance); commonsense-consistent;\
 never relaxing a species- or category-wide rule (e.g. never let a pet eat foods toxic to it); never contradicting the\
 concept's existing facts and never restating or rephrasing them (write about a different aspect); unrelated to any\
 current scene. Every concept gets at least one concrete instance fact (a history, condition, place, habit of THAT\
 item), never filler like "<Name> is a mug the user uses for drinks". No copy-paste: no sentence may repeat another\
 concept's sentence with only the name changed (masked-name token overlap >= 60% fails). keyphrase = 1-8 words copied verbatim from the sentence.
Return exactly the open slots listed for each concept. If `errors` is given, fix every listed error."""
# Production (objects carry handles): the same rules, a third-person example and the voice rule (M4).
MEMTEXT_PROMPT_PROD = MEMTEXT_PROMPT.replace(
    'good: "This mug was the farewell gift from my first team, so it stays at home."',
    'good: "This mug came from the user\'s first team\'s farewell party, so it stays at home."').replace(
    "Return exactly the open slots", "VOICE: every sentence is third person like the entry header - \"the user\","
    " the name, \"it\" or \"this <type>\"; never I, me, my or mine (outside quoted speech).\nBACKGROUND: the new"
    " sentences are background facts (history, a condition, a routine unrelated to moving or handling it): never say"
    " where to put, keep, leave or return it, and never contradict who owns it. They are SCENE-INDEPENDENT: no"
    " current activity, room clearing, visit, outing or time of day - only lasting history, condition or routine."
    "\nReturn exactly the open slots")


def readable(fine_type: str) -> str:
    if fine_type.startswith("singleton_"):
        return "belonging"
    return READABLE.get(fine_type, fine_type.replace("_", " "))


def path(cfg: cm.Cfg, fid: str):  # noqa: ANN201
    """Round 1: s2/memtext_v2/; round r >= 2: s2/memtext_r<r>/ (versioned, round-1 outputs never overwritten)."""
    return cfg.p("s2", MEMTEXT_DIR if cfg.round == 1 else f"memtext_r{cfg.round}", cm.safe_name(fid) + ".json")



def open_slots(fam: dict, key: str, gpt_facts: dict) -> list[str]:
    return [s for s in s2_local.slots_for(fam, key) if s not in gpt_facts.get(key, {})]


def prompt_for(fam: dict) -> str:
    from final_construction.wp_v9_vcm import s4_naming
    return MEMTEXT_PROMPT_PROD if s4_naming.active(fam) else MEMTEXT_PROMPT


def request(fam: dict, gpt_facts: dict, records: list[dict], previous: dict | None = None,
            errors: list | None = None, rid: str | None = None, only: set[str] | None = None) -> dict:
    keys = [k for k in sorted(fam["concepts"]) if open_slots(fam, k, gpt_facts) and (only is None or k in only)]
    rec_ids = [r["record_id"] for r in records]
    concepts = [{"key": k, "name": c["name"], "handle": c.get("handle", ""), "relation": c["relation"], "kind": c["kind"],
                 "type": readable(c["fine_type"]),
                 "role": c["role"], "existing_facts": [f["text"] for f in gpt_facts.get(k, {}).values()],
                 "open_slots": open_slots(fam, k, gpt_facts)} for k, c in sorted(fam["concepts"].items()) if k in keys]
    payload = {"concepts": concepts, "records": [{"record_id": r["record_id"], "text": r["text"],
                                                  "keyphrase": r["keyphrase"]} for r in records]}
    if previous is not None:
        payload.update(previous_output=previous, errors=errors)
    fact = obj({"slot": enum(("o1", "o2", "x")), "sentence": S, "keyphrase": S, "source_record_id": enum(rec_ids + [""])})
    schema = obj({"concepts": arr(obj({"concept_key": enum(keys or ["none"]), "facts": arr(fact)}), len(keys))})
    return {"id": (rid or fam["family_id"]) + ("#retry1" if previous is not None else ""), "instructions": prompt_for(fam),
            "payload": payload, "schema": schema, "name": "s2v9_memtext_family", "images": [], "max_out": 6000}


def validate(fam: dict, gpt_facts: dict, records: list[dict], out: dict | None, only: set[str] | None = None,
             extra: dict | None = None, scene_items: list[dict] | None = None) -> list[str]:
    if not out:
        return ["no valid JSON output"]
    errs, used_records = [], {}
    rec = {r["record_id"]: r for r in records}
    banned = ck.banned_tokens(fam, {})
    got = {c["concept_key"]: c["facts"] for c in out.get("concepts", [])}
    for k in (fam["concepts"] if only is None else sorted(only)):
        want = sorted(open_slots(fam, k, gpt_facts))
        have = sorted(f["slot"] for f in got.get(k, []))
        if have != want:
            errs.append(f"{k}: slots {have} != required {want}")
        n_rec = 0
        for f in got.get(k, []):
            sent, kp = f["sentence"], f["keyphrase"].strip()
            if not kp or len(kp.split()) > 8 or kp.casefold() not in sent.casefold():
                errs.append(f"{k}/{f['slot']}: keyphrase {kp!r} must be 1-8 words copied verbatim from the sentence")
            from final_construction.wp_v9_vcm import s4_coherence, s4_naming
            if s4_naming.active(fam) and s4_coherence.first_person(sent):
                errs.append(f"{k}/{f['slot']}: first person {s4_coherence.first_person(sent)} - write third person")
            if set(ck.words(sent)) & (tx.COLOR_WORDS | tx.PATTERN_WORDS):
                errs.append(f"{k}/{f['slot']}: colour/pattern word in {sent!r}")
            if ck.commonsense_flags(sent):
                errs.append(f"{k}/{f['slot']}: commonsense violation {ck.commonsense_flags(sent)}")
            existing = [x["text"] for x in gpt_facts.get(k, {}).values()]
            kc = ck.content(kp)
            if kc and any(len(kc & ck.content(e)) >= max(2, 0.6 * len(kc)) for e in existing):
                errs.append(f"{k}/{f['slot']}: {kp!r} restates an existing fact of this concept; write about something else")
            from final_construction.wp_v9_vcm import s4_coherence
            for prob in s4_coherence.sentence_problems(sent):
                errs.append(f"{k}/{f['slot']}: {prob}; rewrite the whole sentence as one clear fact about this concept")
            if ck.templated_hits(sent):
                errs.append(f"{k}/{f['slot']}: templated phrasing {ck.templated_hits(sent)}; write a natural saved note")
            if ck.leak_hits(sent, banned):
                errs.append(f"{k}/{f['slot']}: forbidden token {ck.leak_hits(sent, banned)}")
            rid = f["source_record_id"]
            if rid:
                n_rec += 1
                if rid in used_records:
                    errs.append(f"record {rid} used twice ({used_records[rid]}, {k})")
                used_records[rid] = k
                rkp = (rec.get(rid) or {}).get("keyphrase", "").strip()
                if rid not in rec:
                    errs.append(f"{k}: unknown record {rid}")
                elif rkp and rkp.casefold() not in sent.casefold():
                    errs.append(f"{k}/{f['slot']}: adapted record must keep its keyphrase {rkp!r} verbatim")
        if n_rec > 1:
            errs.append(f"{k}: more than one record")
    errs += family_text_errors(fam, gpt_facts, {**(extra or {}), **got})
    from final_construction.wp_v9_vcm import s4_naming, s4_scene
    if scene_items and s4_naming.active(fam):                     # M7: extras never mention the query scene
        errs += s4_scene.memtext_errors(fam, got, scene_items)
    return errs


def family_text_errors(fam: dict, gpt_facts: dict, got: dict) -> list[str]:
    """P5 / M2 on the family's full fact set: a NEW sentence may not clone any other fact (masked names, token Jaccard
    >= 0.6); every non-target concept needs >= 1 instance-specific fact (no filler)."""
    combined = {k: dict(v) for k, v in gpt_facts.items()}
    for k, fs in got.items():
        for f in fs:
            combined.setdefault(k, {})[f["slot"]] = {"text": f["sentence"]}
    new = {(k, f["slot"]) for k, fs in got.items() for f in fs}
    errs = [f"{a}/{sa} and {b}/{sb}: copy-pasted facts (overlap {j}); give each its own reason and wording"
            for a, sa, b, sb, j in ck.cloned_facts(fam, combined) if (a, sa) in new or (b, sb) in new]
    errs += [f"{k}: only generic filler facts; add one concrete instance fact" for k, _ in ck.filler_concepts(fam, combined)]
    from final_construction.wp_v9_vcm import s4_consistency, s4_naming
    if s4_naming.active(fam):                 # production: background facts never instruct, never contradict ownership
        errs += [f"{k}/{f['slot']}: background facts must not say where to put/keep/return it; write history, a"
                 " condition or a routine unrelated to handling" for k, fs in got.items() for f in fs
                 if s4_consistency.destinations(f["sentence"])]
        kinds = {k: c["kind"] for k, c in fam["concepts"].items()}
        errs += [f"{p[0]}: facts disagree on who owns it ({p[1]} vs {p[3]}); keep ownership consistent"
                 for p in s4_consistency.ownership_problems(combined, {}, kinds) if len(p) == 4
                 and ((p[0], p[1]) in new or (p[2], p[3]) in new)]
    return errs


def assemble_text(fam: dict, key: str, sentences: list[tuple[str, str]]) -> str:
    from final_construction.wp_v9_vcm import s4_consistency, s4_naming
    c = fam["concepts"][key]
    if s4_naming.active(fam):                             # production (handles): M5 ownership-aware header
        header = s4_consistency.header(c, readable(c["fine_type"]), [t for _, t in sentences])
    else:
        header = f"{c['name']}: the user's {c['relation'] or readable(c['fine_type'])}."
    order = cm.seeded_order([s for s, _ in sentences], f"order:{fam['family_id']}:{key}")
    body = [dict(sentences)[s] for s in order]
    if c["style"] == "record_list":
        return "\n".join([header] + [f"- {b}" for b in body])
    return " ".join([header] + body)                               # note; short_dialogue -> note (dialogue optional)


def build(fam: dict, gpt_facts: dict, out: dict) -> dict:
    got = {c["concept_key"]: c["facts"] for c in out.get("concepts", [])}
    texts, mapping = {}, {}
    for k in sorted(fam["concepts"]):
        facts = [{"slot": f["slot"], "text": f["sentence"], "keyphrase": f["keyphrase"],
                  "source_record_id": f["source_record_id"]} for f in got.get(k, [])]
        mapping.update({k: f["source_record_id"] for f in facts if f["source_record_id"]})
        sentences = [(s, f["text"]) for s, f in gpt_facts.get(k, {}).items()] + [(f["slot"], f["text"]) for f in facts]
        texts[k] = {"text": assemble_text(fam, k, sentences), "facts": facts, "valid": True}
    return {"texts": texts, "mapping": mapping}


def gpt_facts_of(cfg: cm.Cfg, fid: str) -> dict:
    from final_construction.wp_v9_vcm import s2_revise
    a = s2_author.load_author(cfg, fid)
    return s2_revise.author_facts(a["output"]) if a["valid"] else {}


def reuse_plan(cfg: cm.Cfg, fam: dict) -> tuple[dict | None, set[str] | None]:
    """Lever 2 (production round >= 2): (memtext doc of the revision used before, concepts whose author facts changed
    since) or (None, None) when there is nothing to reuse."""
    from final_construction.wp_v9_vcm import s2_revise
    fid = fam["family_id"]
    if cfg.round < 2 or not cfg.delta:
        return None, None
    from final_construction.wp_v9_vcm import s2_fix                      # repair round: the family's base round
    prev_a = s2_author.load_author(cfg, fid, max_round=s2_fix.base_round(cfg, fid) or cfg.round - 1)
    pp = path(cm.Cfg(**{**cfg.__dict__, "round": max(prev_a.get("revision", 0), 1)}), fid)
    if not prev_a["valid"] or not pp.exists() or not cm.read(pp)["valid"]:
        return None, None
    before, now = s2_revise.author_facts(prev_a["output"]), gpt_facts_of(cfg, fid)
    return {**cm.read(pp), "_path": str(pp)}, {k for k in fam["concepts"] if before.get(k) != now.get(k)}


def _still_failing(cfg: cm.Cfg, fid: str) -> bool:
    """Retry round: a revision that still fails precheck is never assembled, so its memtext call is skipped."""
    from final_construction.wp_v9_vcm import s2_revise
    p = s2_author.author_path(cfg, fid, cfg.round)
    return cfg.retry and p.exists() and bool(s2_revise.blocking(cm.read(p).get("remaining_failures") or []))


def run_memtext_family(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    """Lever 2: in a production round >= 2 only the concepts whose author facts changed get new memory texts; every
    other concept's text is reused byte-identically from the previous revision's memtext (no call if none changed)."""
    plan, pool = cm.plan_raw(cfg), cm.read(cfg.p("s0", "pool.json"))
    todo = [f for f in plan["families"] if not path(cfg, f["family_id"]).exists() and s2_author.in_round(cfg, f["family_id"])
            and s2_author.load_author(cfg, f["family_id"])["valid"] and not _still_failing(cfg, f["family_id"])]
    reuse = {f["family_id"]: reuse_plan(cfg, f) for f in todo}
    status, ctx, only, extra = {}, {}, {}, {}
    for f in list(todo):
        fid = f["family_id"]
        prev, changed = reuse[fid]
        gpt = gpt_facts_of(cfg, fid)
        if prev is not None:
            keep = set(f["concepts"]) - changed
            extra[fid] = {k: [{"slot": x["slot"], "sentence": x["text"], "keyphrase": x["keyphrase"],
                               "source_record_id": x["source_record_id"]} for x in prev["texts"][k]["facts"]]
                          for k in keep if k in prev["texts"]}
            only[fid] = {k for k in changed if open_slots(f, k, gpt)}
        if prev is not None and not only[fid]:                          # nothing to write: no call
            texts = {**build(f, gpt, {"concepts": []})["texts"], **{k: prev["texts"][k] for k in extra[fid]}}
            cm.write_once(path(cfg, fid), {"family_id": fid, "valid": True, "attempts": [], "output": None,
                                           "round": cfg.round, "reused_from": prev["_path"], "texts": texts,
                                           "mapping": {k: v for k, v in prev["mapping"].items() if k in extra[fid]},
                                           "reused_concepts": sorted(extra[fid]), "backend": "reuse"})
            status[fid] = "reused"
            todo.remove(f)
            continue
        used = {v for k, v in (prev or {}).get("mapping", {}).items() if k in extra.get(fid, {})}
        ctx[fid] = (gpt, [s2_local.normalize_record(r) for r in pool["records"].get(fid, [])
                          if r.get("record_id") not in used])
    rid = {f["family_id"]: cm.rid(cfg, f["family_id"]) for f in todo}
    kw = {f["family_id"]: {"only": only.get(f["family_id"])} for f in todo}
    vk = {f["family_id"]: {**kw[f["family_id"]], "extra": extra.get(f["family_id"]), "scene_items": [
        it for it in s2_author.load_author(cfg, f["family_id"])["output"]["items"]]} for f in todo}
    first = llm.run_json(cfg, "memtext_family", [request(f, *ctx[f["family_id"]], rid=rid[f["family_id"]],
                                                         **kw[f["family_id"]]) for f in todo], gpt_fn, generate)
    errs = {f["family_id"]: validate(f, *ctx[f["family_id"]], first.get(rid[f["family_id"]]), **vk[f["family_id"]])
            for f in todo}
    retry_fams = [f for f in todo if errs[f["family_id"]]]
    second = llm.run_json(cfg, "memtext_family", [request(f, *ctx[f["family_id"]], first.get(rid[f["family_id"]]),
                                                          errs[f["family_id"]], rid=rid[f["family_id"]],
                                                          **kw[f["family_id"]]) for f in retry_fams], gpt_fn, generate)
    for f in todo:
        fid = f["family_id"]
        out, attempts = first.get(rid[fid]), [{"errors": errs[fid]}]
        if errs[fid]:
            out = second.get(rid[fid] + "#retry1")
            attempts.append({"errors": validate(f, *ctx[fid], out, **vk[fid])})
        valid = not attempts[-1]["errors"]
        doc = {"family_id": fid, "valid": valid, "attempts": attempts, "output": out, "round": cfg.round,
               "backend": cfg.backends["memtext_family"], "prompt_sha256": cm.h(prompt_for(f))}
        if valid:
            doc.update(build(f, ctx[fid][0], out))
            prev = reuse[fid][0]
            if prev is not None:                                     # unchanged concepts: previous text verbatim
                doc["texts"].update({k: prev["texts"][k] for k in extra[fid]})
                doc["mapping"].update({k: v for k, v in prev["mapping"].items() if k in extra[fid]})
                doc.update(reused_from=prev["_path"], reused_concepts=sorted(extra[fid]))
        cm.write_once(path(cfg, fid), doc)
        status[fid] = "valid" if valid else "invalid_after_retry"
    return {"families": len(status), "retried": len(retry_fams), "status": status,
            "reused_families": sum(v == "reused" for v in status.values())}


def load(cfg: cm.Cfg, fid: str) -> dict:
    return cm.read(path(cfg, fid))

