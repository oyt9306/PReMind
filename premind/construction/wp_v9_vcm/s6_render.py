"""Stage 6 (CPU): render rows for training/eval and emit the reward_v5 row schema.

System prompt = train_interface_v1 final rules (SYSTEM_CORE_FINAL; <decision> ... <context_use> ... </decision>
<answer>) + a VCM memory note (cite memories as [j]). User content follows PMLLM/gpt_cap_gen.build_multimodal_content
(blocks "===== Memory j =====" + image + text, j = 1..K in the stored display order, then "===== New Image =====" +
query image + query), with K = 8 for every image (9 images per prompt).
  passive arm      one user turn: K memory blocks, then New Image + query (owner change 4: exact gpt_cap_gen order);
  memory-call arm  Stage 1 = New Image + query + the Stage-1 instruction (<memory_call>call_pdb|direct_answer);
                   Stage 2 (call_pdb) = the K memory blocks + final instruction (the query image is not re-sent);
                   Stage 2 (direct_answer) = the fixed no-memory notice + final instruction.
Both arms and both retriever arms render from the same stored retrieval of the row.
"""
from __future__ import annotations

from premind.construction.wp_v9_vcm import common as cm
from premind.construction.wp_v9_vcm import s4_checks

VCM_NOTE = ("\n\nPersonal memory entries: an entry \"===== Memory j =====\" pairs an image of one of the user's people,"
            " pets or belongings with the user's saved text about it. Compare the images to decide whether a"
            " remembered individual is the same one as in the new image, and apply a memory only to the individual"
            " it shows. In <context_use>, cite the memory numbers you used as [j] (several as [i, j]), or none.")
VCM_RENDER_VERSION = "s2v9_vcm_render_v1"            # canary renders: unchanged
PROD_RENDER_VERSION = "s2v9_vcm_render_v2"           # P7/P7b: withhold-aligned, person/pet/belonging-neutral wording
NEW_MISSING_FACT_RULE = (
    "If the correct answer depends on a personal fact you do not have \u2014 for example, none of the memories shows"
    " the individual in the new image \u2014 do not guess and do not borrow a fact from a similar-looking memory. Either"
    " ask a specific clarifying question or say that you cannot tell from the user's memories (if a visible acute"
    " hazard exists, address it first). If the answer does not depend on such a fact, answer directly.")
# (old, new) exact substitutions; each old string must occur exactly once (everything else stays byte-identical)
PROD_CORE_EDITS = (
    ("A constraint or condition of another household member applies only if that person is involved in this scene or"
     " request; memories about objects or situations not in this scene are irrelevant.",
     "A fact about one of the user's people, pets or belongings applies only to that individual, and only if it is in"
     " this scene or request; memories about individuals or situations not in this scene are irrelevant."),
    ("<context_use>Which supplied context was relevant, or none.</context_use>",
     "<context_use>Memory numbers you used, as [j] or [i, j], or none.</context_use>"))
PROD_MC_EDITS = (
    ("(household members, health facts, preferences, belongings, routines)",
     "(the user's people, pets and belongings, health facts, preferences, routines)"),
    ("a personal fact about this user or a household member that is not visible in the image or stated in the request.",
     "a personal fact about the user or one of the user's people, pets or belongings that is not visible in the image"
     " or stated in the request."))
PROD_S1_EDITS = (("decide whether you need this person's personal memory", "decide whether you need the user's personal memory"),)
PROD_NOTE_EDITS = (("apply a memory only to the individual it shows.",
                    "apply a memory only to the individual it shows. If no memory shows the same individual as the new"
                    " image, treat the remembered facts as not applying to it."),)


def _edit(text: str, edits: tuple) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"production prompt edit anchor not found exactly once: {old[:50]!r}")
        text = text.replace(old, new)
    return text


def _final(profile: str = "canary") -> tuple[str, str, str, str, str, str]:
    from premind.prompts import render as tr
    from premind.prompts.final_prompts import (MISSING_FACT_RULE, STAGE1_SYSTEM_SECTION,
                                                                     SYSTEM_CORE_FINAL)
    core, mc, s1 = SYSTEM_CORE_FINAL, tr.MEMORY_CALL_SECTION, tr.STAGE1_INSTRUCTION
    if profile == "prod":
        core = _edit(core, ((MISSING_FACT_RULE, NEW_MISSING_FACT_RULE),) + PROD_CORE_EDITS)
        mc, s1 = _edit(mc, PROD_MC_EDITS), _edit(s1, PROD_S1_EDITS)
    return core, STAGE1_SYSTEM_SECTION, mc, s1, tr.NO_MEMORY_NOTICE, tr.FINAL_INSTRUCTION


def systems(profile: str = "canary") -> dict[str, str]:
    core, s1sys, mc, _, _, _ = _final(profile)
    note = _edit(VCM_NOTE, PROD_NOTE_EDITS) if profile == "prod" else VCM_NOTE
    return {"passive": core + note, "memory_call": core + note + mc + s1sys}


def render_version(profile: str) -> str:
    return PROD_RENDER_VERSION if profile == "prod" else VCM_RENDER_VERSION


def memory_blocks(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        out += [{"type": "text", "text": f"===== Memory {e['display_index']} =====\n"},
                {"type": "image", "image": e["image"]}, {"type": "text", "text": e["text"]}]
    return out


def new_image_block(row: dict, extra: str = "") -> list[dict]:
    return [{"type": "text", "text": "===== New Image =====\n"}, {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": f"User query: {row['query']}" + extra}]


def arm_entries(row: dict, arm: str, texts: dict[str, str], images: dict[str, str]) -> list[dict]:
    keys = row["arms"][arm].get("display_keys") or []
    return [{"display_index": j + 1, "key": k, "image": images[k], "text": texts[k]} for j, k in enumerate(keys)]


def render_arm(row: dict, entries: list[dict], profile: str = "canary") -> dict:
    _, _, _, s1_instr, no_mem, final = _final(profile)
    sy = systems(profile)
    return {"passive": [{"role": "system", "content": [{"type": "text", "text": sy["passive"]}]},
                        {"role": "user", "content": memory_blocks(entries) + new_image_block(row)}],
            "memory_call_stage1": [{"role": "system", "content": [{"type": "text", "text": sy["memory_call"]}]},
                                   {"role": "user", "content": new_image_block(row, "\n\n" + s1_instr)}],
            "memory_call_stage2_call": memory_blocks(entries) + [{"type": "text", "text": final}],
            "memory_call_stage2_no_call": [{"type": "text", "text": f"{no_mem}\n\n{final}"}],
            "n_images": len(entries) + 1}


def static_texts(profile: str = "canary") -> list[str]:
    _, _, _, s1_instr, no_mem, final = _final(profile)
    return [*systems(profile).values(), s1_instr, no_mem, final, "===== Memory 1 =====", "===== New Image ====="]


def static_leaks(profile: str = "canary") -> list[str]:
    """Static templates legitimately contain tag names (direct_answer, <memory_call>); only role/label markers and
    ids are checked there. Dynamic text (query, memory texts) gets the full I7 leak check."""
    low = " ".join(static_texts(profile)).casefold()
    import re
    from premind.construction.wp_v9_vcm import taxonomy as tx
    return [m for m in tx.ROLE_MARKERS if re.search(r"\b" + re.escape(m) + r"\b", low)] + \
        [h for t in static_texts(profile) for h in s4_checks._PERSON_ID.findall(t) + s4_checks._KEY.findall(t)]


def reward_fields(row: dict, arm: str) -> dict:
    lab = row["arms"][arm]
    keys = lab.get("display_keys") or []
    by_target = {}
    if lab.get("expected_memory_behavior") == "use":
        for t, pat in zip(row["target_keys"], lab["support_patterns"]):
            by_target[str(keys.index(t) + 1)] = pat
    return {"excluded": lab.get("excluded"), "condition": lab.get("condition"), "absent_reason": lab.get("absent_reason"),
            "gold_answer": lab.get("gold_answer"), "expected_model_action": lab.get("expected_model_action"),
            "b7_subtype": lab.get("b7_subtype"), "expected_memory_behavior": lab.get("expected_memory_behavior"),
            "gold_action_set": lab.get("gold_action_set") or [lab.get("expected_model_action")],
            "gold_answer_reference_only": bool(lab.get("gold_answer_reference_only")),
            "gold_citation_set": lab.get("gold_citation_set"), "support_patterns_by_display_index": by_target,
            "decisive_keyphrases": lab.get("decisive_keyphrases"),
            "wrong_concept_patterns": lab.get("wrong_concept_patterns"),
            "display_to_rank": lab.get("display_to_rank"), "display_keys": keys,
            "retrieval": {k: lab.get(k) for k in ("retriever", "retriever_id", "revision", "k", "ranked")}}


def source_round(row: dict, family_rows: list[dict]) -> int:
    """The round whose author output + memory texts BUILT the row. Round-7 additive rows were assembled from the
    family's base inputs (the max round of its non-additive selected rows, s2_extra._base_round), not from the
    latest author revision <= 7 (a still-failing rev3/rev4 has no memtext and must never be used)."""
    if row.get("base_round"):
        return int(row["base_round"])
    if row.get("extra_kind"):
        return max([int(x.get("round", 1)) for x in family_rows if not x.get("extra_kind")] or [1])
    return int(row.get("round", 1))

