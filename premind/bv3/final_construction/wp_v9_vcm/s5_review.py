"""Stage 5: v8a-pattern review gate on deterministic-pass rows.

Primary (cfg.backends["review_primary"], default GPT; per family batch, all memory images + the new image) and
second (cfg.backends["review_second"], default local Qwen3-VL-8B; per row, interleaved gpt_cap_gen layout) answer problem questions (yes = problem, the v7b framing used by v8a): the owner core
(shortcut_free, memory_grounded, memory_necessary) + the v9 items (target concept visible and is what the gold is
about; a look-alike's fact would change the answer; no plausible gold about another visible object; GT-absent needs
the missing fact). An item where only Qwen reports a problem is re-judged once by GPT at medium effort.
accepted = deterministic checks pass AND GPT accepts AND (Qwen accepts OR adjudicator accepts).
Gate: acceptance >= 0.80 for personal and for control rows (v8a minima, unchanged).
"""
from __future__ import annotations

from collections import Counter

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import gpt_client as gc
from final_construction.wp_v9_vcm import llm

PROBLEMS = {"answer_given_away": "shortcut_free", "answer_unsupported": "memory_grounded",
            "memory_role_wrong": "memory_necessary", "target_not_visible_or_not_subject": "target_visible_and_subject",
            "lookalike_fact_same_answer": "lookalike_fact_changes_answer",
            "other_object_plausible_gold": "no_other_object_gold", "absent_fact_not_required": "absent_needs_fact",
            "action_ambiguous": "action_unambiguous", "commonsense_violation": "commonsense_consistent",
            "contrast_fact_category_level": "contrast_instance_specific",
            "memory_text_incoherent": "memory_texts_coherent"}
REVIEW_PROMPT = (
    "You review benchmark items for a personalized visual assistant. The answerer sees the user's retrieved memories"
    " (each = an image of one of the user's concepts + the user's saved text about it), the NEW IMAGE and the query."
    " For EACH item answer these questions with yes or no, where YES MEANS THERE IS A PROBLEM. Judge each item alone.\n"
    "1. answer_given_away: could the gold answer be written from the query text alone, without the images and memories?\n"
    "2. answer_unsupported: personal_gt_present -> is the gold NOT supported by the cited memories' facts, or applied"
    " wrongly? personal_gt_absent -> does the gold guess the missing personal fact instead of asking or abstaining (after an acute"
    " visible hazard, if any)? no_memory_control -> is the gold wrong without using any memory?\n"
    "3. memory_role_wrong: personal_gt_present -> could the item be answered correctly WITHOUT the cited memories?"
    " personal_gt_absent -> is the missing personal fact actually NOT needed? no_memory_control -> would some memory"
    " change the correct answer?\n"
    "4. target_not_visible_or_not_subject: personal rows -> is the concept of the cited memory (gt_present) or the"
    " concept the query needs (gt_absent) NOT visible in the new image, or is the gold not about it? controls -> no.\n"
    "5. lookalike_fact_same_answer: personal rows -> would applying the fact of a look-alike memory (same kind of"
    " concept, not in the new image) give the SAME answer as the gold? controls -> no.\n"
    "6. other_object_plausible_gold: is there a different plausible gold answer about another visible object?\n"
    "7. absent_fact_not_required: personal_gt_absent -> could a generic answer be correct without the missing"
    " instance fact? other rows -> no.\n"
    "8. action_ambiguous: could a reader interpret the query as asking about a different action or situation"
    " (appliance, place, time or goal unclear)?\n"
    "9. commonsense_violation: does any memory text or the gold contradict common sense or recommend an action that"
    " general knowledge marks as unsafe or damaging for any visible item (e.g. a fragile item in a dishwasher, a food"
    " toxic to that animal)?\n"
    "10. contrast_fact_category_level: do the look-alike memories differ by a fact that contradicts or relaxes a"
    " species- or category-wide rule instead of an instance-specific property (ownership, history, an individual"
    " condition, a commonsense-consistent care preference)?\n"
    "11. memory_text_incoherent: is ANY of the shown memory texts (all memories of the item) self-contradictory,"
    " ungrammatical, missing a referent (e.g. \"sort them\" with nothing to sort), or a fact that makes no sense for"
    " that individual (e.g. \"can kneel without restriction because of a knee condition\")?\n"
    "For personal_gt_absent items answer items 1 and 3 with no (a clarifying question can always be written from the"
    " query, and item 7 is the necessity check for these items). Their label is WITHHOLD: a clarifying question and a"
    " clean abstention (\"I can't tell which one this is from your memories, so I won't guess\") are both correct (an"
    " acute visible hazard is still addressed first); the gold answer is one reference response, so judge whether it"
    " withholds without guessing or applying a look-alike's fact, not its exact wording.\n"
    "Answer yes only for a clear problem. Give a one-sentence rationale.")
ADJ_TEXT = ("\nTEXT-ONLY repair adjudication: no image is attached; the memory and image labels refer to images you"
            " cannot see. Judge from the texts, the hazard evidence and both reviews' rationales; a visual objection of"
            " the image reviewer stands unless the texts clearly refute it.")
ADJ_PROMPT = REVIEW_PROMPT + ("\nTwo reviewers disagreed on these items: `reviews` gives each reviewer's answers and"
                              " rationale. Decide independently and carefully; answer yes only for a clear problem.")


def row_kind(r: dict) -> str:
    if r["expected_memory_behavior"] == "ignore":
        return "no_memory_control"
    return "personal_gt_present" if r["condition"] == "gt_present" else "personal_gt_absent"


def schema(rows: list[dict]) -> dict:
    yn = {"type": "string", "enum": ["yes", "no"]}
    item = {"type": "object", "additionalProperties": False, "required": ["item_id", *PROBLEMS, "rationale"],
            "properties": {"item_id": {"type": "string", "enum": [r["row_id"] for r in rows]},
                           **{p: yn for p in PROBLEMS}, "rationale": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "minItems": len(rows), "maxItems": len(rows), "items": item}}}


def parse(out: object, rows: list[dict]) -> dict[str, dict]:
    got = {it.get("item_id"): it for it in (out.get("items", []) if isinstance(out, dict) else []) if isinstance(it, dict)}
    res = {}
    for r in rows:
        it = got.get(r["row_id"])
        valid = isinstance(it, dict) and all(it.get(p) in ("yes", "no") for p in PROBLEMS)
        res[r["row_id"]] = {"valid": valid, "problems": {p: it[p] for p in PROBLEMS} if valid else None,
                            "rationale": it.get("rationale") if valid else None}
    return res


ABSENT_IGNORED = ("answer_given_away", "memory_role_wrong")    # R2: not applicable to personal_gt_absent rows


def accepts(v: dict | None, row: dict | None = None) -> bool:
    """All problems answered "no"; for personal_gt_absent rows items 1 and 3 are ignored (R2, enforced in code)."""
    if not (v and v.get("valid")):
        return False
    skip = ABSENT_IGNORED if row is not None and row_kind(row) == "personal_gt_absent" else ()
    return all(x == "no" for p, x in v["problems"].items() if p not in skip)


def item_payload(r: dict, label_of: dict[str, str]) -> dict:
    return {"item_id": r["row_id"], "row_type": row_kind(r), "query": r["query"], "gold_answer": r["gold_answer"],
            "risk": r["risk_label"], "action": r["expected_model_action"],
            "memories": [{"memory": e["display_index"], "image": label_of[e["key"]], "text": e["text"]}
                         for e in r["entries"]], "gold_cited_memories": r["gold_citation_set"]}


def _rows(cfg: cm.Cfg) -> list[dict]:
    rows = [r for r in cm.read(cm.rows_path(cfg))["rows"] if r["deterministic_pass"]]
    if cm.is_fix(cfg):                                  # repair round 3: only the affected (changed) rows are reviewed
        from final_construction.wp_v9_vcm import s2_fix
        keep = s2_fix.affected_row_ids(cfg) or set()
        rows = [r for r in rows if r["row_id"] in keep]
    return rows


def _by_family(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: cm.h(f"{cm.SEED}:review_order:{r['row_id']}")):
        out.setdefault(r["family_id"], []).append(r)
    return out


def _images(rows: list[dict]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    keys = sorted({e["key"] for r in rows for e in r["entries"]}, key=lambda k: cm.h(f"img:{k}:{rows[0]['family_id']}"))
    label_of = {k: f"IMG-{n + 1}" for n, k in enumerate(keys)}
    img_of = {e["key"]: e["image"] for r in rows for e in r["entries"]}
    return [("NEW IMAGE", rows[0]["image_path"])] + [(label_of[k], img_of[k]) for k in keys], label_of


QIMG_TEXTS_NOTE = ("\nMEMORY IMAGES ARE NOT ATTACHED: you see the NEW IMAGE and each memory's saved TEXT only (its"
                   " image label is not shown); judge visibility and identity from the new image and the texts.")


def review_mode(cfg: cm.Cfg) -> str:
    """GPT review input: holdout = the query image + memory texts only (owner 2026-10-03); repair round 3 = text
    only; otherwise every image."""
    if cm.is_extra(cfg) or cm.is_rereview(cfg):         # rounds 7/9: text-only GPT, Qwen with images
        return "text_extra"
    return "text_repair" if cm.is_fix(cfg) else "gpt_query_img_texts" if cfg.profile == "holdout" else "gpt_images"


def _gpt_view(cfg: cm.Cfg, images: list, label_of: dict) -> tuple[list, dict, str]:
    """(images, labels, prompt suffix) of a GPT review call under review_mode(cfg)."""
    mode = review_mode(cfg)
    if mode in ("text_repair", "text_extra"):
        return [], label_of, ADJ_TEXT
    if mode == "gpt_query_img_texts":
        return images[:1], {k: "not shown" for k in label_of}, QIMG_TEXTS_NOTE
    return images, label_of, ""


def _write(cfg: cm.Cfg, kind: str, verdicts: dict[str, dict], meta: dict) -> None:
    for rid, v in verdicts.items():
        cm.write_once(cfg.p(cm.s5_dir(cfg), kind, cm.safe_name(rid) + ".json"), {"row_id": rid, **v, **meta})


def _verdict(cfg: cm.Cfg, kind: str, rid: str) -> dict | None:
    p = cfg.p(cm.s5_dir(cfg), kind, cm.safe_name(rid) + ".json")
    return cm.read(p) if p.exists() else None


def review_content(r: dict) -> str:
    """Hash of everything a reviewer sees for a row (item payload fields, memory images, new image)."""
    import json
    return cm.h(json.dumps({"kind": row_kind(r), "q": r["query"], "gold": r["gold_answer"], "risk": r["risk_label"],
                            "action": r["expected_model_action"], "cite": r["gold_citation_set"], "img": r["image_path"],
                            "entries": [[e["display_index"], e["key"], e["image"], e["text"]] for e in r["entries"]]},
                           sort_keys=True))


def reuse_verdicts(cfg: cm.Cfg, kind: str, rows: list[dict]) -> int:
    """Round N >= 2 (pilot fix): a row of a family NOT re-authored since round P whose review content equals its row
    in round P (the latest earlier round with a `kind` verdict) reuses that VALID verdict (copied write-once with
    `reused_from`; never re-paid). Revised families are reviewed afresh, except in the production retry round
    (cfg.retry): there an unchanged row of a revised family also reuses an ACCEPTING verdict."""
    from final_construction.wp_v9_vcm import s2_author
    if cfg.round < 2:
        return 0
    cache: dict[int, dict] = {}
    n = 0
    for r in rows:
        if _verdict(cfg, kind, r["row_id"]):
            continue
        for rnd in range(cfg.round - 1, 0, -1):
            src = cfg.p("s5" if rnd == 1 else f"s5_r{rnd}", kind, cm.safe_name(r["row_id"]) + ".json")
            if not src.exists():
                continue
            if rnd not in cache:
                rp = cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}{'' if rnd == 1 else f'_r{rnd}'}.json")
                cache[rnd] = {x["row_id"]: x for x in cm.read(rp)["rows"]} if rp.exists() else {}
            old = cache[rnd].get(r["row_id"])
            revised = any(s2_author.author_path(cfg, r["family_id"], m).exists() for m in range(rnd + 1, cfg.round + 1))
            ok = not revised or (cfg.retry and accepts(cm.read(src), old))
            if old and ok and cm.read(src).get("valid") and review_content(old) == review_content(r):
                cm.write_once(cfg.p(cm.s5_dir(cfg), kind, cm.safe_name(r["row_id"]) + ".json"),
                              {**cm.read(src), "reused_from": f"{src.parent.parent.name}/{kind}/{src.name}"})
                n += 1
            break
    return n


def run_review_primary(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    if cm.is_fix(cfg):                                  # repair round 3: text-only GPT check (no images)
        from final_construction.wp_v9_vcm import s2_fix
        return s2_fix.run_text_review(cfg, _rows(cfg), gpt_fn, generate)
    if cm.is_extra(cfg) or cm.is_rereview(cfg):         # rounds 7/9: text-only GPT check of the additive rows
        from final_construction.wp_v9_vcm import s2_fix_text, s5_extra
        return s2_fix_text.run_text_review(cfg, _rows(cfg), gpt_fn, generate, s5_extra.REVIEW_PROMPT,
                                           s5_extra.REVIEW_PROBLEMS, "s2v9_text_extra_review_v1", "text_extra",
                                           s5_extra.review_item)
    reused = reuse_verdicts(cfg, "primary", _rows(cfg))
    fams = {fid: todo for fid, rows in _by_family(_rows(cfg)).items()
            if (todo := [r for r in rows if not _verdict(cfg, "primary", r["row_id"])])}
    reqs = []
    for fid, rows in fams.items():
        images, label_of, note = _gpt_view(cfg, *_images(rows))
        reqs.append({"id": cm.rid(cfg, fid), "instructions": REVIEW_PROMPT + note, "payload": {"items": [item_payload(r, label_of) for r in rows]},
                     "schema": schema(rows), "name": "s2v9_review_v1", "images": images,
                     "max_out": 700 * len(rows) + 800})
    res = llm.run_json(cfg, "review_primary", reqs, gpt_fn, generate)
    for fid, rows in fams.items():
        _write(cfg, "primary", parse(res.get(cm.rid(cfg, fid)), rows), {"backend": cfg.backends["review_primary"],
                                                                        "review_mode": review_mode(cfg)})
    return {"families": len(fams), "rows": sum(map(len, fams.values())), "reused": reused,
            "invalid": sum(v is None for v in res.values())}


def second_request(r: dict, suffix: str = "") -> dict:
    parts: list[tuple[str, str]] = []
    for e in r["entries"]:
        parts += [("text", f"===== Memory {e['display_index']} =====\n"), ("image", e["image"]), ("text", e["text"])]
    parts += [("text", "===== New Image =====\n"), ("image", r["image_path"])]
    label_of = {e["key"]: f"Memory {e['display_index']} image" for e in r["entries"]}
    return {"id": r["row_id"] + suffix, "instructions": REVIEW_PROMPT, "payload": {"items": [item_payload(r, label_of)]},
            "schema": schema([r]), "name": "s2v9_review_second", "interleave": parts,
            "images": [("NEW IMAGE", r["image_path"])] + [(label_of[e["key"]], e["image"]) for e in r["entries"]],
            "max_out": 1200}


def run_review_second(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    reused = reuse_verdicts(cfg, "second", _rows(cfg))
    rows = [r for r in _rows(cfg) if not _verdict(cfg, "second", r["row_id"])]
    suffix = "" if cfg.round == 1 else f"@r{cfg.round}"
    res = llm.run_json(cfg, "review_second", [second_request(r, suffix) for r in rows], gpt_fn, generate)
    for r in rows:
        _write(cfg, "second", parse(res.get(r["row_id"] + suffix), [r]), {"backend": cfg.backends["review_second"]})
    return {"reviewed": len(rows), "reused": reused, "invalid": sum(v is None for v in res.values())}


def run_adjudicate(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    reqs, todo_by = [], {}
    reuse_verdicts(cfg, "adj", _rows(cfg))
    for fid, rows in _by_family(_rows(cfg)).items():
        todo = [r for r in rows if accepts(_verdict(cfg, "primary", r["row_id"]), r)
                and not accepts(_verdict(cfg, "second", r["row_id"]), r) and not _verdict(cfg, "adj", r["row_id"])]
        if not todo:
            continue
        images, label_of, note = _gpt_view(cfg, *_images(rows))   # repair: text only; holdout: query image + texts
        items = [{**item_payload(r, label_of), "reviews": {k: {f: (_verdict(cfg, k, r["row_id"]) or {}).get(f)
                                                               for f in ("problems", "rationale")}
                                                           for k in ("primary", "second")}} for r in todo]
        reqs.append({"id": cm.rid(cfg, fid), "instructions": ADJ_PROMPT + note,
                     "payload": {"items": items}, "schema": schema(todo),
                     "name": "s2v9_adjudication_v1", "images": images, "max_out": 900 * len(todo) + 800,
                     "effort": "medium"})
        todo_by[fid] = todo
    res = llm.run_json(cfg, "adjudicate", reqs, gpt_fn, generate)
    for fid, todo in todo_by.items():
        _write(cfg, "adj", parse(res.get(cm.rid(cfg, fid)), todo), {"backend": cfg.backends["adjudicate"], "effort": "medium",
                                                                   "review_mode": review_mode(cfg)})
    return {fid: len(t) for fid, t in todo_by.items()}


def run_gate(cfg: cm.Cfg) -> dict:
    out = cfg.p(cm.s5_dir(cfg), "gate.json")
    if out.exists():
        return cm.read(out)
    all_rows = cm.read(cm.rows_path(cfg))["rows"]
    cats, outcomes = {"positive": [0, 0], "control": [0, 0]}, []
    for r in all_rows:
        if not r["deterministic_pass"]:
            outcomes.append({"row_id": r["row_id"], "accepted": False, "reason": "deterministic"})
            continue
        g, q, a = (_verdict(cfg, k, r["row_id"]) for k in ("primary", "second", "adj"))
        acc = accepts(g, r) and (accepts(q, r) or accepts(a, r))
        cat = "control" if row_kind(r) == "no_memory_control" else "positive"
        cats[cat][0] += acc
        cats[cat][1] += 1
        outcomes.append({"row_id": r["row_id"], "accepted": acc, "primary": accepts(g, r), "second": accepts(q, r),
                         "review_mode": (g or {}).get("review_mode", "gpt_images"),
                         "adjudicated": accepts(a, r) if a else None, "round": cfg.round,
                         "problems": {k: (v or {}).get("problems") for k, v in (("primary", g), ("second", q), ("adj", a))}})
    rates = {c: (a / n if n else None) for c, (a, n) in cats.items()}
    th = cm.THRESHOLDS
    passed = all(rates[c] is not None and rates[c] >= th[f"v8a_{c if c == 'control' else 'positive'}_acceptance_min"]
                 for c in cats)
    acc_ids = {o["row_id"] for o in outcomes if o["accepted"]}
    i17 = [r["row_id"] for r in all_rows if r["row_id"] in acc_ids and not all(
        v["pass"] for k, v in r["checks"].items() if k[:2] in {f"I{i}" for i in range(1, 8)})]
    rep = {"gate": "v9 (v8a pattern)", "rates": rates, "counts": cats, "gate_pass": passed and not i17,
           "i1_i7_fail_on_accepted": i17, "outcomes": outcomes,
           "yield": {"designed_rows": len(all_rows), "deterministic_pass": sum(r["deterministic_pass"] for r in all_rows),
                     "accepted": len(acc_ids)},
           "problem_counts": dict(Counter(p for o in outcomes for v in (o.get("problems") or {}).values() if v
                                          for p, x in v.items() if x == "yes"))}
    cm.write_once(out, rep)
    cm.write_once(cfg.p(cm.s5_dir(cfg), "CANARY_GATE_PASS_v9.json" if rep["gate_pass"] else "CANARY_GATE_FAIL_v9.json"),
                  {"rates": rates})
    return rep
