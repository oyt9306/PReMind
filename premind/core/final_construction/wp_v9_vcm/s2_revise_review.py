"""Review-driven revision round N >= 2 (R5, generalised by R7): `revise_review --round N`.

The previous round P = N-1 is read without modification:
  * families of round P (round 1: every assembled/excluded family; P >= 2: the families of rows_<main>_rP.json);
  * deterministic failures = hard precheck failures of the author output used in round P + the failing checks of its
    round-P rows;
  * review problems = the rejected rows of the latest REVIEWED round <= P that contains the family (gate file exists).
A family is revised in round N if it has deterministic failures, or round P was reviewed and rejected some of its
authored rows (a rejected salvaged general-rule control is not GPT-authored and is simply dropped).
A family that is clean in an UNREVIEWED round P (e.g. round 2 stopped after assemble) is carried forward without a call:
its round-P rows are copied into round N and reviewed there (author_rev<N>/CARRY_FORWARD.json). Revisions go through
s2_revise.revise_with_feedback (verified loop, <= 2 GPT attempts). Families accepted in a reviewed round stay final.
Production retry (`--retry`, round 2 of a final unit, owner 2026-10-03): revised = every family with a rejected
author-fixable row (review or deterministic; not the salvaged general-rule control, not a partial-retrieval exclusion)
plus every family with zero accepted rows (author-excluded families through their deterministic failures; a zero-row
family without any author failure, e.g. memtext_invalid, is copied into author_rev2 unchanged with no call so its
memtext is redone). The unit's first paid step is a NEW run: an open stop file or a start projection above
cm.PROD_BUDGET_STOP refuses it (s7_retry.start_gate) before CARRY_FORWARD.json, the unit's start marker, is written.
"""
from __future__ import annotations

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import gpt_client as gc
from final_construction.wp_v9_vcm import llm
from final_construction.wp_v9_vcm import s2_author, s2_revise


def rows_file(cfg: cm.Cfg, rnd: int):  # noqa: ANN201
    return cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}{'' if rnd == 1 else f'_r{rnd}'}.json")


def gate_file(cfg: cm.Cfg, rnd: int):  # noqa: ANN201
    return cfg.p("s5" if rnd == 1 else f"s5_r{rnd}", "gate.json")


def _verdicts(cfg: cm.Cfg, rnd: int, rid: str) -> dict:
    out = {}
    for kind in ("primary", "second", "adj"):
        p = cfg.p("s5" if rnd == 1 else f"s5_r{rnd}", kind, cm.safe_name(rid) + ".json")
        if p.exists():
            v = cm.read(p)
            out[kind] = {"problems": sorted(k for k, x in (v.get("problems") or {}).items() if x == "yes"),
                         "rationale": v.get("rationale"), "valid": v.get("valid")}
    return out


def _reason(row: dict, outcome: dict) -> str:
    """Gate reason of a rejected row; a deterministic rejection names its exclusion or failing checks."""
    if outcome.get("reason", "review") != "deterministic":
        return outcome.get("reason", "review")
    return "deterministic: " + (row.get("excluded") or ",".join(sorted(k for k, v in row.get("checks", {}).items()
                                                                     if not v["pass"])))


def review_problems(cfg: cm.Cfg, upto: int, fid: str) -> tuple[int | None, list[dict]]:
    """(reviewed round, rejected rows of fid) for the latest reviewed round <= upto that has rows of fid."""
    for rnd in range(upto, 0, -1):
        if not gate_file(cfg, rnd).exists():
            continue
        rows = {r["row_id"]: r for r in cm.read(rows_file(cfg, rnd))["rows"] if r["family_id"] == fid}
        if not rows:
            continue
        rej = [{"row_type": rows[o["row_id"]]["row_type"], "query": rows[o["row_id"]]["query"],
                "gold_answer": rows[o["row_id"]]["gold_answer"], "condition": rows[o["row_id"]]["condition"],
                "gate_reason": _reason(rows[o["row_id"]], o), "review": _verdicts(cfg, rnd, o["row_id"]),
                "excluded": rows[o["row_id"]].get("excluded")}
               for o in cm.read(gate_file(cfg, rnd))["outcomes"] if o["row_id"] in rows and not o["accepted"]]
        return rnd, rej
    return None, []


REVIEW_REVISE_ATTEMPTS_PROD = 1
MEMTEXT_CHECKS = ("I3_keyphrase_verbatim", "M1_matched_fact_structure", "memory_text_no_visual_words", "I7_no_leak")
MEMTEXT_SLOTS = ("o1", "o2", "x")


def memtext_attributable(check: str, detail: object) -> bool:
    """Row failures caused by the memory-text stage (not by the author output): text checks, and A2 hits that only
    involve memtext-written o1/o2/x facts. They never trigger an author revision."""
    if check in MEMTEXT_CHECKS:
        return True
    if check == "M7_scene_anchor":                            # scene wording only in memtext-written extras
        return isinstance(detail, list) and bool(detail) and all(d[1] in MEMTEXT_SLOTS for d in detail)
    if check == "M4_voice":                                   # first person only in memtext-written facts/entries
        return isinstance(detail, list) and bool(detail) and all(d[1] in MEMTEXT_SLOTS + ("entry",) for d in detail)
    if check in ("M5_ownership_header", "M6_action_conflict"):  # a memtext fact / the code header is involved
        return isinstance(detail, list) and bool(detail) and all(
            any(x in MEMTEXT_SLOTS + ("header",) for x in d[1:4:2] if isinstance(x, str)) for d in detail)
    return check == "A2_one_fact_per_target" and isinstance(detail, list) and bool(detail) and all(
        isinstance(d, (list, tuple)) and len(d) == 2 and d[1] in MEMTEXT_SLOTS for d in detail)


def det_failures(cfg: cm.Cfg, fam: dict, rnd: int, full_fam: dict | None = None) -> list[dict]:
    """Author-attributable deterministic failures of round rnd (hard precheck failures + failing row checks)."""
    a = s2_author.load_author(cfg, fam["family_id"], max_round=rnd)
    fails = s2_revise.hard(s2_revise.precheck_output(fam, a["output"] if a["valid"] else None))
    if not rows_file(cfg, rnd).exists():
        return fails
    from final_construction.wp_v9_vcm import s4_rows            # pilot fix: row checks re-run under the current
    use_wrap = cm.read(rows_file(cfg, rnd))["report"].get("use_wrap", False)          # rules (stored ones go stale)
    full_fam = full_fam or next(f for f in cm.load_plan(cfg)["families"] if f["family_id"] == fam["family_id"])
    rows, _ = s4_rows.assemble_family(cfg, full_fam, use_wrap, cm.read(cfg.p("s0", "pool.json"))["identities"], rnd)
    for r in rows:
        if not r["deterministic_pass"] and r["excluded"] is None:
            fails += [{"item": r["row_type"], "check": k, "detail": v["detail"]} for k, v in r["checks"].items()
                      if not v["pass"] and not memtext_attributable(k, v["detail"])]
    return fails


NOT_AUTHOR_FIXABLE = ("partial_retrieval_two_concept", "context_size_mismatch")     # retrieval structure, not text


def _accepted_counts(cfg: cm.Cfg) -> dict[str, int]:
    """Accepted rows per family in the unit's final round-1 result (s5_final; retry only)."""
    from collections import Counter
    rows = {r["row_id"]: r["family_id"] for r in cm.read(cfg.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final.json"))["rows"]}
    return dict(Counter(rows[o["row_id"]] for o in cm.read(cfg.p("s5_final", "gate.json"))["outcomes"]
                        if o["accepted"] and o["row_id"] in rows))


def select_families(cfg: cm.Cfg, policy: bool = True) -> dict:
    """CPU selection of round N (no write, no call): todo (fam, prev output, det, review problems), carried, clean
    final, and (retry) memtext-only zero-row families. Retry: a zero-row family whose only rejections are not
    author-fixable is `unfixable` (not revised); with `policy` and s1/RETRY_FAMILIES_v1.json present (owner policy
    zero_or_dev) an eligible family outside that list is `not_selected` (not revised: finalize keeps round 1)."""
    from final_construction.wp_v9_vcm import s7_retry
    n = cfg.round
    prev = n - 1
    if not rows_file(cfg, prev).exists():
        raise RuntimeError(f"round {prev} has no assembled rows")
    plan = {f["family_id"]: f for f in cm.plan_raw(cfg)["families"]}
    fams = sorted(cm.read(rows_file(cfg, prev))["report"]["families"])
    full = {f["family_id"]: f for f in cm.load_plan(cfg)["families"]}          # with every retriever arm (assemble)
    prev_reviewed = gate_file(cfg, prev).exists()
    acc = _accepted_counts(cfg) if cfg.retry else {}
    chosen = s7_retry.selected(cfg) if cfg.retry and policy else None             # None = full policy
    todo, carried, clean_final, memtext_only, unfixable, not_selected = [], {}, [], [], [], []
    for fid in fams:
        if s2_author.author_path(cfg, fid, n).exists():
            continue
        det = det_failures(cfg, plan[fid], prev, full[fid])
        rnd, rej_all = review_problems(cfg, prev, fid)
        rej = [r for r in rej_all if r["row_type"] != "general_rule_closed"]   # salvaged control: not authored by GPT
        if cfg.retry:
            rej = [r for r in rej if r.get("excluded") not in NOT_AUTHOR_FIXABLE]
            zero_clean = not det and not rej and acc.get(fid, 0) == 0           # zero-row, no fixable problem
            eligible = zero_clean and not rej_all or bool(det) or bool(prev_reviewed and rej)
            if zero_clean and rej_all:
                unfixable.append(fid)
            if chosen is not None and eligible and fid not in chosen:
                not_selected.append(fid)
            if zero_clean or fid in not_selected:
                (memtext_only if eligible and fid not in not_selected else clean_final).append(fid)
                continue
        if det or (prev_reviewed and rej):
            a = s2_author.load_author(cfg, fid, max_round=prev)
            todo.append((plan[fid], a["output"] if a["valid"] else None, det, rej))
        elif not prev_reviewed:
            carried[fid] = prev
        else:
            clean_final.append(fid)
    return {"todo": todo, "carried": carried, "clean_final": clean_final, "memtext_only": memtext_only,
            "unfixable": unfixable, "not_selected": not_selected, "families": len(fams),
            "zero_row": sorted(f for f in fams if cfg.retry and acc.get(f, 0) == 0)}


def _memtext_only(cfg: cm.Cfg, fids: list[str]) -> None:
    """Zero-row family with no author failure (retry): the round-1 author output enters round N unchanged (no call),
    so memtext / assemble / review redo the family; finalize keeps round 1 unless round N does at least as well."""
    for fid in fids:
        a = s2_author.load_author(cfg, fid, max_round=cfg.round - 1)
        cm.write_once(s2_author.author_path(cfg, fid, cfg.round),
                      {"family_id": fid, "valid": a["valid"], "output": a["output"], "revision": cfg.round,
                       "chosen_attempt": None, "attempts": [], "failures_addressed": [], "review_problems": [],
                       "remaining_failures": [], "copied_from_revision": a["revision"], "backend": "copy (no call)",
                       "prompt_sha256": None})


def run_revise_review(cfg: cm.Cfg, gpt_fn: gc.GptFn | None = None, generate: llm.Generate | None = None) -> dict:
    from final_construction.wp_v9_vcm import s7_retry
    if cm.is_fix(cfg):                                  # acute/G0 repair round 3 of a final unit
        from final_construction.wp_v9_vcm import s2_fix
        return s2_fix.run_fix(cfg, gpt_fn, generate)
    if cm.is_extra(cfg):                                # round 7: additive rows (text-only generation)
        from final_construction.wp_v9_vcm import s2_extra
        return s2_extra.run_extra(cfg, gpt_fn, generate)
    n = cfg.round
    if n < 2:
        raise ValueError("revise_review runs with --round N >= 2")
    sel = select_families(cfg)
    todo, carried, clean_final = sel["todo"], sel["carried"], sel["clean_final"]
    carry = cfg.p("s2", f"author_rev{n}", "CARRY_FORWARD.json")
    if not carry.exists():
        if cfg.retry:                       # the unit's first paid step: refused while a stop is open / over budget
            s7_retry.start_gate(cfg, len(todo) + len(sel["memtext_only"]))
        cm.write_once(carry, {"round": n, "carried": carried, "source_rows": str(rows_file(cfg, n - 1).name),
                              "clean_and_reviewed_final": clean_final, "retry": cfg.retry,
                              "revise": sorted(f["family_id"] for f, *_ in todo), "memtext_only": sel["memtext_only"],
                              "families": sel["families"], "unfixable_zero_row": sel["unfixable"],
                              "not_selected": sel["not_selected"], "policy": s7_retry.policy(cfg) if cfg.retry else None})
    _memtext_only(cfg, sel["memtext_only"])
    # Lever 1 (production): ONE GPT attempt per family; a revision that still fails precheck is recorded (attempts/,
    # remaining_failures) and its rows are not assembled - finalize then keeps the family's earlier accepted rows
    rep = s2_revise.revise_with_feedback(cfg, "revise_review", n, todo, gpt_fn, generate,
                                         max_attempts=REVIEW_REVISE_ATTEMPTS_PROD if cfg.delta else s2_revise.MAX_ATTEMPTS)
    return {**rep, "carried": carried, "final_already": clean_final, "memtext_only": sel["memtext_only"]}
