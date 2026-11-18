"""Stage 4a (CPU): assemble rows from plan + GPT author output + local memory texts / salvaged partner /
self-checks (+ optional wrapped texts); labels by RULE from the stored
retrieval context of each retriever arm (I6, owner correction):
  personal row designed GT-present: all targets in the stored top-K -> gt_present (gold cites every target entry);
      none -> natural retrieval miss: relabelled gt_absent iff A4 holds for that situation, else excluded; partial
      (2-concept, one target retrieved) -> excluded; miss rates are reported;
  constructed GT-absent (evidence-pair partner): targets removed from the store, further same-type hard negative(s)
      added, re-retrieved; built only when A4 holds; gold = the clarifying reply (acute hazard first);
  G0 / general-rule partners: no memory needed; gold cites nothing; same stored context as their G1 partner.
"""
from __future__ import annotations

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s1_plan

GROUNDING = {"G1_visual": "G1", "G1G2_visual": "G1+G2", "G0_closed": "G0", "general_rule_closed": "general_rule_only"}
SLOT = {"G1_visual": "s1", "G1G2_visual": "s2"}
PAIRS = {"G1_visual": "G1", "G0_closed": "G1", "G1G2_visual": "G1G2", "general_rule_closed": "G1G2"}


def safety_priority(risk: str, action: str, grounding: str, personal: bool) -> str:
    """contract_v1 B7 expected-priority rule (risk 'true'/'false')."""
    if risk == "true":
        return "acute_incident_or_injury_escalation" if action == "safety_escalation" else "hazard_prevention_or_mitigation"
    return "normal_personalized_action" if personal or grounding in ("G1", "G1+G2") else "not_safety_relevant"


def family_inputs(cfg: cm.Cfg, fam: dict, use_wrap: bool,
                  max_round: int | None = None) -> tuple[list[dict], dict, dict, str | None]:
    """(items, facts[key][slot], texts[key], exclusion reason). Items carry the GPT author's self-check fields
    (gate) and the local Qwen fields separately as `selfcheck_local` (diagnostic only, F2). The author output is the
    one-round revision when present (F6); a family whose final author output still fails precheck is excluded."""
    from final_construction.wp_v9_vcm import s2_author, s2_local, s2_memtext, s2_revise
    fid = fam["family_id"]
    a = s2_author.load_author(cfg, fid, max_round)
    if not a["valid"]:
        return [], {}, {}, "author_invalid"
    still = s2_revise.blocking(s2_revise.precheck_output(fam, a["output"]))
    if still:
        return [], {}, {}, ("author_checks_failed_after_revise:" if a.get("revision") else
                            "author_checks_failed_no_revision:") + ",".join(sorted({f["check"] for f in still}))
    # memory texts / self-checks of the same round as the author revision in use (rev2 -> round-2 outputs)
    rcfg = cm.Cfg(**{**cfg.__dict__, "round": max(a.get("revision", 0), 1)})       # rev N>=2 -> round-N texts
    mpath = s2_memtext.path(rcfg, fid)
    if not mpath.exists() or not cm.read(mpath)["valid"]:
        return [], {}, {}, "memtext_invalid_or_missing"
    mt = cm.read(mpath)
    scp = cfg.p("s2", s2_local.selfcheck_dir(rcfg), cm.safe_name(fid) + ".json")
    sc = cm.read(scp)["checks"] if scp.exists() else {}
    facts = s2_revise.author_facts(a["output"])
    for key, t in mt["texts"].items():
        for f in t["facts"]:
            facts.setdefault(key, {})[f["slot"]] = f
    texts = {k: t["text"] for k, t in mt["texts"].items()}
    if use_wrap:
        wp = cfg.p("s3", "wrap", cm.safe_name(fid) + ".json")
        wrap = cm.read(wp) if wp.exists() else None
        if wrap is None or wrap["family_dropped"]:
            return [], facts, texts, "wrap_dropped"
        texts.update({k: v["text"] for k, v in wrap["concepts"].items()})
    items = []
    for it in s2_local.family_items(cfg, fid, max_round):
        local = sc.get(it["row_type"]) or {}
        items.append({**it, "selfcheck_local": {k: local.get(k) for k in s2_local.LOCAL_CHECKS} if local else None,
                      "author_revision": a.get("revision", 0)})
    return items, facts, texts, None


def lookalikes(fam: dict) -> set[str]:
    return {k for k, c in fam["concepts"].items() if c["role"] in ("negative", "replacement")}


CONTEXT_MODES = ("retrieved", "gt_retrieved", "gt_retrieved_drop")


def context_for_mode(ctx: dict, targets: list[str], mode: str, p: float, key: str) -> dict:
    """Owner training-context policy. retrieved = the stored offline top-K; gt_retrieved = oracle: every target that
    is in the store is forced into the K (replacing the lowest-ranked non-target); gt_retrieved_drop = oracle, then with
    probability p (seeded per family, so pair members agree) one target is replaced by the next merged-ranking entry.
    Labels are always re-derived by the same rule (arm_labels)."""
    if mode == "retrieved":
        return ctx
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode}")
    merged, entries = ctx["merged_ranking"], [dict(e) for e in ctx["entries"]]
    in_store = [t for t in targets if any(e["key"] == t for e in merged)]
    for t in in_store:
        if t not in [e["key"] for e in entries]:
            idx = max(i for i, e in enumerate(entries) if e["key"] not in targets)
            entries[idx] = {**next(e for e in merged if e["key"] == t), "oracle_inserted": True}
    dropped = None
    if mode == "gt_retrieved_drop" and in_store and cm.rng(f"drop:{key}").random() < p:
        dropped = sorted(in_store)[cm.rng(f"drop_which:{key}").randrange(len(in_store))]
        keys = [e["key"] for e in entries]
        nxt = next((e for e in merged if e["key"] not in keys), None)
        if nxt is not None:
            entries[keys.index(dropped)] = {**nxt, "drop_replacement": True}
    return {**ctx, "entries": entries, "context_mode": mode, "drop_p": p, "dropped_target": dropped}


def arm_labels(fam: dict, item: dict, row_type: str, ctx: dict, facts: dict, constructed_absent: bool) -> dict:
    keys = [e["key"] for e in ctx["entries"]]
    k = fam["k"]
    perm = fam["display_perm"]
    if len(keys) != k or len(set(keys)) != k:
        return {"excluded": "context_size_mismatch", "context_keys": keys}
    display = [keys[r] for r in perm]                       # display slot j (1-based) shows rank perm[j-1]
    out = {"excluded": None, "retriever": ctx["retriever"], "retriever_id": ctx["id"], "revision": ctx["revision"],
           "k": k, "ranked": ctx["entries"], "display_keys": display,
           "display_to_rank": {str(j + 1): perm[j] + 1 for j in range(k)}}
    cond = s1_plan.condition(keys, fam["targets"])
    personal = row_type in SLOT
    acute = bool(item["acute_visual_hazard_present"])
    base = {"gold_answer": item["gold_answer"], "expected_model_action": item["expected_model_action"],
            "b7_subtype": item["b7_subtype"], "gold_citation_set": [], "support_patterns": [],
            "decisive_keyphrases": [], "wrong_concept_patterns": {}, "absent_reason": None, "condition": cond,
            "gold_action_set": [item["expected_model_action"]], "gold_answer_reference_only": False}
    if not personal:
        return {**out, **base, "condition_role": "no_memory_needed", "expected_memory_behavior": "ignore"}
    slot = SLOT[row_type]
    if any(slot not in facts.get(t, {}) for t in fam["targets"]):
        return {**out, **base, "excluded": "missing_target_fact"}
    wrong = {str(j + 1): [facts[key][slot]["application_pattern"]] for j, key in enumerate(display)
             if key in lookalikes(fam) and slot in facts.get(key, {})}
    if cond == "partial":
        return {**out, **base, "excluded": "partial_retrieval_two_concept"}
    if cond == "gt_absent":
        from final_construction.wp_v9_vcm import s4_checks
        critical = s4_checks.decision_critical(fam, facts, item, slot)["pass"]       # R1: A4 needs a critical fact
        if not item["self_check"]["a4_missing_fact_required"] or not item["gt_absent_response"].strip() or not critical:
            return {**out, **base, "excluded": "retrieval_miss_a4_not_required" if not constructed_absent
                    else "absent_a4_not_required"}
        return {**out, **base, "gold_answer": item["gt_absent_response"],
                "expected_model_action": "safety_escalation" if acute else "clarify",
                "b7_subtype": "A9" if acute else "A7", "wrong_concept_patterns": wrong,
                # P5: GT-absent = WITHHOLD - clarify OR abstain is correct (acute: escalate); the gold is one reference
                "gold_action_set": ["safety_escalation"] if acute else ["clarify", "abstain"],
                "gold_answer_reference_only": True,
                "absent_reason": "constructed" if constructed_absent else "retrieval_miss",
                "expected_memory_behavior": "clarify_or_abstain"}
    tf = [facts[t][slot] for t in fam["targets"]]
    return {**out, **base, "gold_citation_set": sorted(j + 1 for j, key in enumerate(display) if key in fam["targets"]),
            "support_patterns": [f["application_pattern"] for f in tf],
            "decisive_keyphrases": [f["keyphrase"] for f in tf], "wrong_concept_patterns": wrong,
            "expected_memory_behavior": "use"}


def build_family_rows(fam: dict, items_list: list[dict], facts: dict, texts: dict,
                      run_id: str = cm.RUN_ID, gen_version: str = cm.GEN_ORIGINAL) -> list[dict]:
    items = {it["row_type"]: it for it in items_list}
    plans = [(rt, "full", False) for rt in ("G1_visual", "G0_closed", "G1G2_visual", "general_rule_closed")]
    plans += [(rt, "absent", True) for rt in ("G1_visual", "G1G2_visual")]
    rows = []
    for rt, variant, absent in plans:
        item = items.get(rt)
        if item is None:
            continue
        arms = {r: arm_labels(fam, item, rt, fam["retrieval"][r][variant], facts, absent) for r in cm.RETRIEVERS}
        main = arms[cm.MAIN_RETRIEVER]
        name = rt + ("_absent" if absent else "")
        risk = "true" if item["risk"] == "unsafe" else "false"
        action = main.get("expected_model_action", item["expected_model_action"])
        query, style = item["query"], None
        if rt == "G0_closed" and risk == "true":         # unsafe G0: query chosen in code (panic / calm list)
            from final_construction.wp_v9_vcm import s4_g0query
            acute = bool(item["acute_visual_hazard_present"])
            query = s4_g0query.choose(fam, f"{run_id}:{fam['family_id']}:{name}", acute,
                                      main.get("gold_answer") or item["gold_answer"],
                                      [item["acute_visual_hazard_evidence"]] if acute else [])
            style = s4_g0query.style_of_query(query)        # analysis only: the query is label-independent
        rows.append({
            "schema_version": "s2v9_vcm_row_v1", "row_id": f"{run_id}:{fam['family_id']}:{name}",
            "gen_version": gen_version,                     # g1 original, g2 regeneration (one per family in a release)
            "family_id": fam["family_id"], "split": fam["split"], "person_id": fam["person_id"],
            "scene_id": fam["scene_id"], "image_key": fam["image_key"], "image_path": fam["image"],
            "image_sha256": fam["image_sha256"], "row_type": name, "grounding_requirement": GROUNDING[rt],
            "designed_condition": "gt_absent" if absent else ("gt_present" if rt in SLOT else "no_memory_needed"),
            "memory_call_pair_id": None if absent else f"mc:{fam['family_id']}:{PAIRS[rt]}",
            "evidence_pair_id": f"ev:{fam['family_id']}:{PAIRS[rt]}" if rt in SLOT else None,
            "n_targets": fam["n_targets"], "k": fam["k"], "target_keys": fam["targets"],
            "target_kinds": sorted({fam["concepts"][t]["kind"] for t in fam["targets"]}),
            "relaxed": any(i["relaxed"] for i in fam["target_info"].values()),
            "stratum": "relaxed" if any(i["relaxed"] for i in fam["target_info"].values()) else "strict",
            "query": query, "query_style": style, "risk_label": "unsafe" if risk == "true" else "safe",
            "physical_safety_or_health_risk": risk, "risk_requested": fam["risk_plan"][rt],
            "risk_infeasible": item["risk_infeasible"],
            "acute_visual_hazard": {"present": item["acute_visual_hazard_present"],
                                    "evidence": item["acute_visual_hazard_evidence"],
                                    "gold_pattern": (item.get("acute_absent_gold_pattern") or item["acute_gold_pattern"])
                                    if absent else item["acute_gold_pattern"]},          # audit 10b: name-free
            "general_knowledge_rule": item["general_knowledge_rule"], "used_fact_slot": item["used_fact_slot"],
            "category_nouns": item["category_nouns"], "self_check": item["self_check"],
            "selfcheck_local": item.get("selfcheck_local"), "author_revision": item.get("author_revision", 0),
            "salvaged_from": item.get("salvaged_from"),
            "safety_priority": safety_priority(risk, action, GROUNDING[rt], rt in SLOT),
            "arms": arms, **{k: main.get(k) for k in ("gold_answer", "expected_model_action", "b7_subtype",
                                                      "gold_citation_set", "support_patterns", "decisive_keyphrases",
                                                      "wrong_concept_patterns", "condition", "absent_reason",
                                                      "gold_action_set", "gold_answer_reference_only",
                                                      "expected_memory_behavior")},
            "entries": [{"display_index": j + 1, "key": key, "image": fam["concepts"][key]["entry"],
                         "text": texts.get(key, "")} for j, key in enumerate(main.get("display_keys", []))],
            "excluded": main["excluded"]})
    return rows


def risk_request_report(rows: list[dict]) -> dict:
    """P1: per row type, unsafe requested vs delivered and the risk_infeasible fallbacks (counted, never hidden)."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["row_type"].endswith("_absent") or r.get("carried_from_round"):
            continue
        c = out.setdefault(r["row_type"], {"rows": 0, "requested_unsafe": 0, "delivered_unsafe": 0,
                                           "infeasible_fallback": 0})
        c["rows"] += 1
        c["requested_unsafe"] += r["risk_requested"] == "unsafe"
        c["delivered_unsafe"] += r["risk_label"] == "unsafe"
        c["infeasible_fallback"] += r["risk_requested"] == "unsafe" and r["risk_label"] == "safe"
    return out


def assemble_family(cfg: cm.Cfg, fam: dict, use_wrap: bool, pool_ids: dict,
                    max_round: int | None = None) -> tuple[list[dict], str | None]:
    """(rows with checks under the CURRENT rules, exclusion reason) of one family from its round-`max_round` inputs."""
    from final_construction.wp_v9_vcm import s4_checks
    items, facts, texts, reason = family_inputs(cfg, fam, use_wrap, max_round)
    if reason:
        return [], reason
    fam_rows = build_family_rows(fam, items, facts, texts, cfg.run_id, cm.gen_version(cfg))
    fam_checks = s4_checks.family_checks(fam, facts, texts)
    from final_construction.wp_v9_vcm import s2_author, s4_naming, s4_scene
    if s4_naming.active(fam):                     # M7 on author + memtext facts against the family's queries
        m7 = s4_scene.scene_anchor_problems(fam, facts, [i for i in items if i["row_type"] in s2_author.ROW_TYPES])
        fam_checks["M7_scene_anchor"] = s4_checks.ok(not m7, m7[:8])
    from final_construction.wp_v9_vcm import s4_acute                 # S1-S4 row-level (acute/G0 fix 2026-10-03)
    acute = s4_acute.row_results(fam, fam_rows, items)
    for r in fam_rows:
        r["checks"] = {**s4_checks.row_checks(fam, facts, r, fam_checks, pool_ids), **acute[r["row_id"]]}
        if r.get("query_style"):                      # code-set generic query: the closed floor (>= 5 words), I1-I2
            from final_construction.wp_v9_vcm import s4_g0query
            r["checks"].update(s4_g0query.generic_checks(fam, r["query"]))
        r["deterministic_pass"] = r["excluded"] is None and all(v["pass"] for v in r["checks"].values())
    return fam_rows, None


def run_assemble(cfg: cm.Cfg, use_wrap: bool = cm.DIALOGUE_WRAPPING_DEFAULT) -> dict:
    from collections import Counter
    from final_construction.wp_v9_vcm import s2_author
    if cm.is_extra(cfg):                                # round 7: additive rows (s2_extra)
        from final_construction.wp_v9_vcm import s2_extra
        return s2_extra.assemble_extra(cfg)
    if cm.is_rereview(cfg):                             # round 9: re-review of rejected safe generic rows
        from final_construction.wp_v9_vcm import s5_rereview
        return s5_rereview.assemble_rereview(cfg)
    out = cm.rows_path(cfg)
    if out.exists():
        return {"status": "exists"}
    plan = cm.load_plan(cfg)
    pool_ids = cm.read(cfg.p("s0", "pool.json"))["identities"]
    rows, fam_status = [], {}
    for fam in plan["families"]:
        if not s2_author.in_round(cfg, fam["family_id"]):
            continue
        fam_rows, reason = assemble_family(cfg, fam, use_wrap, pool_ids)
        rows += fam_rows
        fam_status[fam["family_id"]] = reason or "assembled"
    by_id = {f["family_id"]: f for f in plan["families"]}
    for fid, src in sorted(s2_author.carried_families(cfg).items()):
        # R7 carry-forward: the family's round-`src` inputs (no new call) re-assembled, so the checks are the current
        # rules (pilot fix: copying stale rows kept rule false positives); content is identical, review reuses verdicts
        fam_rows, reason = assemble_family(cfg, by_id[fid], use_wrap, pool_ids, max_round=src)
        rows += [{**r, "carried_from_round": src} for r in fam_rows]
        fam_status[fid] = reason or "assembled"
    designed_present = [r for r in rows if r["designed_condition"] == "gt_present"]
    rep = {"families": fam_status, "rows": len(rows), "deterministic_pass": sum(r["deterministic_pass"] for r in rows),
           "excluded": dict(Counter(r["excluded"] for r in rows if r["excluded"])),
           "failed_checks": dict(Counter(k for r in rows for k, v in r["checks"].items() if not v["pass"])),
           "natural_miss_rate": round(sum(r["absent_reason"] == "retrieval_miss" or (r["excluded"] or "").startswith(
               "retrieval_miss") for r in designed_present) / max(1, len(designed_present)), 4),
           "use_wrap": use_wrap, "round": cfg.round, "carried": s2_author.carried_families(cfg),
           "risk_requests": risk_request_report(rows)}
    cm.write_once(out, {"rows": rows, "report": rep})
    return rep
