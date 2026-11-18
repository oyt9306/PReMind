#!/usr/bin/env python3
"""Write-once S1 canary c1b replay assembler. No stores/draft phase: zero new paid calls here.

Reads the preserved, already-paid canary_c1/stores + canary_c1/drafts (sha256-pinned in
PREREG_S1_CANARY_v1b.json) and writes canary_c1b/. Two fixes over canary_construct_v1:
  1. safety priority comes from the shared contract_v1b.expected_safety_priority (single rule);
  2. b7_subtype is derived from (risk, action, withhold) via contract_v1b.derive_b7_subtype;
     the LLM draft value is kept only for direct_answer+risk false when in A1..A4, otherwise
     the row is fail-closed EXCLUDED (B7_subtype_unresolvable). Risk/action are never repaired.
Rows remain pending_review. paid_gpt/preflight(paid=True) serve the v1b review stage only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from final_construction.wp_g_final_benchmark_v1.contract_v1 import (  # noqa: E402
    LABEL_POLICY_VERSION, RISK_MASK_POLICY_VERSION, SCHEMA_VERSION, check_family, check_row,
)
from final_construction.wp_g_final_benchmark_v1.contract_v1b import (  # noqa: E402
    derive_b7_subtype, expected_safety_priority, is_withhold,
)
from final_construction.wp_g_final_benchmark_v1.canary_construct_v1 import (  # noqa: E402
    CANARY_CAP, CELLS, COLLAB, OLD_USAGE, OLD_USAGE_SHA, OLD_USAGE_TOKENS, PACKAGE_CAP,
    PER_CALL_RESERVE, PLAN, RENDERER, USAGE, gpt, sha, usage_tokens, write_once,
)

PREREG = HERE / "PREREG_S1_CANARY_v1b.json"
SOURCE = HERE / "canary_c1"
OUT = HERE / "canary_c1b"
SOURCE_FILES = ("STORES_DONE.json", "DRAFTS_DONE.json",
                *(f"{kind}/{sid}.json" for kind in ("stores", "drafts")
                  for sid in ("c1_bicycle", "c1_clothes", "c1_equipment", "c1_plush")))
CODE_PINS = {
    "runner_sha256": HERE / "canary_construct_v1b.py",
    "review_runner_sha256": HERE / "canary_review_v1b.py",
    "contract_sha256": HERE / "contract_v1.py",
    "contract_v1b_sha256": HERE / "contract_v1b.py",
    "construct_v1_sha256": HERE / "canary_construct_v1.py",
    "review_v1_sha256": HERE / "canary_review_v1.py",
    "prereg_v1_sha256": HERE / "PREREG_S1_CANARY_v1.json",
    "renderer_sha256": RENDERER,
    "adapter_sha256": V3 / "final_construction/train_interface_v1/adapter.py",
    "reward_bridge_sha256": V3 / "final_construction/train_interface_v1/reward_bridge.py",
    "reward_v3_3_sha256": V3 / "rl_sweep/rlvr_v3_prep/r6/rewards_v3_3.py",
    "reward_m1_sha256": V3 / "rl_sweep/rlvr_v3_prep/r4/rewards_v3_1.py",
    "owner_addendum_sha256": V3 / "docs/OWNER_DECISIONS_PHASE_AB_2026-10-01.md",
    "brief_sha256": V3 / "docs/briefs/CODEX_BRIEF_WPG_FINAL_BENCHMARK_v1.md",
}


def source_hashes() -> dict[str, str]:
    return {name: sha(SOURCE / name) for name in SOURCE_FILES}


def preflight(paid: bool = False) -> dict:
    plan = json.loads(PLAN.read_text())
    prereg = json.loads(PREREG.read_text())
    if plan["run_id"] != prereg["source_run_id"]:
        raise RuntimeError("source run ID mismatch")
    if sha(PLAN) != prereg["pins"]["plan_sha256"]:
        raise RuntimeError("plan hash mismatch")
    for name, path in CODE_PINS.items():
        if sha(path) != prereg["pins"][name]:
            raise RuntimeError(f"pinned hash mismatch: {name}")
    if source_hashes() != prereg["source_artifacts_sha256"]:
        raise RuntimeError("preserved c1 stores/drafts hash mismatch")
    if sha(OLD_USAGE) != OLD_USAGE_SHA:
        raise RuntimeError("prior WP-G usage prefix changed; stop for reforecast")
    prefix = USAGE.read_bytes()[:prereg["budget"]["c1_usage_bytes"]]
    if hashlib.sha256(prefix).hexdigest() != prereg["budget"]["c1_usage_sha256"]:
        raise RuntimeError("c1 usage prefix changed; stop for reforecast")
    if len(plan["slots"]) != 4 or plan["row_cap"] != 20 or plan["candidate_call_cap"] != 24:
        raise RuntimeError("unregistered canary shape")
    for slot in plan["slots"]:
        if sha(Path(slot["image_path"])) != slot["image_sha256"]:
            raise RuntimeError(f"source image hash mismatch: {slot['slot_id']}")
    if paid:
        if os.environ.get("PTV2_USAGE_LOG") != str(USAGE):
            raise RuntimeError("PTV2_USAGE_LOG must point to this package's usage.jsonl")
        if prereg["prelaunch_line"] not in COLLAB.read_text():
            raise RuntimeError("owner-approved exact v1b prelaunch line absent")
        spent = usage_tokens()
        if spent + PER_CALL_RESERVE > CANARY_CAP or OLD_USAGE_TOKENS + spent + PER_CALL_RESERVE > PACKAGE_CAP:
            raise RuntimeError(f"BUDGET_BLOCKED: actual={spent}, projected={spent + PER_CALL_RESERVE}, cap={CANARY_CAP}")
    return {**plan, "run_id": prereg["run_id"]}


def paid_gpt(prompt: str, payload: dict, schema: dict, name: str, tag: str, image: str | None = None) -> dict:
    preflight(paid=True)
    if USAGE.exists() and tag in USAGE.read_text():
        raise RuntimeError(f"consumed paid tag cannot be reused: {tag}")
    return gpt(prompt, payload, schema, name, tag, image_path=image, max_out=3500)["output"]


def build_row(plan: dict, slot: dict, index: int, item: dict, items: list[dict], recs: list[dict]) -> tuple[dict | None, list[str]]:
    """One row from one preserved draft item; (None, [reason]) when fail-closed excluded."""
    sid = slot["slot_id"]
    cell = item["cell"]
    indices = item["selected_record_indices"]
    errors: list[str] = []
    if len(indices) != 5 or len(set(indices)) != 5 or any(not isinstance(i, int) or i < 0 or i >= 6 for i in indices):
        errors.append("selected_indices_invalid")
        indices = [0, 1, 2, 3, 4]
    d_idx = item["decisive_record_index"]
    if not isinstance(d_idx, int) or d_idx < -1 or d_idx >= 6:
        errors.append("decisive_index_invalid")
        d_idx = -1
    if item["ownership_source_span"] and item["ownership_source_span"] not in item["query"]:
        errors.append("ownership_span_not_in_query")
    if not item["visual_evidence"] or item["visual_evidence"] not in item["query"]:
        errors.append("visual_span_not_in_query")
    if cell == "G1_gt_absent_evidence_pair" and (item["missing_cue"] != "query_reference" or
                                                 not item["cue_span"] or item["cue_span"] not in item["query"]):
        errors.append("verified_query_reference_cue_missing")
    grounding = ("G1+G2" if cell == "G1+G2_safety" else "general_rule_only" if cell == "general_rule_only" else
                 "G0" if cell == "G0_irrelevant_memory" else "G1")
    need = grounding in {"G1", "G1+G2"}
    present = need and cell != "G1_gt_absent_evidence_pair"
    if need:
        try:
            if (not item["application_support_pattern"] or item["application_support_pattern"] == ".*" or
                    (present and re.search(item["application_support_pattern"], item["gold_answer"], re.IGNORECASE) is None)):
                errors.append("application_pattern_does_not_match_gold_answer")
        except re.error:
            errors.append("application_pattern_regex_invalid")
    if cell == "G1_gt_absent_evidence_pair" and item["application_support_pattern"] != items[0]["application_support_pattern"]:
        errors.append("evidence_pair_application_contract_changed")
    risk = item["physical_safety_or_health_risk"]
    action = item["expected_model_action"]
    conflict = item["personal_constraint_conflict"]
    missing_cue = item["missing_cue"] if need and not present else "none"
    subtype, subtype_source = derive_b7_subtype(risk, action, is_withhold(need, present, missing_cue, action),
                                                item["b7_subtype"])
    if subtype is None:
        return None, ["B7_subtype_unresolvable"]
    decisive = recs[d_idx]["record_id"] if present and d_idx >= 0 else None
    selected = [recs[i]["record_id"] for i in indices]
    category = item["general_knowledge_category"] if grounding in {"G1+G2", "general_rule_only"} else "none"
    masked = bool(cell == "G1_gt_absent_evidence_pair" and item["risk_material_dependence"])
    acute = item["acute_visual_hazard_present"]
    critical_masked = masked and not acute
    source_record = recs[d_idx] if need and d_idx >= 0 else None
    fact = ({"source_record_id": source_record["record_id"], "fact_type": source_record["fact_type"],
             "fact_predicate": source_record["fact_predicate"],
             "fact_value_or_polarity": source_record["fact_value_or_polarity"],
             "same_fact_group_id": source_record["same_fact_group_id"],
             "keyphrases": [source_record["keyphrase"]],
             "application_contract": {"version": "m1.v1", "support_patterns": [item["application_support_pattern"]],
                                      "contrary_patterns": [], "wrong_person_patterns": [],
                                      "hallucination_patterns": [], "condition_name_patterns": []}}
            if source_record else None)
    no_call_targets = ({"risk_label": "uncertain" if item["risk_material_dependence"] and not acute else
                        {"true": "unsafe", "false": "safe", "uncertain": "uncertain"}[risk],
                        "safety_priority": "acute_incident_or_injury_escalation" if acute else
                                           "clarification_under_uncertain_risk" if item["risk_material_dependence"] else
                                           expected_safety_priority(risk, "clarify", conflict, grounding),
                        "expected_action": "safety_escalation" if acute else "clarify",
                        "expected_memory_behavior": "escalate_on_visible_hazard" if acute else "clarify_or_abstain",
                        "risk_reward_masked": bool(item["risk_material_dependence"]),
                        "action_reward_masked": False} if need else None)
    row = {
        "schema_version": SCHEMA_VERSION, "row_id": f"{plan['run_id']}:{sid}:{index}",
        "source_manifest": str(PLAN.relative_to(V3)), "source_manifest_sha256": sha(PLAN),
        "split": "canary_diag", "family_id": f"family:{sid}", "scene_id": slot["donor_scene"],
        "person_id": f"person:{sid}", "store_id": f"store:{sid}", "template_id": f"template:{cell}",
        "image_path": slot["image_path"], "image_sha256": slot["image_sha256"], "query": item["query"],
        "query_target_person_id": f"person:{sid}", "viewer_person_id": f"person:{sid}",
        "grounding_requirement": grounding, "memory_required_for_query": need,
        "decisive_record_present": present, "decisive_record_id": decisive,
        "relevant_record_present": present, "context_condition": "gt_present" if present else
        "gt_absent" if need else "irrelevant_memory_control" if grounding == "G0" else "query_only",
        "expected_memory_behavior": "use" if present else "escalate_on_visible_hazard" if need and acute else
                                    "clarify_or_abstain" if need else "ignore",
        "missing_cue": missing_cue,
        "missing_cue_freeze_id": f"provisional:{plan['run_id']}:{sid}:{index}:K5",
        "missing_cue_evidence": {"source_type": "query", "span": item["cue_span"]} if need and not present else {},
        "general_knowledge_required": grounding in {"G1+G2", "general_rule_only"},
        "general_knowledge_source_category": category,
        "general_knowledge_evidence_status": "pending_review" if category != "none" else "not_applicable",
        "general_knowledge_evidence": {"rule_statement": item["general_knowledge_rule"],
                                       "independent_reviewer_id": "pending_review", "requires_retrieved_kdb": False}
                                      if category != "none" else None,
        "personal_constraint_conflict": conflict, "physical_safety_or_health_risk": risk,
        "constraint_type": item["constraint_type"], "medical_rationale_present": item["medical_rationale_present"],
        "risk_supervision_valid": True, "risk_reward_masked": masked,
        "action_supervision_valid": True, "action_reward_masked": False,
        "label_policy_version": LABEL_POLICY_VERSION, "risk_mask_policy_version": RISK_MASK_POLICY_VERSION,
        "risk_mask_reason": ("omitted_personal_risk_acute_visual_remains" if acute else
                             "omitted_decisive_record_material_to_risk") if masked else "none",
        "risk_material_dependence_verified": item["risk_material_dependence"] if masked else False,
        "remaining_risk_evidence": {"visual": item["remaining_visual_risk_evidence"],
                                    "validated_general": item["remaining_general_risk_evidence"]},
        "active_reward_components": (["action", "personal_content", "format"] if masked else
                                     ["action", "personal_content", "risk", "format"]) + (["critical_miss"] if acute else []),
        "visible_hazard_carveout": acute,
        "visual_acute_evidence": {"present": acute, "evidence": item["acute_visual_hazard_evidence"]},
        "critical_miss_reward_masked": critical_masked,
        "critical_miss_mask_reason": "risk_undeterminable_without_personal_memory" if critical_masked else "none",
        "scenario_group": slot["profile_domain"], "storage_discard": None,
        "decisive_fact": fact, "wrong_person_records": [], "no_call_branch_targets": no_call_targets,
        "renderer_version": "train_interface_v1.exact5_render_v1",
        "adjudication_status": "pending_review",
        "risk_label": {"true": "unsafe", "false": "safe", "uncertain": "uncertain"}[risk],
        "b7_action_mapping": {"model_tag": action, "subtype": subtype, "version": "b7_option_b_v1"},
        "b7_subtype_source": subtype_source, "b7_subtype_llm_draft": item["b7_subtype"],
        "safety_priority": expected_safety_priority(risk, action, conflict, grounding),
        "expected_model_action": action, "records": recs,
        "selected_record_ids": selected, "decisive_ranks": [indices.index(d_idx) + 1] if d_idx in indices and present else [],
        "gt_ranks": [indices.index(d_idx) + 1] if d_idx in indices and present else [],
        "retriever_version": "canary_seeded_selection_v1", "render_k": 5,
        "pair_type": "memory_call" if cell in {"G1_non_safety", "G0_irrelevant_memory"} else
                     "evidence_handling" if cell == "G1_gt_absent_evidence_pair" else None,
        "pair_id": f"pair:{sid}:memory_call" if cell in {"G1_non_safety", "G0_irrelevant_memory"} else
                   f"pair:{sid}:evidence" if cell == "G1_gt_absent_evidence_pair" else None,
        "variant_id": cell,
        "label_group_id": f"label:{sid}:evidence" if cell in {"G1_non_safety", "G1_gt_absent_evidence_pair"} else f"label:{sid}:{cell}",
        "gold_answer_draft": item["gold_answer"], "visual_evidence_draft": item["visual_evidence"],
        "ownership_source_span_draft": item["ownership_source_span"],
    }
    # Diagnostic structural check on a simulated-reviewed copy; emitted row stays pending_review.
    structural = deepcopy(row)
    structural["adjudication_status"] = "adjudicated_valid"
    for record in structural["records"]:
        record["binding_evidence_status"] = "verified"
    if category != "none":
        structural["general_knowledge_evidence_status"] = "verified"
        structural["general_knowledge_evidence"]["independent_reviewer_id"] = "structural_simulation_only"
    return row, sorted(set(errors + check_row(structural)))


def assemble() -> None:
    plan = preflight()
    if OUT.exists():
        raise RuntimeError(f"write-once output directory exists: {OUT}")
    write_once(OUT / "SOURCE_MANIFEST.json", {"run_id": plan["run_id"], "source_dir": str(SOURCE.relative_to(V3)),
                                              "source_artifacts_sha256": source_hashes(), "verified_against": PREREG.name,
                                              "new_paid_calls": 0})
    rows: list[dict] = []
    structural_errors: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for slot in plan["slots"]:
        sid = slot["slot_id"]
        recs = json.loads((SOURCE / "stores" / f"{sid}.json").read_text())["records"]
        items = json.loads((SOURCE / "drafts" / f"{sid}.json").read_text())["raw_response"].get("items", [])
        if len(items) != 5 or [x.get("cell") for x in items] != list(CELLS):
            structural_errors[sid] = ["five_fixed_cells_not_generated"]
            continue
        if any(rec.get("record_subject_id") != f"person:{sid}" or rec.get("applicability_relation") != "self" for rec in recs):
            structural_errors[sid] = ["canary_store_unexpected_other_person_without_reviewed_application_patterns"]
            continue
        for index, item in enumerate(items):
            row, errors = build_row(plan, slot, index, item, items, recs)
            rid = f"{plan['run_id']}:{sid}:{index}"
            if row is None:
                excluded[rid] = errors
                continue
            if errors:
                structural_errors[rid] = errors
            rows.append(row)
    for slot in plan["slots"]:
        group = [r for r in rows if r["store_id"] == f"store:{slot['slot_id']}"]
        if len(group) != 5:
            continue
        base, _, control, _, missing = group
        evidence_base = deepcopy(base)
        evidence_base["pair_type"] = "evidence_handling"
        evidence_base["pair_id"] = missing["pair_id"]
        fam = structural_errors.setdefault(f"family:{slot['slot_id']}", [])
        fam += check_family([base, control, evidence_base, missing])
        if base["query"] != missing["query"]:
            fam.append("evidence_pair_query_not_identical")
        if base["decisive_record_id"] in missing["selected_record_ids"]:
            fam.append("decisive_record_not_dropped")
        if base["risk_label"] != missing["risk_label"]:
            fam.append("risk_label_changed_after_drop")
        if not fam:
            del structural_errors[f"family:{slot['slot_id']}"]
    write_once(OUT / "rows_pending_review.json", {"run_id": plan["run_id"], "rows": rows,
                                                    "status": "pending_independent_review_not_final_data"})
    write_once(OUT / "construction_report.json", {
        "run_id": plan["run_id"], "source_run_id": json.loads(PLAN.read_text())["run_id"],
        "attempted_rows": len(rows) + len(excluded), "emitted_rows": len(rows),
        "structural_errors": structural_errors, "excluded_rows": excluded,
        "b7_subtype_source_counts": {s: sum(r["b7_subtype_source"] == s for r in rows) for s in ("derived", "llm_A1_A4")},
        "usage_tokens": usage_tokens(), "new_paid_calls": 0, "accepted_rows": 0, "review_status": "not_started"})
    if structural_errors or excluded or len(rows) != plan["row_cap"]:
        write_once(OUT / "CONSTRUCTION_FAIL.json", {"error_count": len(structural_errors),
                                                     "excluded_count": len(excluded), "row_count": len(rows)})
        raise RuntimeError("construction gate failed; preserve outputs and stop")
    write_once(OUT / "CONSTRUCTION_PASS.json", {"row_count": len(rows), "not_a_quality_or_release_gate": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "assemble"])
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
        print("S1 c1b preflight PASS (no paid call)")
    else:
        assemble()


if __name__ == "__main__":
    main()
