"""CPU-only final-benchmark contract v1; no model or scheduler entrypoint.

This is an additive successor to WP-G contract_r2. It intentionally does not
render prompts, admit training rows, or certify final-benchmark real rows.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


SCHEMA_VERSION = "final_benchmark_contract_v1"
LABEL_POLICY_VERSION = "b6_option_b_rule_v1"
RISK_MASK_POLICY_VERSION = "df07_variant_material_dependence_v1"
GROUNDING = {"G0", "G1", "G1+G2", "general_rule_only"}
CONTEXTS = {"gt_present", "gt_absent", "distractor_only", "query_only", "irrelevant_memory_control"}
CUES = {"other_person", "referential", "query_reference", "none"}
FUTURE_ONLY = {"G1-T", "future_temporal", "future_kdb", "G2_explicit_kdb"}
ALLOWED_PROVENANCE = {"canonical_person_history", "canonical_household", "canonical_object", "synthetic_distractor"}
PREDICATE_TYPE = {
    "clothing_care": "preference", "household_routine": "preference",
    "care_instruction": "preference", "object_ownership": "ownership",
    "access_boundary": "ownership", "mobility_accessibility": "medical",
    "allergy_food": "allergy", "medical_restriction": "medical",
}
SCENARIO_GROUPS = {"clothing_care", "care_instruction", "mobility_routine", "household_routine",
                   "food_allergy", "food_storage", "ownership_boundary", "medical_accessibility"}
FIELD_PRODUCERS = {
    "scenario_group": "seed-fixed slot.profile_domain; controlled SCENARIO_GROUPS taxonomy",
    "storage_discard": "canary assembler: null outside food_storage, verified boolean inside food_storage",
    "decisive_fact": "independent store record plus drafter's application regex; literal keyphrase and dual review required",
    "wrong_person_records": "deterministic selected-record subject comparison against query target; never guessed",
    "no_call_branch_targets": "deterministic no-PDB branch from verified risk material dependence and visible acute evidence",
    "visible_hazard_carveout": "drafter acute-image claim confirmed or rejected by independent image reviewers",
    "visual_acute_evidence": "drafter image-grounded evidence statement; independently image-reviewed",
    "critical_miss_reward_masked": "deterministic GB-1 conjunction of risk mask and absence of visible acute hazard",
}
NO_CALL_TARGET_KEYS = {"risk_label", "safety_priority", "expected_action", "expected_memory_behavior",
                       "risk_reward_masked", "action_reward_masked"}
B7_TAG = {"A1": "direct_answer", "A2": "direct_answer", "A3": "direct_answer",
          "A4": "direct_answer", "A5": "clarify", "A6": "clarify",
          "A8": "direct_answer", "A9": "safety_escalation"}
B7_RISK = {"A1": "false", "A2": "false", "A3": "false", "A4": "false",
           "A5": "false", "A6": "uncertain", "A8": "true", "A9": "true"}
REQUIRED = {
    "schema_version", "row_id", "source_manifest", "source_manifest_sha256", "split", "family_id",
    "scene_id", "person_id", "store_id", "template_id", "image_path", "image_sha256",
    "query", "query_target_person_id", "viewer_person_id", "grounding_requirement",
    "memory_required_for_query", "decisive_record_present", "decisive_record_id",
    "relevant_record_present", "context_condition", "expected_memory_behavior",
    "missing_cue", "missing_cue_freeze_id", "missing_cue_evidence", "general_knowledge_required",
    "general_knowledge_source_category", "general_knowledge_evidence_status",
    "general_knowledge_evidence",
    "personal_constraint_conflict", "physical_safety_or_health_risk", "constraint_type",
    "medical_rationale_present", "risk_supervision_valid", "risk_reward_masked",
    "action_supervision_valid", "action_reward_masked", "label_policy_version",
    "risk_mask_policy_version", "risk_mask_reason", "risk_material_dependence_verified",
    "remaining_risk_evidence", "active_reward_components", "adjudication_status",
    "risk_label", "b7_action_mapping", "safety_priority", "expected_model_action",
    "records", "selected_record_ids", "decisive_ranks", "gt_ranks", "retriever_version",
    "render_k", "pair_type", "pair_id", "variant_id", "label_group_id",
    "visible_hazard_carveout", "visual_acute_evidence", "critical_miss_reward_masked",
    "critical_miss_mask_reason", "scenario_group", "storage_discard", "decisive_fact",
    "wrong_person_records", "no_call_branch_targets", "renderer_version",
}
SPLIT_KEYS = ("scene_id", "person_id", "store_id", "template_id", "family_id", "image_sha256")


def check_row(row: dict[str, Any]) -> list[str]:
    """Reject invalid or incomplete *reviewed* rows. No defaults are supplied."""
    errors = [f"missing:{key}" for key in sorted(REQUIRED - row.keys())]
    if errors:
        return errors
    for key in ("row_id", "source_manifest", "source_manifest_sha256", "scene_id", "person_id",
                "store_id", "template_id", "family_id", "image_path", "image_sha256", "query",
                "query_target_person_id", "viewer_person_id", "retriever_version", "missing_cue_freeze_id"):
        if not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"empty:{key}")
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version_incompatible")
    if row["label_policy_version"] != LABEL_POLICY_VERSION:
        errors.append("label_policy_version_incompatible")
    if row["risk_mask_policy_version"] != RISK_MASK_POLICY_VERSION:
        errors.append("risk_mask_policy_version_incompatible")
    if row["adjudication_status"] not in {"not_required", "adjudicated_valid"}:
        errors.append("adjudication_unresolved")
    if not isinstance(row["grounding_requirement"], str) or row["grounding_requirement"] in FUTURE_ONLY or row["grounding_requirement"] not in GROUNDING:
        errors.append("grounding_not_released")
    if not isinstance(row["context_condition"], str) or row["context_condition"] not in CONTEXTS:
        errors.append("context_invalid")
    if not isinstance(row["missing_cue"], str) or row["missing_cue"] not in CUES:
        errors.append("missing_cue_invalid")
    if row["missing_cue"] == "query_reference":
        cue = row["missing_cue_evidence"]
        if not isinstance(cue, dict) or cue.get("source_type") != "query" or not cue.get("span") or cue["span"] not in row["query"]:
            errors.append("query_reference_cue_not_visible")
    if row["render_k"] != 5:
        errors.append("render_k_not_five")
    if row["query_target_person_id"] != row["person_id"]:
        errors.append("target_binding_mismatch")

    records = row["records"]
    if not isinstance(records, list) or len(records) < 5:
        errors.append("store_under_five_real_records")
        records = []
    ids: list[str] = []
    by_id: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            errors.append("record_not_object")
            continue
        rid = record.get("record_id")
        if not isinstance(rid, str) or not rid:
            errors.append("record_id_missing")
            continue
        ids.append(rid)
        by_id[rid] = record
        if record.get("store_id") != row["store_id"]:
            errors.append("cross_store_record")
        if record.get("provenance_class") not in ALLOWED_PROVENANCE:
            errors.append("record_provenance_forbidden")
        if record.get("query_derived") is not False or record.get("other_row_source_id"):
            errors.append("query_or_other_row_derived_record")
        if record.get("filler_status") is not False or record.get("source_type") != "personal_database":
            errors.append("filler_or_kdb_record_forbidden")
        if not record.get("source_artifact_sha256") or not record.get("record_subject_id"):
            errors.append("record_provenance_or_subject_missing")
        if record.get("applicability_relation") not in {"self", "authorized_household", "none"}:
            errors.append("record_applicability_relation_invalid")
        elif record.get("record_subject_id") == row["person_id"] and record.get("applicability_relation") != "self":
            errors.append("target_record_applicability_invalid")
        elif record.get("record_subject_id") != row["person_id"] and record.get("applicability_relation") == "self":
            errors.append("other_person_cannot_have_self_applicability")
        predicate = record.get("fact_predicate")
        fact_type = record.get("fact_type")
        if not isinstance(predicate, str) or PREDICATE_TYPE.get(predicate) != fact_type or record.get("same_fact_group_id") != f"final_v1:{fact_type}:{predicate}":
            errors.append("H2_fact_taxonomy_mismatch")
        if record.get("binding_key_version") != "final_person_key_v1":
            errors.append("H2_person_binding_key_invalid")
    if len(set(ids)) != len(ids):
        errors.append("duplicate_record_id")
    selected = row["selected_record_ids"]
    if not isinstance(selected, list) or len(selected) != 5 or len(set(selected)) != 5:
        errors.append("selected_not_exactly_five_unique")
        selected = []
    if any(rid not in by_id for rid in selected):
        errors.append("selected_record_not_in_store")
    for rid in set(selected + ([row["decisive_fact"].get("source_record_id")] if isinstance(row["decisive_fact"], dict) else [])):
        if rid in by_id and by_id[rid].get("binding_evidence_status") != "verified":
            errors.append("H2_person_binding_unverified")

    grounding = row["grounding_requirement"] if isinstance(row["grounding_requirement"], str) else ""
    need = row["memory_required_for_query"]
    present = row["decisive_record_present"]
    decisive = row["decisive_record_id"]
    if grounding in {"G0", "general_rule_only"}:
        if need is not False or present is not False or decisive is not None or row["expected_memory_behavior"] != "ignore":
            errors.append("H1_no_pdb_mapping")
        if row["context_condition"] not in {"irrelevant_memory_control", "query_only"}:
            errors.append("H1_no_pdb_context")
    elif grounding in {"G1", "G1+G2"}:
        if need is not True:
            errors.append("H1_pdb_need_missing")
        if present is True:
            if decisive not in selected or row["expected_memory_behavior"] != "use":
                errors.append("H1_present_decisive_not_selected")
            if decisive in by_id and by_id[decisive].get("record_subject_id") != row["person_id"]:
                errors.append("H1_decisive_wrong_person")
            if row["context_condition"] != "gt_present":
                errors.append("H1_present_context_mismatch")
        elif present is False:
            acute_escalation = row["visible_hazard_carveout"] is True and row["expected_model_action"] == "safety_escalation"
            if decisive is not None or row["expected_memory_behavior"] != ("escalate_on_visible_hazard" if acute_escalation else "clarify_or_abstain"):
                errors.append("H1_absent_mapping")
            if row["context_condition"] not in {"gt_absent", "distractor_only", "query_only"}:
                errors.append("H1_absent_context_mismatch")
            if row["missing_cue"] == "none" or row["expected_model_action"] not in ({"clarify", "abstain", "safety_escalation"} if row["visible_hazard_carveout"] else {"clarify", "abstain"}):
                errors.append("H1_cued_miss_requires_withhold_action")
        else:
            errors.append("H1_present_not_bool")
    if row["relevant_record_present"] is not True and present is True:
        errors.append("H1_decisive_but_irrelevant")
    if row["decisive_ranks"] != ([selected.index(decisive) + 1] if decisive in selected else []):
        errors.append("decisive_ranks_mismatch")
    if any(not isinstance(rank, int) or rank < 1 or rank > 5 for rank in row["gt_ranks"]):
        errors.append("gt_ranks_invalid")

    knowledge = row["general_knowledge_required"]
    category = row["general_knowledge_source_category"]
    if grounding in {"G1+G2", "general_rule_only"}:
        if knowledge is not True or not isinstance(category, str) or category not in {"parametric_task", "parametric_safety"}:
            errors.append("G2_parametric_knowledge_required")
        evidence = row["general_knowledge_evidence"]
        if (row["general_knowledge_evidence_status"] != "verified" or
            not isinstance(evidence, dict) or not evidence.get("rule_statement") or
            evidence.get("independent_reviewer_id") in (None, "", "pending_review") or
            evidence.get("requires_retrieved_kdb") is not False):
            errors.append("G2_evidence_unverified")
    elif knowledge is not False or category != "none":
        errors.append("G1_or_G0_general_knowledge_mismatch")

    risk = row["physical_safety_or_health_risk"]
    if not isinstance(risk, str) or risk not in {"true", "false", "uncertain"}:
        errors.append("option_b_risk_invalid")
    elif row["risk_label"] != {"true": "unsafe", "false": "safe", "uncertain": "uncertain"}[risk]:
        errors.append("option_b_risk_label_mismatch")
    active = row["active_reward_components"]
    if not isinstance(active, list) or any(not isinstance(name, str) for name in active):
        errors.append("active_reward_components_invalid")
        active = []
    if any(type(row[key]) is not bool for key in ("risk_reward_masked", "risk_supervision_valid", "action_reward_masked", "action_supervision_valid")):
        errors.append("D8_mask_or_validity_not_boolean")
    if row["risk_reward_masked"] is not (not row["risk_supervision_valid"] or
                                          (present is False and row["risk_material_dependence_verified"] is True)):
        errors.append("DF07_variant_mask_mismatch")
    if row["risk_reward_masked"]:
        if not row["risk_mask_reason"] or not isinstance(row["remaining_risk_evidence"], dict) or not isinstance(row["active_reward_components"], list):
            errors.append("DF07_mask_audit_missing")
        elif row["risk_mask_reason"] != ("omitted_personal_risk_acute_visual_remains" if row["visible_hazard_carveout"] else
                                         "omitted_decisive_record_material_to_risk"):
            errors.append("DF07_mask_reason_variant_mismatch")
        elif row["remaining_risk_evidence"].get("validated_general") is True:
            errors.append("DF07_independent_general_risk_evidence_cannot_mask")
        if "risk" in active:
            errors.append("DF07_masked_risk_component_active")
    elif row["risk_mask_reason"] not in (None, "none"):
        errors.append("DF07_unmasked_has_reason")
    elif row["risk_supervision_valid"] is True and "risk" not in active:
        errors.append("DF07_valid_risk_component_missing")
    if row["action_reward_masked"] is not (row["action_supervision_valid"] is not True):
        errors.append("D8_action_mask_mismatch")
    if not isinstance(row["b7_action_mapping"], dict) or row["b7_action_mapping"].get("model_tag") != row["expected_model_action"]:
        errors.append("B7_action_mapping_invalid")
    else:
        subtype = row["b7_action_mapping"].get("subtype")
        if subtype == "A7":
            if not (need and present is False and row["missing_cue"] != "none" and
                    row["expected_model_action"] in {"clarify", "abstain"}):
                errors.append("B7_withhold_subtype_invalid")
        elif not isinstance(subtype, str) or subtype not in B7_TAG or B7_TAG[subtype] != row["expected_model_action"] or B7_RISK[subtype] != risk:
            errors.append("B7_subtype_risk_or_tag_invalid")
        expected_priority = ("acute_incident_or_injury_escalation" if risk == "true" and row["expected_model_action"] == "safety_escalation" else
                             "hazard_prevention_or_mitigation" if risk == "true" else
                             "clarification_under_uncertain_risk" if risk == "uncertain" else
                             "normal_personalized_action" if row["personal_constraint_conflict"] in ("present", "possible") or grounding in ("G1", "G1+G2") else
                             "not_safety_relevant")
        if row["safety_priority"] != expected_priority:
            errors.append("B7_safety_priority_mismatch")
    if not isinstance(row["expected_model_action"], str) or row["expected_model_action"] not in {"direct_answer", "clarify", "abstain", "safety_escalation"}:
        errors.append("B7_unsupported_terminal_action")
    acute = row["visual_acute_evidence"]
    if (not isinstance(acute, dict) or type(acute.get("present")) is not bool or
        not isinstance(acute.get("evidence"), str) or
        (acute.get("present") and not acute.get("evidence", "").strip()) or
        row["visible_hazard_carveout"] is not acute.get("present")):
        errors.append("GB1_visual_acute_evidence_invalid")
    if row["visible_hazard_carveout"] is True and (
        row["physical_safety_or_health_risk"] != "true" or row["expected_model_action"] != "safety_escalation"
    ):
        errors.append("GB1_acute_hazard_requires_escalation")
    if type(row["critical_miss_reward_masked"]) is not bool or row["critical_miss_reward_masked"] is not (row["risk_reward_masked"] and not row["visible_hazard_carveout"]):
        errors.append("GB1_critical_miss_mask_mismatch")
    if row["critical_miss_mask_reason"] != ("risk_undeterminable_without_personal_memory" if row["critical_miss_reward_masked"] else "none"):
        errors.append("GB1_critical_miss_reason_mismatch")
    if row["critical_miss_reward_masked"] and "critical_miss" in active:
        errors.append("GB1_masked_critical_miss_active")
    if row["visible_hazard_carveout"] and "critical_miss" not in active:
        errors.append("GB1_visible_acute_critical_miss_inactive")
    if row["risk_reward_masked"] and row["visible_hazard_carveout"] and row["remaining_risk_evidence"].get("visual") is not True:
        errors.append("GB1_acute_visual_evidence_not_retained")
    if row["renderer_version"] != "train_interface_v1.exact5_render_v1":
        errors.append("GB2_renderer_version_incompatible")
    if not isinstance(row["scenario_group"], str) or row["scenario_group"] not in SCENARIO_GROUPS:
        errors.append("scenario_group_invalid")
    if row["scenario_group"] != "food_storage" and row["storage_discard"] is not None:
        errors.append("storage_discard_not_applicable")
    if row["scenario_group"] == "food_storage" and type(row["storage_discard"]) is not bool:
        errors.append("storage_discard_required")
    if not isinstance(row["wrong_person_records"], list):
        errors.append("wrong_person_records_invalid")
    else:
        expected_wrong = [rid for rid in selected if rid in by_id and
                          by_id[rid].get("record_subject_id") != row["person_id"] and
                          by_id[rid].get("applicability_relation") == "none"]
        observed_wrong = [r.get("record_id") for r in row["wrong_person_records"] if isinstance(r, dict)]
        if observed_wrong != expected_wrong or len(observed_wrong) != len(row["wrong_person_records"]):
            errors.append("wrong_person_records_not_derived_from_visible_context")
        for item in row["wrong_person_records"]:
            if not isinstance(item, dict):
                continue
            rid = item.get("record_id")
            source = by_id.get(rid, {})
            rank_valid = rid in selected and item.get("record_rank") == selected.index(rid) + 1
            if (item.get("record_subject") != source.get("record_subject_id") or
                item.get("query_target") != row["person_id"] or
                item.get("applicability_relation") != "none" or
                not rank_valid):
                errors.append("wrong_person_record_binding_invalid")
            for key in ("target_application_patterns", "source_attribution_patterns"):
                pats = item.get(key)
                if not isinstance(pats, list) or (key == "target_application_patterns" and not pats):
                    errors.append("wrong_person_patterns_invalid")
                    continue
                try:
                    if any(not isinstance(p, str) or not p.strip() or len(p) > 180 for p in pats):
                        errors.append("wrong_person_patterns_invalid")
                    else:
                        for p in pats:
                            re.compile(p)
                except re.error:
                    errors.append("wrong_person_patterns_invalid")
    fact = row["decisive_fact"]
    if need is True:
        if not isinstance(fact, dict):
            errors.append("decisive_fact_required")
        else:
            source = by_id.get(fact.get("source_record_id"))
            if not source or source.get("record_subject_id") != row["person_id"] or any(
                fact.get(k) != source.get(k) for k in ("fact_type", "fact_predicate", "fact_value_or_polarity", "same_fact_group_id")
            ):
                errors.append("decisive_fact_source_mismatch")
            phrases = fact.get("keyphrases")
            if not isinstance(phrases, list) or not 1 <= len(phrases) <= 4 or any(
                not isinstance(p, str) or not p.strip() or len(p.split()) > 8 or not source or p not in source.get("text", "") for p in phrases
            ):
                errors.append("decisive_fact_keyphrases_invalid")
            app = fact.get("application_contract")
            if not isinstance(app, dict) or app.get("version") != "m1.v1" or any(
                not isinstance(app.get(k), list) or (k == "support_patterns" and not app.get(k)) or
                any(not isinstance(p, str) or not p.strip() or p == ".*" for p in app.get(k, []))
                for k in ("support_patterns", "contrary_patterns", "wrong_person_patterns", "hallucination_patterns", "condition_name_patterns")
            ):
                errors.append("decisive_fact_application_contract_invalid")
            else:
                try:
                    for patterns in (v for k, v in app.items() if k.endswith("_patterns")):
                        for pattern in patterns:
                            re.compile(pattern)
                except re.error:
                    errors.append("decisive_fact_application_regex_invalid")
        targets = row["no_call_branch_targets"]
        if not isinstance(targets, dict) or set(targets) != NO_CALL_TARGET_KEYS or any(v is None for v in targets.values()):
            errors.append("no_call_branch_targets_invalid")
        elif (targets["expected_memory_behavior"], targets["expected_action"]) not in (
            {("escalate_on_visible_hazard", "safety_escalation")} if row["visible_hazard_carveout"] else
            {("clarify_or_abstain", "clarify"), ("clarify_or_abstain", "abstain")}
        ):
            errors.append("no_call_branch_target_not_withhold")
        elif row["visible_hazard_carveout"] and (
            targets["risk_label"] != "unsafe" or
            targets["safety_priority"] != "acute_incident_or_injury_escalation"
        ):
            errors.append("no_call_acute_hazard_target_invalid")
    elif fact is not None or row["no_call_branch_targets"] is not None:
        errors.append("no_memory_reward_fields_must_be_null")
    return sorted(set(errors))


def check_family(rows: list[dict[str, Any]]) -> list[str]:
    """Check pair semantics and invariant risk labels across variants."""
    errors: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("label_group_id"):
            groups[row["label_group_id"]].append(row)
    for group_id, group in groups.items():
        if len({(r.get("risk_label"), r.get("physical_safety_or_health_risk"), r.get("label_policy_version")) for r in group}) != 1:
            errors.append(f"{group_id}:risk_label_not_invariant")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
    for pair_id, group in pairs.items():
        if len(group) != 2 or len({r.get("pair_type") for r in group}) != 1:
            errors.append(f"{pair_id}:pair_shape_invalid")
            continue
        a, b = group
        if a["image_sha256"] != b["image_sha256"]:
            errors.append(f"{pair_id}:image_changed")
        if a["pair_type"] == "memory_call":
            if a["query"] == b["query"] or {a["memory_required_for_query"], b["memory_required_for_query"]} != {True, False}:
                errors.append(f"{pair_id}:memory_need_not_changed_by_query")
            if a["store_id"] != b["store_id"]:
                errors.append(f"{pair_id}:store_changed")
        elif a["pair_type"] == "evidence_handling":
            if a["query"] != b["query"] or {a["decisive_record_present"], b["decisive_record_present"]} != {True, False}:
                errors.append(f"{pair_id}:evidence_only_difference_invalid")
        else:
            errors.append(f"{pair_id}:unknown_pair_type")
    return sorted(set(errors))


def check_cross_split(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for key in SPLIT_KEYS:
        seen: dict[str, str] = {}
        for row in rows:
            value, split = row.get(key), row.get("split")
            if value in seen and seen[value] != split:
                errors.append(f"cross_split:{key}:{value}")
            else:
                seen[value] = split
    return sorted(set(errors))
