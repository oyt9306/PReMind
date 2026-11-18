"""Final-row -> training-item adapter (PROVISIONAL; FIELD_MAP placeholders until WP-G freezes the schema).

Produces, from one admitted final row:
  (a) a passive-RAG item: image + query + exact-five PDB context;
  (b) a memory-call two-turn episode: Stage 1 = image + query only; Stage 2 shows the
      exact-five context only after a parsed ``call_pdb``.
Model-visible content and evaluator-only content are returned as separate objects, and every
visible text passes ``leak_check``. Any missing/invalid field excludes the row (DF-12); nothing
is defaulted.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
for _p in (HERE, V3):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import render as RD  # noqa: E402

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "train_interface_v1.adapter_v1"
# owner 2026-10-02: seed of the label-independent record display order (project SEED, shortcut_audit_v1/stats.py).
RECORD_ORDER_SEED = 20261001
# logical name -> final-schema field name. Names mirror the WP-G draft `contract_v1.REQUIRED`
# where one exists; the last five are interface placeholders WP-G has not defined yet.
FIELD_MAP: dict[str, str] = {
    "schema_version": "schema_version", "row_id": "row_id", "split": "split", "family_id": "family_id",
    "image_path": "image_path", "image_sha256": "image_sha256", "query": "query",
    "grounding": "grounding_requirement", "memory_required": "memory_required_for_query",
    "decisive_present": "decisive_record_present", "decisive_record_id": "decisive_record_id",
    "context_condition": "context_condition", "expected_memory_behavior": "expected_memory_behavior",
    "missing_cue": "missing_cue", "risk_label": "risk_label", "safety_priority": "safety_priority",
    "expected_action": "expected_model_action", "b7_action_mapping": "b7_action_mapping",
    "risk_reward_masked": "risk_reward_masked", "action_reward_masked": "action_reward_masked",
    "risk_mask_policy_version": "risk_mask_policy_version", "risk_mask_reason": "risk_mask_reason",
    "active_reward_components": "active_reward_components", "label_policy_version": "label_policy_version",
    "adjudication_status": "adjudication_status", "records": "records",
    "selected_record_ids": "selected_record_ids", "decisive_ranks": "decisive_ranks", "gt_ranks": "gt_ranks",
    "retriever_version": "retriever_version", "render_k": "render_k",
    # --- placeholders (not yet in WP-G contract_v1) ---
    "scenario": "scenario_group", "storage_discard": "storage_discard",
    "decisive_fact": "decisive_fact", "wrong_person_records": "wrong_person_records",
    "no_call_branch_targets": "no_call_branch_targets",
    # GB-1 (owner ruling, Addendum 2, 2026-10-01): acute-hazard visual-evidence flag for the
    # critical_miss carve-out on a risk-masked row. Named to match analysis_v1's
    # gold.visible_hazard_carveout (schema.py), the same underlying WP-G field.
    "visual_acute_evidence": "visible_hazard_carveout",
}
# Fields whose value may legitimately be null (presence of the key is still required).
NULLABLE = {"decisive_record_id", "storage_discard", "decisive_fact", "no_call_branch_targets", "risk_mask_reason"}
MODEL_VISIBLE = {"query", "image_path"}  # plus the `text` of the five selected records
RECORD_VISIBLE_KEYS = {"text"}
NO_CALL_TARGET_KEYS = {"risk_label", "safety_priority", "expected_action", "expected_memory_behavior",
                       "risk_reward_masked", "action_reward_masked"}
EXCLUDED_GROUNDING = {"G1-T": "scope_excluded:G1-T(D-5)", "future_kdb": "scope_excluded:future_kdb(D-2)",
                      "G2_explicit_kdb": "scope_excluded:future_kdb(D-2)",
                      "future_temporal": "scope_excluded:G1-T(D-5)"}
EXPECTED_RISK_MASK_POLICY = "df07_variant_material_dependence_v1"
# owner 2026-10-02: Stage 1 may contain brief reasoning (512-token budget). The decision is the single
# <memory_call> tag anywhere in the reply (models may wrap it, e.g. inside <answer>); zero tags, an invalid value or
# two or more tags -> decision_invalid.
STAGE1_TAG = re.compile(r"<memory_call>\s*([a-z_]+)\s*</memory_call>")
STAGE1_RE = STAGE1_TAG  # kept name for importers
# Identifier/label-like evaluator values (ids, enums, versions, hashes, paths): no spaces, and
# containing one of `_ : / .` or >= 16 hex chars. Plain words are not scanned (false positives).
_IDENTIFIER = re.compile(r"(?=.{6,})(?:[^\s\\]*[_:/.][^\s\\]*|[0-9a-f]{16,})")
_MARKERS = ("filler", "is_gt", "gt_rank", "decisive", "missing_cue", "provenance", "risk_mask", "label_policy")


class ExclusionError(ValueError):
    """Row is ineligible; ``reason`` goes to the exclusion manifest."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LeakError(ExclusionError):
    """An evaluator-only value or marker reached model-visible text."""


UPSTREAM_SCHEMA = "final_benchmark_contract_v1"
WP_G_CONTRACT = V3 / "final_construction" / "wp_g_final_benchmark_v1" / "contract_v1.py"
# sha256 of the WP-G contract_v1.py this interface was verified against (2026-10-01 22:01 version).
PINNED_CONTRACT_SHA256 = "811f7c68de06f35a88ada4d1905b22b5f3125774b4cf3d2803779159c0b7f91c"
Validator = Callable[[dict], list]


def contract_sha256(path: Path = WP_G_CONTRACT) -> str:
    """sha256 of the live WP-G contract file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wp_g_validator() -> tuple[Validator, str]:
    """Import the live WP-G ``check_row``. Raises (never skips) when it cannot be imported (DF-12)."""
    try:
        from final_construction.wp_g_final_benchmark_v1 import contract_v1
    except ImportError as exc:
        raise RuntimeError(f"WP-G contract_v1 not importable; refusing to admit rows: {exc}") from exc
    return contract_v1.check_row, contract_sha256()


class Adapter:
    """Admission with an injected upstream validator; the validator's identity is recorded."""

    def __init__(self, *, validator: Validator, contract_sha256: str) -> None:
        if not callable(validator) or not isinstance(contract_sha256, str) or not contract_sha256:
            raise ValueError("an upstream validator and its contract sha256 are required")
        self.validator, self.contract_sha256 = validator, contract_sha256

    @classmethod
    def live(cls, *, require_pinned: bool = True) -> "Adapter":
        """Adapter bound to the live WP-G contract; refuses an unpinned contract unless told otherwise."""
        validator, sha = load_wp_g_validator()
        if require_pinned and sha != PINNED_CONTRACT_SHA256:
            raise RuntimeError(f"WP-G contract sha256 {sha[:12]} != pinned {PINNED_CONTRACT_SHA256[:12]}; re-verify")
        return cls(validator=validator, contract_sha256=sha)

    def canonicalize(self, raw: dict) -> dict:
        """Validate a final row and return it under logical names. Raises ExclusionError (fail closed)."""
        if not isinstance(raw, dict):
            raise ExclusionError("row_not_object")
        grounding = raw.get(FIELD_MAP["grounding"])
        if grounding in EXCLUDED_GROUNDING:
            raise ExclusionError(EXCLUDED_GROUNDING[grounding])
        row: dict[str, Any] = {}
        for logical, src in FIELD_MAP.items():
            if src not in raw:
                raise ExclusionError(f"missing_field:{src}")
            if raw[src] is None and logical not in NULLABLE:
                raise ExclusionError(f"null_field:{src}")
            row[logical] = raw[src]
        if row["schema_version"] != UPSTREAM_SCHEMA:
            raise ExclusionError(f"schema_version_unsupported:{row['schema_version']}")
        errors = self.validator(raw)
        if not isinstance(errors, list):
            raise ExclusionError("upstream_validator_returned_non_list")
        if errors:
            raise ExclusionError("upstream_contract:" + ",".join(map(str, errors)))
        _check_interface_fields(row)
        return row

    def admit(self, raw_rows: list[dict], arm: str) -> tuple[list[TrainItem], list[dict]]:
        """Admit rows for one arm. Returns (items, exclusion manifest entries with the contract sha256)."""
        if arm not in {"passive_rag", "memory_call"}:
            raise ValueError(f"unknown arm {arm!r}")
        from reward_bridge import validate_reward_inputs  # local import: avoids a cycle

        items, excluded = [], []
        for raw in raw_rows:
            row_id = raw.get("row_id") if isinstance(raw, dict) else None
            try:
                row = self.canonicalize(raw)
                validate_reward_inputs(row)
                items.append(passive_item(row) if arm == "passive_rag" else stage1_item(row))
            except ExclusionError as exc:
                excluded.append({"row_id": row_id, "arm": arm, "reason": exc.reason,
                                 "contract_sha256": self.contract_sha256})
        return items, excluded


def _check_interface_fields(row: dict) -> None:
    if row["render_k"] != RD.RENDER_K:
        raise ExclusionError("render_k_not_five")
    if row["risk_mask_policy_version"] != EXPECTED_RISK_MASK_POLICY:
        raise ExclusionError("risk_mask_policy_version_incompatible")
    if type(row["memory_required"]) is not bool:
        raise ExclusionError("memory_required_not_bool")
    if type(row["visual_acute_evidence"]) is not bool:
        raise ExclusionError("visual_acute_evidence_not_bool")
    by_id = {r["record_id"]: r for r in row["records"]}
    for rec in row["records"]:
        if rec.get("filler_status") is not False:
            raise ExclusionError("filler_record_in_store")
    if len(row["records"]) < RD.RENDER_K:
        raise ExclusionError("store_under_five_real_records")
    selected = row["selected_record_ids"]
    if not isinstance(selected, list) or len(selected) != RD.RENDER_K or len(set(selected)) != RD.RENDER_K:
        raise ExclusionError("selected_not_exactly_five_unique")
    if any(rid not in by_id for rid in selected):
        raise ExclusionError("selected_record_not_in_store")
    active = row["active_reward_components"]
    if not isinstance(active, list):
        raise ExclusionError("active_reward_components_invalid")
    if row["risk_reward_masked"] is ("risk" in active):
        raise ExclusionError("risk_mask_vs_active_components_contradiction")
    if row["action_reward_masked"] is ("action" in active):
        raise ExclusionError("action_mask_vs_active_components_contradiction")
    if not isinstance(row["scenario"], str) or not row["scenario"]:
        raise ExclusionError("scenario_invalid")
    if row["storage_discard"] is not None and type(row["storage_discard"]) is not bool:
        raise ExclusionError("storage_discard_invalid")
    if not isinstance(row["wrong_person_records"], list):
        raise ExclusionError("wrong_person_records_invalid")
    if row["memory_required"]:
        if not isinstance(row["decisive_fact"], dict):
            raise ExclusionError("decisive_fact_required_when_memory_required")
        targets = row["no_call_branch_targets"]
        if not isinstance(targets, dict) or set(targets) != NO_CALL_TARGET_KEYS:
            raise ExclusionError("no_call_branch_targets_required_when_memory_required")
        if any(targets[k] is None for k in NO_CALL_TARGET_KEYS):
            raise ExclusionError("no_call_branch_targets_partial")
    elif row["no_call_branch_targets"] is not None:
        raise ExclusionError("no_call_branch_targets_must_be_null_when_not_required")


def record_order(row_id: str, ids: Sequence) -> list:
    """Label-independent display order of ``ids`` (owner 2026-10-02, wave-1 audit: decisive rank vs labels).

    Deterministic Fisher-Yates permutation driven only by
    ``sha256(f"{RECORD_ORDER_SEED}:record_order:{row_id}")``, so it depends on the row id and the
    number of ids, never on labels. Pure; ``ids`` is not mutated.
    """
    x = int.from_bytes(hashlib.sha256(f"{RECORD_ORDER_SEED}:record_order:{row_id}".encode()).digest(), "big")
    out = list(ids)
    for i in range(len(out) - 1, 0, -1):
        x, j = divmod(x, i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def context_entries(row: dict) -> list[dict]:
    """Exact-five entries in ``record_order`` display order.

    ``rank`` is the display position (1..5, what the model sees and cites); the evaluator-only
    ``retriever_rank`` is the position in ``selected_record_ids`` (used by the evaluation-only
    top-k renderer and to map gt/decisive ranks to display ranks); ``is_filler`` is evaluator-only.
    """
    by_id = {r["record_id"]: r for r in row["records"]}
    retriever = {rid: i for i, rid in enumerate(row["selected_record_ids"], 1)}
    return [{"rank": pos, "retriever_rank": retriever[rid], "text": by_id[rid]["text"],
             "is_filler": by_id[rid]["filler_status"]}
            for pos, rid in enumerate(record_order(row["row_id"], row["selected_record_ids"]), 1)]


def display_rank_map(row: dict) -> dict[int, int]:
    """Retriever rank (position in ``selected_record_ids``) -> displayed rank under ``record_order``."""
    return {e["retriever_rank"]: e["rank"] for e in context_entries(row)}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}


def _text_user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _system(text: str) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text}]}


@dataclass(frozen=True)
class TrainItem:
    """Model-visible prompt (``visible``) kept apart from evaluator-only data (``row``)."""
    row_id: str
    arm: str
    visible: dict
    row: dict


def visible_texts(messages: list[dict]) -> list[str]:
    """All text strings in a chat message list."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(content)
            continue
        out += [part["text"] for part in content if part.get("type") == "text"]
    return out


def passive_item(row: dict) -> TrainItem:
    """(a) Passive-RAG training row: image + query + exact-five context."""
    block = RD.render_context(context_entries(row))
    messages = [_system(RD.SYSTEM_PASSIVE), _user(RD.passive_user_text(row["query"], block))]
    visible = {"prompt": messages, "image_path": row["image_path"]}
    leak_check(visible_texts(messages), row, context_shown=True)
    return TrainItem(row["row_id"], "passive_rag", visible, row)


def stage1_item(row: dict) -> TrainItem:
    """(b) Memory-call Stage 1: image + query only; no memory of any kind."""
    messages = [_system(RD.SYSTEM_MEMORY_CALL), _user(RD.stage1_user_text(row["query"]))]
    visible = {"prompt": messages, "image_path": row["image_path"]}
    leak_check(visible_texts(messages), row, context_shown=False)
    return TrainItem(row["row_id"], "memory_call", visible, row)


def parse_stage1(text: str) -> str | None:
    """Stage-1 parser: optional reasoning, then exactly one final <memory_call> tag; else ``decision_invalid``."""
    tags = STAGE1_TAG.findall(text) if isinstance(text, str) else []
    return tags[0] if len(tags) == 1 and tags[0] in ("call_pdb", "direct_answer") else None


def stage2_messages(row: dict, stage1_output: str) -> tuple[list[dict], str | None, bool]:
    """Append the environment turn after a Stage-1 output.

    Returns:
        (messages, parsed decision or None, called). An invalid decision continues in the
        not-called branch (fail closed) and is scored as wrong by the reward bridge.
    """
    decision = parse_stage1(stage1_output)
    called = decision == "call_pdb"
    block = RD.render_context(context_entries(row)) if called else None
    messages = stage1_item(row).visible["prompt"] + [
        {"role": "assistant", "content": [{"type": "text", "text": stage1_output}]},
        _text_user(RD.stage2_user_text(block))]
    # Only environment-authored text is checked; the assistant turn is the policy's own output.
    leak_check(visible_texts(messages[:2] + messages[3:]), row, context_shown=called)
    return messages, decision, called


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def leak_check(texts: list[str], row: dict, *, context_shown: bool) -> None:
    """Fail if evaluator-only content reaches environment-authored model-visible text.

    Checks (1) evaluator-only field names and markers, (2) distinctive evaluator-only values
    (length >= 6, not part of the static templates or of the allowed visible content), and
    (3) when ``context_shown`` is False, any store record text or the context header.
    """
    joined = "\n".join(texts)
    shown = [e["text"] for e in context_entries(row)] if context_shown else []
    # Only static templates are exempt: query and record texts are data and are scanned too.
    names = set(FIELD_MAP.values()) - {FIELD_MAP[k] for k in MODEL_VISIBLE}
    names |= {k for r in row["records"] for k in r} - RECORD_VISIBLE_KEYS
    for token in sorted(names | set(_MARKERS)):
        if len(token) >= 6 and token in joined and not any(token in t for t in RD.STATIC_TEMPLATES):
            raise LeakError(f"leak:field_or_marker:{token}")
    evaluator = {k: v for k, v in row.items() if k not in MODEL_VISIBLE and k != "records"}
    evaluator["record_meta"] = [{k: v for k, v in r.items() if k not in RECORD_VISIBLE_KEYS} for r in row["records"]]
    for value in set(_strings(evaluator)):
        if (_IDENTIFIER.fullmatch(value) and value in joined
                and not any(value in t for t in RD.STATIC_TEMPLATES)):
            raise LeakError(f"leak:evaluator_value:{value[:40]}")
    if not context_shown and (RD.header(RD.RENDER_K) in joined or RD.count_rendered_entries(joined)):
        raise LeakError("leak:context_before_call")
    for rec in row["records"]:  # unselected (or, before a call, any) store record must stay hidden
        if rec["text"] not in shown and rec["text"] in joined and rec["text"] not in row["query"]:
            raise LeakError("leak:record_text_not_selected_or_before_call")
