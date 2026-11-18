"""Read the immutable VCM release JSONL into trainer_v2's canonical VcmItem."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from eval_v1 import topk_context as TK
from trainer_v2 import paths_v2 as P
from trainer_v2 import vcm_data as VD
from trainer_v2 import vef as VF
from trainer_v2 import vef2 as VF2

MODES = VD.CONTEXT_MODES


def row_id_filter() -> tuple[set[str], dict] | None:
    """EVAL_ROW_IDS_FILE (regen_v1 partial re-eval, 2026-10-12): newline-separated release sample_ids; only these
    rows are converted (unset = every row, unchanged behaviour). The file sha256 enters the run manifest."""
    f = os.environ.get("EVAL_ROW_IDS_FILE")
    if not f:
        return None
    data = Path(f).read_bytes()
    ids = {x.strip() for x in data.decode("utf-8").splitlines() if x.strip()}
    if not ids:
        raise ValueError(f"EVAL_ROW_IDS_FILE is empty: {f}")
    return ids, {"file": str(Path(f).resolve()), "sha256": hashlib.sha256(data).hexdigest(), "n_ids": len(ids)}


def release_file(release: Path, split: str, mode: str, *, final_holdout: bool = False) -> Path:
    release = Path(release)
    if not final_holdout:
        P.refuse_holdout_path(release)
        P.refuse_holdout_path(split)
    if split not in ("train", "dev") and not (final_holdout and split == "holdout"):
        raise ValueError(f"unsupported split: {split}")
    if mode not in MODES:
        raise ValueError(f"unsupported render mode: {mode}")
    path = release / ("" if mode == "retrieved" else mode) / f"{split}.jsonl"
    if not final_holdout:
        P.refuse_holdout_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolved_image(root: Path, name: str, *, final_holdout: bool = False) -> str:
    path = (root / name).resolve()
    if not final_holdout:
        P.refuse_holdout_path(path)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"image escapes release root: {name}")
    return str(path)


def convert(row: dict, root: Path, arms: tuple[str, ...], *, final_holdout: bool = False,
            vef: bool = False) -> list[VD.VcmItem]:
    """``vef``: also copy the fields the VEF QA pool reads (trainer_v2/vef.source_row) into ``row["vef_src"]``
    (VEF reward presets only; default False leaves every item unchanged). ``vef_src["match"]`` = the memory names /
    handles / roles the VEF v2 Match parser reads (trainer_v2/vef2.match_src; v1's row_pool ignores it)."""
    ev = row["evaluator"]
    if isinstance(ev, str):
        ev = json.loads(ev)
    if not ev.get("accepted", False):
        return []
    if ev.get("expected_memory_behavior") is None:
        return []  # This release row has no reward_v5 target; count explicitly in the manifest.
    concepts = ev["context_concepts"]
    if len(row["contexts"]) != len(row["context_texts"]) or len(concepts) != len(row["contexts"]):
        raise ValueError(f"{row['sample_id']}: context lengths differ")
    keys = [c["key"] for c in concepts]
    entries = [{"display_index": j + 1, "key": key, "image": _resolved_image(root, img, final_holdout=final_holdout), "text": txt}
               for j, (key, img, txt) in enumerate(zip(keys, row["contexts"], row["context_texts"], strict=True))]
    arm = {"excluded": None, "display_keys": keys, "display_to_rank": ev.get("display_to_rank") or {},
           "condition": row["context_condition"], "expected_model_action": row["gold_action"],
           "expected_memory_behavior": ev["expected_memory_behavior"], "gold_action_set": row["gold_action_set"],
           "gold_citation_set": row["gold_citations"], "support_patterns_by_display_index":
           ev["support_patterns_by_display_index"], "wrong_concept_patterns": ev["wrong_concept_patterns"],
           "decisive_keyphrases": ev.get("decisive_keyphrases") or [], "b7_subtype": row.get("b7_subtype"),
           "gold_answer": row.get("gold_answer"), "gold_answer_reference_only": row.get("reference_only", False)}
    canonical = {"row_id": row["sample_id"], "family_id": row["family_id"], "split": row["split"],
                 "row_type": row["row_type"], "query": row["query_text"],
                 "image_path": _resolved_image(root, row["query"], final_holdout=final_holdout), "image_sha256": ev["source_image_sha256"],
                 "risk_label": row["risk"], "safety_priority": ev["safety_priority"],
                 "acute_visual_hazard": ev["acute_visual_hazard"],
                 "general_knowledge_rule": ev.get("general_knowledge_rule"),
                 "grounding_requirement": row["grounding_requirement"],
                 "evidence_pair_id": row.get("evidence_pair_id"), "memory_call_pair_id": row.get("pair_id"),
                 "n_targets": row.get("n_concepts"), "target_kinds": None, "stratum": ev.get("stratum"),
                 "query_style": row.get("query_style"), "request_form": row.get("request_form"),
                 "rule_id": ev.get("rule_id"),
                 "risk": row["risk"], "condition": row["context_condition"],
                 "render_mode": row["render_mode"]}
    items = VD.build_items(canonical, arm, entries, arms=arms, order="display",
                           render_version="s2v9_vcm_render_v2")
    # build_items retains only ROW_FIELDS; attach release-only analysis fields without changing its prompts or target.
    return [VD.VcmItem(it.row_id, it.arm, it.messages, it.env_call, it.env_no_call, it.image_paths,
                       it.target, {**it.row, **{k: canonical[k] for k in
                       ("rule_id", "risk", "render_mode", "query_style", "request_form")},
                       **({"vef_src": {**VF.source_row(row), "match": VF2.match_src(row)}} if vef else {})},
                      it.arm_fields) for it in items]


def check_pending(row: dict) -> None:
    """v4 b-lite / memdiv rows (v4_build/blite_overlay.py, v4_build/memdiv.py) are evaluable only after their retrieval
    recompute; rows without such a block (every older release) are unaffected."""
    for tag in ("blite_v4", "memdiv_v4", "acute_add_v4c", "collapsed_v1", "acute_v5"):  # acute_add_v4c: orchestrator 2026-10-14
        b = row.get(tag)
        if isinstance(b, dict) and "retrieval_siglip2" in (b.get("pending_recompute") or []):
            raise ValueError(f"{row['sample_id']}: {tag} row pending retrieval recompute (v4_build)")


def mix_k(sample_id: str, ks: tuple[int, ...], seed: int) -> int:
    """Deterministic per-row K for the K-mix training ablation."""
    h = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16)
    return ks[h % len(ks)]


def load_release(release: Path, split: str, mode: str, *, arms: tuple[str, ...] = VD.ARMS,
                 limit: int | None = None, final_holdout: bool = False,
                 context_k: int = TK.FULL_K, context_k_mix: tuple[int, ...] = (),
                 mix_seed: int = 0, vef: bool = False, online_k: Optional[int] = None,
                 online_query_emb: Optional[Path] = None) -> tuple[list[VD.VcmItem], dict]:
    """``online_k`` (v4, owner 2026-10-13; default None = unchanged): online SigLIP2 retrieval at K in {1,2,4,8}
    (eval_v1/online_retrieval.py; retrieved mode only; equivalence gate vs the stored top-K, retrieve_relabel labels,
    per-row ``absent_case``); ``online_query_emb`` = fresh query-embedding sidecar dir (None: frozen s0 query views).
    ``context_k`` < 8: top-K context ablation (eval_v1/topk_context.py; retrieved mode only).
    ``context_k_mix`` (training only, owner 2026-10-06 K-mix ablation): each row gets K drawn from the tuple by a
    seeded hash of its sample_id and is truncated with the same top-K rule (any mode; retriever ranks are stored)."""
    path = release_file(release, split, mode, final_holdout=final_holdout)
    TK.check_k(context_k)
    for k in context_k_mix:
        TK.check_k(k)
    topk = context_k != TK.FULL_K or bool(context_k_mix)
    online = online_k is not None
    if online:
        TK.check_k(online_k)
        if mode != "retrieved" or topk:
            raise ValueError("online_k is defined for the retrieved mode only and excludes context_k / context_k_mix")
        from eval_v1 import online_retrieval as OR
        retr = OR.OnlineRetriever(online_query_emb, release=release)
        partners = OR.partner_index([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()])
    online_rows: dict[str, dict] = {}
    if context_k != TK.FULL_K and mode != "retrieved":
        raise ValueError("context_k < 8 is defined for the retrieved mode only")
    sibs = TK.sibling_index([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]) \
        if topk else {}
    topk_rows: dict[str, dict] = {}
    if not set(arms) <= set(VD.ARMS) or not arms:
        raise ValueError(f"invalid arms: {arms}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root = Path(release).resolve().parent
    items: list[VD.VcmItem] = []
    digest = hashlib.sha256()
    n_rows = 0
    excluded: dict[str, int] = {}
    rf = row_id_filter()
    n_matched = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split or row.get("render_mode") != mode:
                raise ValueError(f"{path}: split/mode mismatch at line {n_rows + 1}")
            n_rows += 1
            if rf is not None:
                if row["sample_id"] not in rf[0]:
                    continue
                n_matched += 1
            check_pending(row)          # after the explicit row filter: a filtered-out pending row is never evaluated
            if limit is None or len(items) < limit * len(arms):
                info = None
                ev0 = TK.evaluator(row)
                if topk and ev0.get("accepted") and ev0.get("expected_memory_behavior") is not None:
                    k_row = mix_k(row["sample_id"], context_k_mix, mix_seed) if context_k_mix else context_k
                    row_k, info = TK.truncate(row, k_row, sibs)
                    topk_rows[row["sample_id"]] = info
                    if row_k is None:
                        reason = "topk:" + str(info["topk_exclusion"])
                        excluded[reason] = excluded.get(reason, 0) + 1
                        continue
                    row = row_k
                elif online and ev0.get("accepted") and ev0.get("expected_memory_behavior") is not None:
                    row_k, info = OR.online_row(row, online_k, retr, partners)
                    online_rows[row["sample_id"]] = info
                    if row_k is None:
                        reason = "online:" + str(info["topk_exclusion"])
                        excluded[reason] = excluded.get(reason, 0) + 1
                        continue
                    row = row_k
                converted = convert(row, root, arms, final_holdout=final_holdout, vef=vef)
                if info is not None:
                    converted = [VD.VcmItem(it.row_id, it.arm, it.messages, it.env_call, it.env_no_call,
                                            it.image_paths, it.target, {**it.row, **info}, it.arm_fields)
                                 for it in converted]
                if not converted:
                    ev = row["evaluator"]
                    if isinstance(ev, str):
                        ev = json.loads(ev)
                    reason = ("missing_reward_target:" + str(row.get("context_condition"))
                              if ev.get("accepted") else "not_accepted")
                    excluded[reason] = excluded.get(reason, 0) + 1
                items.extend(converted)
    manifest = {"path": str(path), "sha256": digest.hexdigest(), "n_rows": n_rows,
                "n_items": len(items), "split": split, "mode": mode, "arms": list(arms),
                "excluded": excluded}
    if rf is not None:
        manifest["row_filter"] = {**rf[1], "n_matched": n_matched}
    if online:
        summ = OR.summarize(online_rows)
        manifest["online"] = {"online_k": online_k, **retr.info, **summ, "rows": online_rows}
        if summ["n_online_stored_mismatch_topk"]:
            raise RuntimeError(f"REFUSED: online top-{online_k} differs from the stored top-K for "
                               f"{summ['n_online_stored_mismatch_topk']} rows (v4_build/online/equivalence_check.py)")
    if topk:
        manifest["topk"] = {"context_k": context_k, "context_k_mix": list(context_k_mix), "mix_seed": mix_seed,
                            "rows": topk_rows,
                            "n_relabelled": sum(i["relabelled"] for i in topk_rows.values()),
                            "n_excluded": sum(i["topk_excluded"] for i in topk_rows.values()),
                            "n_target_retained": sum(i["target_retained"] is True for i in topk_rows.values())}
    return items, manifest
