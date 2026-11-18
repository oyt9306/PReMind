"""Stage 6b (CPU): canonical release layout, CoViP style (PMLLM/Benchmark_renew), HF-datasets friendly.

<root>/images/sample_<n>/query.png              the query image the model sees
<root>/images/sample_<n>/context_<j>.png        j = 0..K-1 in DISPLAY order, the preprocessed 448x448 entry images
<root>/<split>.jsonl                            one line per row, RELATIVE paths only
One folder per row (simplest, never shared). Model-facing and label keys are top level; provenance lives under the
single evaluator-only key "evaluator" (retriever id/revision, ranks/scores, concept ids, keyphrases, patterns, checks,
review outcome, legacy-arm labels). Holdout is written the same way under a separate root, by script only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s7_probes

EVALUATOR_KEY = "evaluator"


def _copy_png(src: str, dst: Path) -> None:
    if src.lower().endswith(".png"):
        shutil.copyfile(src, dst)
        return
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        ImageOps.exif_transpose(im).convert("RGB").save(dst, format="PNG")


def export_row(root: Path, n: int, row: dict, fam: dict, checks: dict) -> dict:
    a = row["arms"][row["main_arm"]]
    folder = Path("images") / f"sample_{n}"
    (root / folder).mkdir(parents=True, exist_ok=True)
    _copy_png(row["image_path"], root / folder / "query.png")
    keys = a["display_keys"]
    for j, k in enumerate(keys):
        shutil.copyfile(fam["concepts"][k]["entry"], root / folder / f"context_{j}.png")
    texts = s7_probes.entry_texts(a)
    legacy = {arm: {k: v for k, v in lab.items() if k not in ("render",)} for arm, lab in row["arms"].items()
              if arm != row["main_arm"]}
    return {
        "sample_id": f"sample_{n}", "query": str(folder / "query.png"),
        "contexts": [str(folder / f"context_{j}.png") for j in range(len(keys))], "context_texts": texts,
        "query_text": row["query"], "query_style": row.get("query_style"), "row_type": row["row_type"],
        "grounding_requirement": row["grounding_requirement"],
        "risk": row["risk_label"], "gold_action": a["expected_model_action"], "b7_subtype": a["b7_subtype"],
        "gold_action_set": a.get("gold_action_set") or [a["expected_model_action"]],
        "gold_answer": a["gold_answer"], "reference_only": bool(a.get("gold_answer_reference_only")), "gold_citations": a["gold_citation_set"], "context_condition": a["condition"],
        "pair_id": row["memory_call_pair_id"], "evidence_pair_id": row["evidence_pair_id"],
        "family_id": row["family_id"], "person_id": fam["person_id"], "split": row["split"],
        "n_concepts": row["n_targets"], "k": len(keys), "render_mode": row["context_mode"],
        EVALUATOR_KEY: {
            "row_id": row["row_id"], "accepted": row["accepted"], "stratum": row["stratum"],
            "review_mode": row.get("review_mode", "gpt_images"), "query_style": row.get("query_style"),
            "retriever": a["retrieval"], "display_to_rank": a["display_to_rank"],
            "context_concepts": [{"key": k, "identity_id": fam["concepts"][k]["iid"], "role": fam["concepts"][k]["role"],
                                  "of_target": fam["concepts"][k]["of_target"], "fine_type": fam["concepts"][k]["fine_type"],
                                  "name": fam["concepts"][k]["name"],
                                  "handle": fam["concepts"][k].get("handle", "")} for k in keys],
            "target_keys": fam["targets"], "decisive_keyphrases": a["decisive_keyphrases"],
            "target_identity_ids": [fam["concepts"][t]["iid"] for t in fam["targets"]],    # audit 8.1 (plan)
            "support_patterns_by_display_index": a["support_patterns_by_display_index"],
            "wrong_concept_patterns": a["wrong_concept_patterns"], "absent_reason": a["absent_reason"],
            "expected_memory_behavior": a["expected_memory_behavior"], "safety_priority": row["safety_priority"],
            "acute_visual_hazard": row["acute_visual_hazard"], "general_knowledge_rule": row["general_knowledge_rule"],
            "general_knowledge_rule_id": rule_id(row["general_knowledge_rule"]),
            "degeneration_gate_tokens": row["degeneration_gate_tokens"], "source_image_sha256": row["image_sha256"],
            "failed_checks": sorted(k for k, v in checks.items() if not v["pass"]), "legacy_arms": legacy}}


def rule_id(text: str | None) -> str | None:
    """Stable id of a general-knowledge rule = final_audit_vcm_v1 scheme: vcmrule_<sha256(stripped text)[:12]>."""
    t = (text or "").strip()
    return "vcmrule_" + hashlib.sha256(t.encode()).hexdigest()[:12] if t else None


def split_excluded(rows: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Rows whose MAIN arm is excluded by rule in this rendered mode (e.g. gt_retrieved_drop drops a target:
    partial_retrieval_two_concept / retrieval_miss_a4_not_required) have no defined target, so they are left out
    of this mode's jsonl with the reason; every exported row then has a non-null expected_memory_behavior."""
    keep, out = [], {}
    for r in rows:
        a = r["arms"][r["main_arm"]]
        if a.get("excluded") or a.get("expected_memory_behavior") is None:
            out[r["row_id"]] = a.get("excluded") or "expected_memory_behavior_undefined"
        else:
            keep.append(r)
    return keep, out


def run_export(cfg: cm.Cfg, mode: str = "retrieved", root: Path | None = None) -> dict:
    """Write the release for one rendered context mode (default: the reviewed retrieved contexts)."""
    tag = cm.final_tag(cfg)                            # retry selection finalized -> s6_r2/ -> release_r2/
    src = cfg.p("s6" + tag, "rendered_rows.json" if mode == "retrieved" else f"rendered_rows_{mode}.json")
    root = root or cfg.p("release" + tag, mode)
    if (root / "MANIFEST.json").exists():
        return {"status": "exists", "root": str(root)}
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"release root not empty and has no manifest: {root}")
    plan = {f["family_id"]: f for f in cm.load_plan(cfg)["families"]}
    checks = {r["row_id"]: r["checks"] for r in cm.read(cm.final_rows_path(cfg))["rows"]}
    rows = sorted(cm.read(src)["rows"], key=lambda r: (r["split"], r["row_id"]))
    dropped = [r for r in rows if cm.no_general_rule(cfg) and r["row_type"] in cm.DROPPED_ROW_TYPES]
    rows = [r for r in rows if r not in dropped]                     # general-rule control removed (owner 2026-10-03)
    rows, by_rule = split_excluded(rows)                             # eval_v1 2026-10-03: no null memory labels
    by_split: dict[str, list[dict]] = {}
    for n, r in enumerate(rows):
        cm.assert_open_only(r["image_path"], cfg.profile == "holdout")
        rec = export_row(root, n, r, plan[r["family_id"]], checks[r["row_id"]])
        if cfg.profile == "holdout":                             # identity rule: seen/unseen concept flags
            from final_construction.wp_v9_vcm import h_holdout
            rec[EVALUATOR_KEY]["concept_identity"] = h_holdout.seen_flags(cfg, plan[r["family_id"]])
        by_split.setdefault(r["split"], []).append(rec)
    for split, recs in by_split.items():
        with open(root / f"{split}.jsonl", "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {"layout": "CoViP Benchmark_renew style: images/sample_<n>/{query.png, context_<j>.png} + <split>.jsonl",
                "render_mode": mode, "rows": {s: len(v) for s, v in by_split.items()},
                "jsonl_sha256": {s: cm.sha_file(root / f"{s}.jsonl") for s in by_split},
                "evaluator_only_key": EVALUATOR_KEY, "relative_paths": True,
                "excluded_row_types": {t: {"rows": sum(r["row_type"] == t for r in dropped), "reason": cm.DROP_REASON}
                                       for t in cm.DROPPED_ROW_TYPES} if cm.no_general_rule(cfg) else {},
                "excluded_by_rule": {"rows": len(by_rule), "reasons": dict(Counter(by_rule.values())),
                                     "row_ids": dict(sorted(by_rule.items())),
                                     "rule": "s4_rows.arm_labels on the context of THIS mode: partial retrieval of a "
                                             "two-concept row and a retrieval miss whose fact is not decision-critical "
                                             "are excluded by rule (no target defined)"}}
    cm.write_once(root / "MANIFEST.json", manifest)
    return {"root": str(root), **manifest}


def load_release(root: Path) -> list[dict]:
    return [json.loads(x) for s in sorted(Path(root).glob("*.jsonl")) for x in s.read_text().splitlines() if x.strip()]
