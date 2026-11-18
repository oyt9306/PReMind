"""K-retention (text-memory retention) loader: shelved text-track rows rendered as train_interface_v1 passive items.

EVALUATION ONLY, never training. Rows come from text_track_v2 (release v1.0.1, text-cued G1 / G1+G2, decisive record
present). Each row is rendered exactly like ``train_interface_v1.adapter.passive_item``: SYSTEM_PASSIVE, the user
query and the exact-five context block in the label-independent ``record_order`` display order. Two variants:

* ``image``    -- query image + text (the passive arm as trained);
* ``no_image`` -- the image part is dropped, the text is byte-identical (truly text-only setting).

Exclusions (counted in the manifest, never defaulted):
* ``tt2_patch_ambiguous`` -- rows carrying ``tt2_patch`` (acute-flag patch; owner judged them ambiguous);
* ``reward_input_invalid:<reason>`` -- rows reward_v4.0-passive cannot score (fail closed, DF-12);
* ``render_invalid:<reason>`` -- exact-five render or leak-check failures.

Scoring override (documented in every manifest): the text-track rows are WP-G ``final_benchmark_contract_v1`` rows
still at ``adjudication_status="pending_review"``; reward_v4 admits only adjudicated rows. The scoring view sets it to
``not_required`` and changes nothing else. The holdout file is opened only with ``final_holdout=True``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FC = HERE.parent
if str(FC) not in sys.path:
    sys.path.insert(0, str(FC))
if str(FC / "reward_v4") not in sys.path:
    sys.path.insert(0, str(FC / "reward_v4"))
sys.dont_write_bytecode = True  # never write __pycache__ into the frozen reward / interface / rl_sweep packages

import rewards_v4 as RV4  # noqa: E402  (read-only; pins its own lineage sha256 at import)

from trainer_v2 import paths_v2 as P  # noqa: E402

logger = logging.getLogger(__name__)

AD = RV4.AD
RD = AD.RD
TT = FC / "text_track_v2"
OPEN_FILES = {"train": TT / "out" / "curated_train.json", "dev": TT / "out" / "curated_dev.json"}
HOLDOUT_FILE = TT / "holdout_sealed" / "curated_holdout.json"
VARIANTS = ("image", "no_image")
KRET_DATA_VERSION = "eval_v1.kret_data_v1"
SCORING_OVERRIDE = {"adjudication_status": "not_required"}
EXCLUDE_PATCHED = "tt2_patch_ambiguous"


@dataclass(frozen=True)
class KretItem:
    """Model-visible ``messages`` kept apart from the evaluator-only scoring ``view``."""
    row_id: str
    family_id: str
    variant: str
    messages: list[dict]
    image_path: str | None
    image_sha256: str | None
    view: dict
    meta: dict


def source_file(split: str, *, final_holdout: bool = False) -> Path:
    """Path of a split; the holdout path is refused (before opening) without ``final_holdout``."""
    if split == "holdout":
        if not final_holdout:
            raise P.HoldoutRefusedError("refused: K-retention holdout requires --final-holdout")
        return HOLDOUT_FILE
    if split not in OPEN_FILES:
        raise ValueError(f"unsupported split: {split}")
    path = OPEN_FILES[split]
    if not final_holdout:
        P.refuse_holdout_path(path)
    return path


def scoring_view(raw: dict) -> dict:
    """reward_v4 input view of a raw row (only the documented adjudication override)."""
    return {**raw, **SCORING_OVERRIDE}


def _system(text: str) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def render(raw: dict, variant: str) -> KretItem:
    """Render one row; raises ``RV4.RewardInputError`` / ``AD.ExclusionError`` / ``RD.RenderError`` (fail closed)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    view = scoring_view(raw)
    RV4.reward_passive(view, "")  # fail-closed reward-input check (pure; empty completion)
    row = RV4.canonical_row(view)
    block = RD.render_context(AD.context_entries(row))
    text = RD.passive_user_text(row["query"], block)
    user = [{"type": "text", "text": text}]
    if variant == "image":
        user = [{"type": "image", "image": row["image_path"]}, *user]
    messages = [_system(RD.SYSTEM_PASSIVE), {"role": "user", "content": user}]
    AD.leak_check(AD.visible_texts(messages), row, context_shown=True)
    meta = {"grounding": raw["grounding_requirement"], "risk_label": raw["risk_label"],
            "expected_action": raw["expected_model_action"], "safety_priority": raw["safety_priority"],
            "scenario_group": raw["scenario_group"]}
    return KretItem(raw["row_id"], raw["family_id"], variant, messages,
                    row["image_path"] if variant == "image" else None,
                    raw["image_sha256"] if variant == "image" else None, view, meta)


def exclusion_reason(raw: dict) -> str | None:
    """Row-level exclusion reason, or None. Patched rows are excluded before any rendering."""
    if raw.get("tt2_patch"):
        return EXCLUDE_PATCHED
    try:
        render(raw, "no_image")
    except RV4.RewardInputError as exc:
        return "reward_input_invalid:" + ":".join(exc.reason.split(":")[1:3])
    except (RD.RenderError, AD.ExclusionError) as exc:
        return f"render_invalid:{getattr(exc, 'reason', type(exc).__name__)}"
    return None


def load_rows(split: str, *, final_holdout: bool = False) -> tuple[list[dict], dict]:
    """Raw ``track_rows`` of a split plus file provenance (sha256, row count)."""
    path = source_file(split, final_holdout=final_holdout)
    data = path.read_bytes()
    doc = json.loads(data)
    rows = doc["track_rows"]
    if any(r.get("split") != split for r in rows):
        raise ValueError(f"{path}: split field mismatch")
    return rows, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "n_rows": len(rows),
                  "rule_version": doc.get("rule_version")}


def load_kret(split: str, *, variants: tuple[str, ...] = VARIANTS, limit: int | None = None,
              final_holdout: bool = False) -> tuple[list[KretItem], dict]:
    """K-retention items (row-major, one per variant) and a manifest of counts and exclusions."""
    if not variants or not set(variants) <= set(VARIANTS):
        raise ValueError(f"invalid variants: {variants}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows, prov = load_rows(split, final_holdout=final_holdout)
    items: list[KretItem] = []
    excluded: dict[str, int] = {}
    excluded_ids: dict[str, list[str]] = {}
    n_rows_kept = 0
    for raw in rows:
        reason = exclusion_reason(raw)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            excluded_ids.setdefault(reason, []).append(raw["row_id"])
            continue
        if limit is not None and n_rows_kept >= limit:
            continue
        n_rows_kept += 1
        items.extend(render(raw, v) for v in variants)
    manifest = {**prov, "split": split, "variants": list(variants), "limit": limit,
                "n_rows_kept": n_rows_kept, "n_items": len(items), "excluded": excluded,
                "n_excluded": sum(excluded.values()),
                # holdout: counts only (row ids are not echoed for the sealed split)
                "excluded_row_ids": None if split == "holdout" else excluded_ids,
                "scoring_override": {"adjudication_status": "pending_review->not_required"},
                "renderer_version": RD.RENDERER_VERSION, "prompt_version": RD.PROMPT_VERSION,
                "reward_version": RV4.REWARD_VERSION_PASSIVE, "data_version": KRET_DATA_VERSION,
                "purpose": "evaluation_only_never_training"}
    logger.info("kret %s: rows=%d kept=%d excluded=%s items=%d", split, prov["n_rows"], n_rows_kept,
                excluded, len(items))
    return items, manifest


def item_images(item: KretItem) -> dict[str, Any]:
    """path -> PIL (sha256-verified) for the image variant; empty for ``no_image``."""
    if item.image_path is None:
        return {}
    from trainer_v2 import vcm_data as VD
    return {item.image_path: VD.load_image(item.image_path, item.image_sha256)}


def main(argv: list[str] | None = None) -> int:
    """CPU count report: ``python -m eval_v1.kret_data --split dev``."""
    import argparse
    ap = argparse.ArgumentParser(description="K-retention row counts (CPU)")
    ap.add_argument("--split", default="dev", choices=("train", "dev", "holdout"))
    ap.add_argument("--final-holdout", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    _, man = load_kret(args.split, final_holdout=args.final_holdout)
    out = {k: v for k, v in man.items() if k != "excluded_row_ids"}
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
