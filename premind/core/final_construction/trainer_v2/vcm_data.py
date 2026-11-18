"""VCM data adapter: rendered VCM rows (train/dev) -> multi-image training items for both arms, images lazily loaded.

Sources (holdout paths and non-train/dev rows are refused):
  * ``s6`` rendered rows (schema ``s2v9_reward_v5_row_v1``; ``rendered_rows[_<mode>].json``) - the release path;
  * canary ``s4`` rows (schema ``s2v9_vcm_row_v1``, e.g. canary_v9/s4/rows_siglip2.json) - SMOKE ONLY: rendered in
    memory with the same wp_v9_vcm s6_render functions (no file is written; no gate acceptance exists yet).
Context modes (owner training-context policy, s4_rows.context_for_mode): ``retrieved`` | ``gt_retrieved`` |
``gt_retrieved_drop`` (p, seeded per family); labels are re-derived by the s4 rule, excluded arms are dropped.
Prompts are built by ``s6_render.render_arm`` (the inference renderer): passive = one turn with K memory blocks +
New Image + query (K + 1 = 9 images); memory call = Stage 1 (query image + query + Stage-1 instruction), Stage-2 env
turn = K memory blocks + final instruction (called) or the no-memory notice (not called). ``order="rank"`` renders
the memories in retrieval-rank order (ablation); reward labels follow through the display-order map.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import paths_v2 as P
from . import reward_v5 as R5

logger = logging.getLogger(__name__)
ARMS = ("passive", "memory_call")
CONTEXT_MODES = ("retrieved", "gt_retrieved", "gt_retrieved_drop")
ORDERS = ("display", "rank")
SCHEMA_RENDERED, SCHEMA_CANARY = "s2v9_reward_v5_row_v1", "s2v9_vcm_row_v1"
# render_version -> s6_render profile (WP-T1). Canary s4 rows carry no render_version: canary profile.
RENDER_PROFILES = {"s2v9_vcm_render_v1": "canary", "s2v9_vcm_render_v2": "prod"}
CANARY_S4_RENDER = "s2v9_vcm_render_v1"
ROW_FIELDS = ("row_id", "family_id", "split", "row_type", "query", "image_path", "image_sha256", "risk_label",
              "safety_priority", "acute_visual_hazard", "general_knowledge_rule", "grounding_requirement",
              "evidence_pair_id", "memory_call_pair_id", "n_targets", "target_kinds", "stratum")


@dataclass(frozen=True)
class VcmItem:
    """One training prompt. ``messages`` = passive prompt or Stage-1 prompt (image parts hold file paths)."""
    row_id: str
    arm: str
    messages: list
    env_call: Optional[list]          # memory-call Stage-2 user content when call_pdb (K memories + instruction)
    env_no_call: Optional[list]       # memory-call Stage-2 user content otherwise
    image_paths: tuple[str, ...]      # every image of the episode in sequence order (prompt first, then env)
    target: dict                      # reward_v5.vcm_target(...)
    row: dict = field(default_factory=dict)   # evaluator-side fields (hints, gate); never rendered
    arm_fields: dict = field(default_factory=dict)


def _s6():
    from final_construction.wp_v9_vcm import common as cm
    from final_construction.wp_v9_vcm import s6_render as s6
    return cm, s6


def image_parts(content: list) -> list[str]:
    return [p["image"] for p in content if p.get("type") == "image"]


def entries_from_blocks(blocks: list, display_keys: list[str]) -> list[dict]:
    """Recover (display_index, key, image, text) from rendered memory blocks (header, image, text) x K."""
    out = []
    for j in range(len(display_keys)):
        head, img, txt = blocks[3 * j:3 * j + 3]
        if head.get("text") != f"===== Memory {j + 1} =====\n" or img.get("type") != "image":
            raise ValueError("memory blocks are not in the s6_render layout")
        out.append({"display_index": j + 1, "key": display_keys[j], "image": img["image"], "text": txt["text"]})
    return out


def reorder(entries: list[dict], display_to_rank: dict, order: str) -> list[dict]:
    if order == "display":
        return entries
    if order != "rank":
        raise ValueError(f"order must be one of {ORDERS}")
    ranked = sorted(entries, key=lambda e: display_to_rank[str(e["display_index"])])
    return [{**e, "display_index": j + 1} for j, e in enumerate(ranked)]


def render_profile(render_version: str) -> str:
    if render_version not in RENDER_PROFILES:
        raise ValueError(f"unknown render_version {render_version!r}; known {sorted(RENDER_PROFILES)}")
    return RENDER_PROFILES[render_version]


def concept_name(text: str) -> Optional[str]:
    """Stored name / handle of a memory entry: the s2 memory texts open with "<Name>: the user's ..."."""
    head = text.split(":", 1)[0].strip()
    return head if ":" in text and 0 < len(head) <= 40 and len(head.split()) <= 4 else None


def build_items(row: dict, arm_fields: dict, entries: list[dict], *, arms: tuple[str, ...], order: str,
                stored_render: Optional[dict] = None, render_version: str = CANARY_S4_RENDER) -> list[VcmItem]:
    """Items for one row arm, rendered with the row's own profile (render_version). With the stored order, the
    re-render must equal the stored render byte for byte."""
    _, s6 = _s6()
    shown = reorder(entries, arm_fields.get("display_to_rank") or {}, order)
    rendered = s6.render_arm(row, shown, render_profile(render_version))
    if stored_render is not None and order == "display":
        a = json.dumps({k: stored_render[k] for k in rendered}, sort_keys=True)
        if a != json.dumps(rendered, sort_keys=True):
            raise AssertionError(f"{row['row_id']}: re-render differs from the stored inference render")
    target = R5.vcm_target(row, arm_fields, rendered_keys=[e["key"] for e in shown],
                           concept_names=[concept_name(e["text"]) for e in shown])
    keep = {k: row.get(k) for k in ROW_FIELDS} | {"condition": arm_fields.get("condition"),
                                                    "render_version": render_version}
    out = []
    for arm in arms:
        if arm == "passive":
            msgs = rendered["passive"]
            imgs = tuple(image_parts(msgs[1]["content"]))
            out.append(VcmItem(row["row_id"], arm, msgs, None, None, imgs, target, keep, arm_fields))
        elif arm == "memory_call":
            msgs = rendered["memory_call_stage1"]
            imgs = tuple(image_parts(msgs[1]["content"]) + image_parts(rendered["memory_call_stage2_call"]))
            out.append(VcmItem(row["row_id"], arm, msgs, rendered["memory_call_stage2_call"],
                               rendered["memory_call_stage2_no_call"], imgs, target, keep, arm_fields))
        else:
            raise ValueError(f"arm must be one of {ARMS}")
    return out


def _check_split(row: dict) -> bool:
    split = str(row.get("split"))
    if "holdout" in split.lower() or split in ("eval", "test"):
        raise P.HoldoutRefusedError(f"{row.get('row_id')}: split {split!r} refused")
    return split in P.OPEN_SPLITS


def _canary_rows(rows: list[dict], retriever: str, mode: str, p: float, cfg_out: Path) -> list[tuple]:
    """Canary s4 rows -> (row, arm_fields, entries, None) through the s6_render functions (in memory)."""
    cm, s6 = _s6()
    cfg = cm.Cfg(out=cfg_out)
    plan = {f["family_id"]: f for f in cm.load_plan(cfg)["families"]}
    cache: dict = {}
    out = []
    for r in rows:
        if not r.get("deterministic_pass"):
            out.append((r, {"excluded": "deterministic_pass_false"}, [], None))
            continue
        fam = plan[r["family_id"]]
        try:
            r = {**r, "arms": s6.mode_arms(cfg, fam, r, mode, p, cache)}
        except Exception as exc:  # noqa: BLE001 - live upstream state (e.g. family_inputs StopIteration): skip row
            logger.warning("%s: s6_render.mode_arms(%s) failed: %r", r["row_id"], mode, exc)
            out.append((r, {"excluded": f"mode_arms_failed:{type(exc).__name__}"}, [], None))
            continue
        lab = r["arms"][retriever]
        if lab.get("excluded"):
            out.append((r, {"excluded": lab["excluded"]}, [], None))
            continue
        texts = {e["key"]: e["text"] for e in r["entries"]}
        if any(k not in texts for k in lab.get("display_keys") or []):
            texts.update(s6._all_texts(cfg, fam))
        images = {k: c["entry"] for k, c in fam["concepts"].items()}
        out.append((r, s6.reward_fields(r, retriever), s6.arm_entries(r, retriever, texts, images), None))
    return out


def _rendered_rows(rows: list[dict], retriever: str, mode: str, require_accepted: bool) -> list[tuple]:
    out = []
    for r in rows:
        if r.get("context_mode") != mode:
            raise ValueError(f"{r['row_id']}: file context_mode {r.get('context_mode')!r} != requested {mode!r}")
        a = r["arms"][retriever]
        if require_accepted and not r.get("accepted"):
            out.append((r, {"excluded": "not_accepted"}, [], None))
        elif a.get("excluded") or a.get("i7_leaks"):
            out.append((r, {"excluded": a.get("excluded") or "i7_leaks"}, [], None))
        else:
            fields = {k: v for k, v in a.items() if k not in ("render", "i7_leaks")}
            ents = entries_from_blocks(a["render"]["memory_call_stage2_call"], a["display_keys"])
            out.append((r, fields, ents, a["render"]))
    return out


def load_items(path: Path, *, arms: tuple[str, ...] = ARMS, mode: str = "retrieved", drop_p: float = 0.2,
               retriever: Optional[str] = None, order: str = "display", require_accepted: bool = True,
               limit: Optional[int] = None, splits: tuple[str, ...] = P.OPEN_SPLITS) -> tuple[list[VcmItem], dict]:
    """Load train/dev VCM rows into items (``splits`` narrows, e.g. ("train",) for training). Returns (items,
    manifest with exclusion counts)."""
    if not set(splits) <= set(P.OPEN_SPLITS):
        raise P.HoldoutRefusedError(f"splits {splits} outside {P.OPEN_SPLITS}")
    if mode not in CONTEXT_MODES:
        raise ValueError(f"mode must be one of {CONTEXT_MODES}")
    P.refuse_holdout_path(path)
    data = json.loads(Path(path).read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    schema = {r.get("schema_version") for r in rows}
    if len(schema) != 1:
        raise ValueError(f"mixed or missing schema_version: {schema}")
    schema = schema.pop()
    versions = {r.get("render_version") or CANARY_S4_RENDER for r in rows} if schema == SCHEMA_RENDERED \
        else {CANARY_S4_RENDER}
    if len(versions) != 1:
        raise ValueError(f"mixed render_version in one file: {sorted(map(str, versions))}")
    render_version = versions.pop()
    render_profile(render_version)
    open_rows = [r for r in rows if _check_split(r) and r["split"] in splits]
    cm, _ = _s6()
    retriever = retriever or cm.MAIN_RETRIEVER
    if schema == SCHEMA_CANARY:
        triples = _canary_rows(open_rows, retriever, mode, drop_p, Path(path).resolve().parents[1])
    elif schema == SCHEMA_RENDERED:
        triples = _rendered_rows(open_rows, retriever, mode, require_accepted)
    else:
        raise ValueError(f"unsupported schema_version {schema!r}")
    items, excluded = [], {}
    for row, fields, entries, stored in triples:
        if fields.get("excluded"):
            excluded[fields["excluded"]] = excluded.get(fields["excluded"], 0) + 1
            continue
        try:
            items += build_items(row, fields, entries, arms=arms, order=order, stored_render=stored,
                                 render_version=render_version)
        except R5.RewardInputError as exc:
            excluded[f"reward_input:{exc}"] = excluded.get(f"reward_input:{exc}", 0) + 1
    if limit is not None:
        items = items[:limit * len(arms)]
    manifest = {"path": str(path), "sha256": P.sha256_file(Path(path)), "schema": schema, "render_version": render_version, "mode": mode,
                "drop_p": drop_p if mode == "gt_retrieved_drop" else None, "splits": list(splits), "retriever": retriever, "order": order,
                "n_rows": len(rows), "n_open_rows": len(open_rows), "n_items": len(items), "excluded": excluded,
                "smoke_only": schema == SCHEMA_CANARY, "images_per_prompt": sorted({len(i.image_paths) for i in items})}
    logger.info("vcm_data: %s", json.dumps({k: v for k, v in manifest.items() if k != "sha256"}))
    return items, manifest


# ----------------------------------------------------------------------------- lazy images
def load_image(path: str, expected_sha256: Optional[str] = None) -> Any:
    """Same loader as the VCM eval runners (EXIF-upright RGB, no resize); optional sha256 check (query image)."""
    import io

    from PIL import Image, ImageOps
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"image_sha256 mismatch: {path}")
    with Image.open(io.BytesIO(data)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def item_images(item: VcmItem) -> dict[str, Any]:
    """path -> PIL for every image of the episode (query image verified against the row sha256)."""
    q = item.row["image_path"]
    return {p: load_image(p, item.row["image_sha256"] if p == q else None) for p in dict.fromkeys(item.image_paths)}


def lazy_dataset(items: list[VcmItem], loader=item_images):
    """HF Dataset of row ids only; images are attached per accessed batch by ``set_transform`` (never eager PIL
    columns: those overflowed Arrow's 2 GB offsets in the trainer_v1 C2 run)."""
    from datasets import Dataset
    by_key = {f"{it.arm}|{it.row_id}": it for it in items}
    ds = Dataset.from_list([{"key": f"{it.arm}|{it.row_id}", "row_id": it.row_id} for it in items])
    ds.set_transform(lambda b: {**b, "images": [loader(by_key[k]) for k in b["key"]]})
    return ds
