"""personalization_tool_v2 canary v1 — shared paths, write-once IO, logged GPT calls, source-image selection.

See PERSONALIZATION_TOOL_V2_CANARY_PLAN.md. Every paid call is appended to canary/usage.jsonl with its tokens.
"""
from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import time
from pathlib import Path

CAN = Path(__file__).resolve().parent
EXT = CAN.parent
V3 = EXT.parent
INV = EXT / "inventory"
GPT_MODEL, GPT_EFFORT = "gpt-6-sol", "low"
QWEN = (str(Path.home()) + "/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
SPLIT_MAP = {"ext_train": "train", "ext_dev": "dev", "ext_heldout_diagnostic": "heldout_diag"}
FAMILY_MAP = {"F1_identified_ownership": "F1_ownership", "F2_temporal_update": "F2_temporal",
              "F3_same_person_long_memory": "F3_long_memory", "F4_cross_person_routing": "F4_routing",
              "F5_kdb_rule_diverse": "F5_kdb_grounding", "F6_tool_required_curriculum": "F6_tool_curriculum"}
QUERY_DATE = "2026-09-30"
CUE_TOPICS = {"food": ["food_safety"], "pet": ["animal", "food_safety"], "medication": ["medication"],
              "chemical": ["toxic_chemical"], "electrical": ["electrical"], "stairs_fall": ["fall_slip"],
              "rug": ["fall_slip"], "stove_hot": ["fire_heat"], "heat": ["fire_heat"], "sharp": ["sharp"],
              "child": ["child_hazard"]}
GENERIC_RULE_PREFIX = "kdb-W5-F033"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(p: Path, obj) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == data, f"immutable output differs: {p}"
    else:
        p.write_text(data)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p)]


def gpt(instructions: str, payload: dict, schema: dict, name: str, tag: str, image_path: str | None = None,
        max_out: int = 8000, effort: str = GPT_EFFORT) -> dict:
    """One logged GPT-6-Sol-low call with a strict JSON schema (and optionally the image)."""
    from openai import OpenAI
    assert os.environ.get("OPENAI_API_KEY")
    content = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    if image_path:
        url = "data:image/png;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    r = OpenAI(max_retries=2, timeout=600.0).responses.create(
        model=GPT_MODEL, reasoning={"effort": effort}, store=False, instructions=instructions,
        input=[{"role": "user", "content": content}], max_output_tokens=max_out,
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}})
    u = getattr(r, "usage", None)
    usage = u.model_dump() if hasattr(u, "model_dump") else {}
    # per-package usage log (co-working agents set PTV2_USAGE_LOG so budgets never mix); default = canary/usage.jsonl
    with open(Path(os.environ.get("PTV2_USAGE_LOG", str(CAN / "usage.jsonl"))), "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": GPT_MODEL, "effort": effort,
                             "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                             "total_tokens": usage.get("total_tokens")}) + "\n")
    return {"output": json.loads(r.output_text), "usage": usage, "model": getattr(r, "model", GPT_MODEL)}


def inventory() -> dict[str, dict]:
    return {r["image_sha256"]: r for r in jl(INV / "image_inventory_v1.jsonl")}


def image_queue() -> list[dict]:
    """Deterministic image slots: the inventory's selected images first, then fallbacks per (family, split group)
    in sha order (strong family support, no prior eval-diagnostic use, never reused across families)."""
    inv, cand = inventory(), json.loads((INV / "canary_source_candidates_v1.json").read_text())
    slots = []
    for fam, parts in cand["families"].items():
        for part, pv in parts.items():
            if not isinstance(pv, dict) or "selected" not in pv:
                continue
            group = ("ext_train", "ext_dev") if part == "train_dev" else ("ext_heldout_diagnostic",)
            fallbacks = sorted((r for r in inv.values() if r["proposed_ext_split"] in group
                                and fam in (r.get("family_support_strong") or {}) and not r["prior_eval_diagnostic_use"]),
                               key=lambda r: r["image_sha256"])
            for s in pv["selected"]:
                slots.append({"family": FAMILY_MAP[fam], "inv_family": fam, "part": part,
                              "primary": s["image_sha256"], "roles": s.get("roles", []),
                              "fallbacks": [r["image_sha256"] for r in fallbacks]})
    return slots


def image_record(sha256: str) -> dict:
    r = inventory()[sha256]
    return {"sha256": r["image_sha256"], "path": r["path"], "donor_scene": r["donor_scene"],
            "main_split": "eval" if r["main_split"] == "eval" else "train", "ext_split": SPLIT_MAP[r["proposed_ext_split"]],
            "caption_excerpt": r["caption_excerpt"], "cue_tags": r["cue_tags"], "strong_cues": r["strong_cues"]}


def kdb_rules(heldout: bool) -> dict[str, str]:
    """Train/dev: the train corpus only. Held-out: active English v2 rules. The generic aisle rule is excluded."""
    if not heldout:
        rules = {r["doc_id"]: r["text"] for r in jl(V3 / "rl_sweep/train_tool_corpus_v1/kdb.jsonl")}
    else:
        rules = {}
        for f in sorted(glob.glob(str(V3.parent / "benchmark_v2/data/knowledge_db/text_only/*.jsonl"))):
            for line in open(f):
                try:
                    x = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(x, dict) and x.get("status") == "active" and x.get("language") == "en" and x.get("record_id"):
                    rules[x["record_id"]] = x["text"]
    return {k: v for k, v in rules.items() if not k.startswith(GENERIC_RULE_PREFIX)}


def candidate_rules(cues: list[str], heldout: bool, k: int = 12) -> list[dict]:
    topics = json.loads((EXT / "audit/summary_v1.json").read_text())["rule_topics"]
    want = {t for c in cues for t in CUE_TOPICS.get(c, [])}
    rules = kdb_rules(heldout)
    hits = [rid for rid in sorted(rules) if want & set(topics.get(rid, []))]
    return [{"kdb_id": rid, "text": rules[rid]} for rid in hits[:k]]
