"""Paid GPT (gpt-6-sol) calls for the S2 v9 canary: multi-image strict-JSON client + the paid-call guard.

The client mirrors personalization_tool_v2/canary/common_v1.gpt (same model/effort default, same usage-ledger line
format, strict json_schema) but sends several labelled images. The guard runs before EVERY paid call:
  * PTV2_USAGE_LOG == this canary's ledger; PREREG present and every pinned file sha256 unchanged;
  * the exact prelaunch line (s2v9 `prelaunch` phase; plan + PREREG hashes) is in COLLABORATION.md;
  * ledger tokens + PER_CALL_RESERVE <= cap (BudgetBlocked otherwise); a consumed tag is never re-paid.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Callable

from final_construction.wp_v9_vcm import common as cm

GPT_MODEL, GPT_EFFORT = "gpt-6-sol", "low"
GptFn = Callable[..., dict]


def gpt_multi(instructions: str, payload: dict, schema: dict, name: str, tag: str, images: list[tuple[str, str]],
              usage_log: Path, max_out: int = 12000, effort: str = GPT_EFFORT) -> dict:  # pragma: no cover - paid
    """One logged strict-JSON call; images = [(label, path)] sent in order, each preceded by its label."""
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    content = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    for label, path in images:
        url = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        content += [{"type": "input_text", "text": label}, {"type": "input_image", "image_url": url, "detail": "high"}]
    r = OpenAI(max_retries=2, timeout=900.0).responses.create(
        model=GPT_MODEL, reasoning={"effort": effort}, store=False, instructions=instructions,
        input=[{"role": "user", "content": content}], max_output_tokens=max_out,
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}})
    u = getattr(r, "usage", None)
    usage = u.model_dump() if hasattr(u, "model_dump") else {}
    with open(usage_log, "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": GPT_MODEL,
                             "effort": effort, "input_tokens": usage.get("input_tokens"),
                             "output_tokens": usage.get("output_tokens"),
                             "total_tokens": usage.get("total_tokens")}) + "\n")
    return {"output": json.loads(r.output_text), "usage": usage, "model": getattr(r, "model", GPT_MODEL)}


def ledger(usage_log: Path) -> tuple[int, set[str]]:
    if not Path(usage_log).exists():
        return 0, set()
    rows = [json.loads(x) for x in Path(usage_log).read_text().splitlines() if x.strip()]
    if any(not isinstance(r.get("total_tokens"), int) for r in rows):
        raise RuntimeError("usage ledger has unknown token spend; stop")
    return sum(r["total_tokens"] for r in rows), {r["tag"] for r in rows}


def pin_files() -> dict[str, Path]:
    files = {p.name: p for p in sorted(cm.HERE.glob("*.py"))}
    files.update({"SPEC_S2V9.md": cm.SPEC, "generate_dialogue.py": cm.DIALOGUE_SRC, "gpt_cap_gen.py": cm.CAP_GEN_SRC,
                  "final_prompts.py": cm.FC / "train_interface_v1/final_prompts.py",
                  "render.py": cm.FC / "train_interface_v1/render.py",
                  "audit_data.py": cm.AUDIT_DIR / "audit_data.py", "encoders.py": cm.AUDIT_DIR / "encoders.py",
                  "detect_crops.py": cm.AUDIT_DIR / "detect_crops.py", "audit_crops.json": cm.AUDIT_CROPS,
                  "retriever_audit_PREREGISTRATION.md": cm.AUDIT_DIR / "PREREGISTRATION.md",
                  "open_image_to_refs.json": cm.REFS, "qwen3_vl_8b_config.json": cm.QWEN8B / "config.json"})
    return files


def pins() -> dict[str, str]:
    return {k: cm.sha_file(p) for k, p in sorted(pin_files().items()) if p.exists()}


def prelaunch_line(cfg: cm.Cfg) -> str:
    plan = cfg.p("s1", "plan.json")
    if not plan.exists() or not cfg.prereg.exists():
        raise RuntimeError("plan.json and PREREG are required before the prelaunch line exists")
    fams = [f["family_id"] for f in cm.read(plan)["families"]]
    if cfg.profile == "holdout":                              # sealed holdout: one line for every holdout chunk
        lv3 = cfg.p("s1", "concept_labels_v3.json")
        return (f"REGISTER S2V9 VCM SEALED HOLDOUT (round 1 + round-7 and round-10 additive rows; counts-and-hashes output; chunks of {cm.PROD_CHUNK}; "
                f"GPT review = query image + memory texts only (review_mode gpt_query_img_texts), Qwen all images; "
                f"no {','.join(cm.DROPPED_ROW_TYPES)}; checks S1,S2,S2b,S2c,S3,S4,S5,S6,S7,S8; round {cm.EXTRA_ROUND} "
                f"+ {cm.EXTRA2_ROUND} additive rows + label-independent generic queries (r11); "
                f"budget stop: projected > {cm.HOLDOUT_BUDGET_STOP} tokens) — cap={cfg.cap} tokens (separate ledger) — "
                f"run_id={cfg.run_id} — families={len(fams)} — source_aggregate_sha256={cm.HOLDOUT_AGGREGATE_V1} — "
                f"plan_sha256={cm.sha_file(plan)} — labels_v3_sha256={cm.sha_file(lv3) if lv3.exists() else 'MISSING'}"
                f" — prereg_sha256={cm.sha_file(cfg.prereg)}")
    if cfg.profile == "prod" and (cfg.chunk or cfg.regen):    # one line for every chunk and regen run
        from final_construction.wp_v9_vcm import s7_retry
        rv2, lv3 = cfg.p("s1", "risk_plan_v2.json"), cfg.p("s1", "concept_labels_v3.json")
        rpo = cfg.p("s1", "regen_pilot_overlay.json")
        n = len(fams) - cm.PROD_PILOT_N
        return (f"REGISTER S2V9 VCM PRODUCTION RUN (cost-cut author/delta; v8a-style review; chunks of "
                f"{cm.PROD_CHUNK}; per-chunk budget rule: projected total > {cm.PROD_BUDGET_STOP} tokens -> stop, > "
                f"{cm.PROD_BUDGET_RECOMMEND} -> no review round; any row type < {cm.PROD_COVERAGE_STOP} unsafe -> stop) — "
                f"cap={cfg.cap} tokens incl. pilot — run_id={cfg.run_id} — "
                f"families={n} of {len(fams)} (after pilot {cm.PROD_PILOT_N}) — plan_sha256={cm.sha_file(plan)} — "
                f"risk_plan_v2_sha256={cm.sha_file(rv2) if rv2.exists() else 'MISSING'} — "
                f"labels_v3_sha256={cm.sha_file(lv3) if lv3.exists() else 'MISSING'} — "
                f"regen=pilot+chunk01-03 (round 1, @g2) regen_pilot_overlay_sha256="
                f"{cm.sha_file(rpo) if rpo.exists() else 'MISSING'} — retry=one review-driven revise round per final "
                f"unit (regen pilot/1-3 + chunks {cm.RETRY_FIRST_CHUNK}+; round 2, @r2, one delta attempt; s5_final_r2: "
                f"per family round 2 iff accepted_r2 >= accepted_r1; policy {s7_retry.policy(cfg)}) — repair=round "
                f"{cm.FIX_ROUND} acute/G0 fix + pass 2 round {cm.FIX_ROUNDS[1]} for its unaccepted rows (checks "
                f"S1,S2,S2b,S3,S4,S5,S6; labels normalized in code; TEXT-ONLY GPT: items-only delta, <= 2 attempts "
                f"(2nd only for item-level text failures), copy without a call when labels suffice, text check + text "
                f"adjudication, Qwen second review with images; affected rows only; per row: pass 2 > pass 1 > "
                f"dropped (s5_final_r3, s5_final_r4)) — extra=round {cm.EXTRA_ROUND} additive rows (safe generic G0 + "
                f"G1_panic_memory; text-only GPT, Qwen review with images; S6 exclusivity) + r8 dataset-level "
                f"label-independent generic queries (S2d) — r10=extended safe generic G0 rows + r11 dataset-wide "
                f"re-assignment (query-only must not predict; G0 coverage on generic-query G0) — removed_row_types="
                f"{','.join(cm.DROPPED_ROW_TYPES)} — prereg_sha256={cm.sha_file(cfg.prereg)}")
    if cfg.profile == "prod":
        return (f"REGISTER S2V9 VCM PRODUCTION PILOT (scene-aware risk crossing; verified first-pass author + delta "
                f"retries; v8a-style review; stop rule {cm.PROD_TOKENS_PER_FAMILY_MAX} tokens/family) — "
                f"cap={cfg.cap} tokens — run_id={cfg.run_id} — pilot={cfg.pilot} of {len(fams)} families — "
                f"plan_sha256={cm.sha_file(plan)} — prereg_sha256={cm.sha_file(cfg.prereg)}")
    return (f"REGISTER S2V9 VCM CANARY (concept-anchored; retrieval-aware stores; GPT author + v8a-style review) — "
            f"cap={cfg.cap} tokens — run_id={cfg.run_id} — families={len(fams)} — "
            f"plan_sha256={cm.sha_file(plan)} — prereg_sha256={cm.sha_file(cfg.prereg)}")


def check_pins(cfg: cm.Cfg) -> None:
    prereg = cm.read(cfg.prereg)
    now = pins()
    bad = [k for k, v in prereg["pins_sha256"].items() if now.get(k) != v]
    if bad:
        raise RuntimeError(f"pinned files changed since PREREG: {bad}")


IN_FLIGHT_STAGES = frozenset({"author_delta", "revise", "memtext_family", "review_primary", "adjudicate", "extra_text",
                              "revise_review", "salvage_rewrite", "selfcheck", "review_second", "describe"})


def open_stops(cfg: cm.Cfg) -> list:
    """Stop files (s1/*STOP*.json: PILOT_STOP, PROD_STOP_chunk<KK>) without a matching owner override
    (<stem>_OVERRIDE.json bound to the stop file's sha)."""
    d = cfg.p("s1")
    out = []
    for stop in sorted(d.glob("*STOP*.json")) if d.exists() else []:
        if stop.stem.endswith("_OVERRIDE"):
            continue
        ov = stop.with_name(stop.stem + "_OVERRIDE.json")
        if not (ov.exists() and cm.read(ov).get("stop_sha256") == cm.sha_file(stop)):
            out.append(stop)
    return out


def guard(cfg: cm.Cfg, tag: str) -> None:
    """Raises unless this paid call is authorised; BudgetBlocked when the cap would be exceeded."""
    stage = tag.split("::")[1] if tag.count("::") >= 2 else None
    for stop in open_stops(cfg):           # stop scope (2026-10-03): a stop blocks NEW runs (their author stage, or a
        if stage in IN_FLIGHT_STAGES:      # tag without a stage); runs already past author finish their chunk
            break
        raise RuntimeError(f"stop rule fired (s1/{stop.name}): no new run may start "
                           "(owner override: run_v9.py --resume-after-stop REASON --owner-instruction TEXT)")
    kind = "HOLDOUT" if cfg.profile == "holdout" else "PROD"
    if cfg.chunk and cfg.chunk > 1 and not cfg.p("s1", f"{kind}_CHECK_chunk{cfg.chunk - 1:02d}.json").exists():
        raise RuntimeError(f"chunk {cfg.chunk}: run `{kind.lower()}_check --chunk {cfg.chunk - 1}` first")
    if cfg.mock:
        return
    if os.environ.get("PTV2_USAGE_LOG") != str(cfg.usage_log):
        raise RuntimeError("PTV2_USAGE_LOG must point to the S2 v9 canary ledger")
    if not cfg.prereg.exists():
        raise RuntimeError("PREREG_S2V9_CANARY.json missing")
    check_pins(cfg)
    if prelaunch_line(cfg) not in cfg.collab.read_text():
        raise RuntimeError("owner-approved exact S2 v9 canary prelaunch line absent from COLLABORATION.md")
    spent, tags = ledger(cfg.usage_log)
    if tag in tags:
        raise RuntimeError(f"consumed paid tag cannot be reused: {tag}")
    if spent + cm.PER_CALL_RESERVE > cfg.cap:
        raise cm.BudgetBlocked(f"BUDGET_BLOCKED spent={spent} reserve={cm.PER_CALL_RESERVE} cap={cfg.cap}")


def call(cfg: cm.Cfg, gpt_fn: GptFn | None, instructions: str, payload: dict, schema: dict, name: str, tag: str,
         images: list[tuple[str, str]], max_out: int, effort: str = GPT_EFFORT) -> dict | None:
    """Guarded call. Every paid response is saved write-once (s7/responses/, resume safety 2026-10-03: a step that
    aborts mid-batch loses nothing); a consumed tag returns its saved response and is never re-paid; a consumed tag
    WITHOUT a saved response (paid before this fix) is recorded in s7/lost_paid_responses/ and returns None."""
    _, tags = ledger(cfg.usage_log)
    saved = cfg.p("s7", "responses", cm.safe_name(tag.split("::", 1)[-1]) + ".json")
    if tag in tags:
        if saved.exists():
            return cm.read(saved)["output"]
        lost = cfg.p("s7", "lost_paid_responses", cm.safe_name(tag.split("::", 1)[-1]) + ".json")
        if not lost.exists():
            cm.write_once(lost, {"tag": tag, "note": "paid, response not saved (pre-fix run); never re-paid"})
        return None
    guard(cfg, tag)
    fn = gpt_fn or gpt_multi
    try:
        try:
            out = fn(instructions, payload, schema, name, tag, images, cfg.usage_log, max_out=max_out,
                     effort=effort)["output"]
        except Exception as exc:  # noqa: BLE001 - cost cut: a rejected lower effort falls back (no spend, no ledger)
            if effort == GPT_EFFORT or "effort" not in str(exc).lower():
                raise
            out = fn(instructions, payload, schema, name, tag, images, cfg.usage_log, max_out=max_out,
                     effort=GPT_EFFORT)["output"]
    except json.JSONDecodeError:
        return None
    if not saved.exists():
        cm.write_once(saved, {"tag": tag, "output": out})
    return out
