"""Round 9 (owner/coordinator 2026-10-03): re-review of the round-7 safe generic G0 rows that the text-only check
rejected because of my reviewer bug (it flagged ordinary scene mentions and had no scene texts). TEXT-ONLY GPT check
with the fixed prompt + scene texts; the local Qwen verdicts of round 7 (same row content, images) are reused by
copy; disagreements go to text-only adjudication; gate. Panic-memory rows are not re-reviewed. The accepted rows
join the v4 selection (s7_release3.release_v4). Runner: prod_rereview.sh (CPU + paid text only, no GPU step).
"""
from __future__ import annotations

from final_construction.wp_v9_vcm import common as cm


def _r7(uc: cm.Cfg) -> tuple[dict, dict]:
    rows = {r["row_id"]: r for r in cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_r7.json"))["rows"]}
    gate = {o["row_id"]: o for o in cm.read(uc.p("s5_r7", "gate.json"))["outcomes"]}
    return rows, gate


def candidates(uc: cm.Cfg) -> list[dict]:
    """Rejected, deterministic-pass safe generic rows of round 7."""
    if not uc.p("s5_r7", "gate.json").exists():
        return []
    rows, gate = _r7(uc)
    return [r for rid, r in sorted(rows.items()) if r.get("extra_kind") == "A" and r["deterministic_pass"]
            and not gate[rid]["accepted"]]


def needed(uc: cm.Cfg) -> bool:
    return bool(candidates(uc))


def assemble_rereview(cfg: cm.Cfg) -> dict:
    """`assemble --round 9`: rows_<main>_r9.json (the candidates + scene texts) and the reused Qwen verdicts."""
    from final_construction.wp_v9_vcm import s2_extra
    out = cm.rows_path(cfg)
    if out.exists():
        return {"status": "exists"}
    sel = s2_extra.selected(cm.Cfg(**{**cfg.__dict__, "round": cm.EXTRA_ROUND}))
    rows = []
    for r in candidates(cfg):
        texts = s2_extra.scene_texts([x for x in sel if x["family_id"] == r["family_id"]])
        rows.append({**r, "scene_texts": texts, "round": cfg.round, "rereview_of": "s5_r7"})
        src = cfg.p("s5_r7", "second", cm.safe_name(r["row_id"]) + ".json")
        dst = cfg.p(cm.s5_dir(cfg), "second", cm.safe_name(r["row_id"]) + ".json")
        if src.exists() and not dst.exists():
            cm.write_once(dst, {**cm.read(src), "reused_from": "s5_r7/second (same row content, Qwen with images)"})
    cm.write_once(out, {"rows": rows, "report": {"round": cfg.round, "rereview": len(rows), "use_wrap": False,
                                                 "families": {r["family_id"]: "assembled" for r in rows}}})
    return {"rows": len(rows)}


def accepted(uc: cm.Cfg) -> list[dict]:
    """Accepted round-9 rows (with their gate outcome under `_outcome`, consumed by the v4 selection)."""
    g = uc.p("s5_r9", "gate.json")
    if not g.exists():
        return []
    rows = {r["row_id"]: r for r in cm.read(uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_r9.json"))["rows"]}
    return [{**rows[o["row_id"]], "review_mode": "text_extra", "_outcome": {**o, "round": 9}}
            for o in cm.read(g)["outcomes"] if o["accepted"]]
