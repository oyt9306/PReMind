"""Shared paths / system registry / pins for the conflict-subset evaluation (RELEASE_TEST_v4q_conflict_v1). CPU only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parents[1]
SUB = FC / "v4_build/conflict_subset_v1"
RELEASE = SUB / "RELEASE_TEST_v4q_conflict_v1"
HOLDOUT = RELEASE / "holdout.jsonl"
KEY_ACTIONS = SUB / "KEY_ACTIONS_conflict_v1.jsonl"
INVENTORY = SUB / "INVENTORY_conflict_v1.jsonl"
QUERY_EMB = FC / "v4_build/online/query_emb_test_v4i"
N_ROWS = 81
RUNS = HERE / "runs"
JUDGES = HERE / "judges"
APPROVAL = HERE / "OWNER_APPROVAL.jsonl"

LOCAL = {
    "base4b": ("4b", None),
    "base9b": ("9b", None),
    "ours_4b": ("4b", FC / "eval_v1/runs/single_v4qS_cur22s_ws125_stillpen_hca1_v2_v20_4b_s0/checkpoints/step_000025/lora"),
    "ours_9b": ("9b", FC / "eval_v1/runs/single_v4qS_cur22s_ws125_stillpen_hca1_v2_v20_9b_s0/checkpoints/step_000025/lora"),
}
LOCAL["ours_4b_p2d"] = ("4b", FC / "eval_v1/runs/p2d_mie_v1_4b_s0/checkpoints/step_000050/lora")   # final 4B (Phase II''' s50, 2026-10-24)
LOCAL["ours_4b_p2e"] = ("4b", FC / "eval_v1/runs/p2e_mie_v1_4b_s0/checkpoints/step_000025/lora")   # Phase II-quadruple-prime s25 candidate (2026-10-24)
LOCAL["ours_9b_p2f"] = ("9b", FC / "eval_v1/runs/p2f_corr_v1_9b_s0/checkpoints/step_000010/lora")   # final PReMind-9B (9B correction run p2f s10, adapter 035df9ae2c78, 2026-10-24)
LOCAL["ours_4b_p2e_s1"] = ("4b", FC / "eval_v1/runs/p2e_mie_v1_4b_s1/checkpoints/step_000025/lora")   # seed-1 replicate of the final PReMind-4B (p2e s1 s25, adapter eaa8225c98f0, PREREG Amendment 9q, 2026-10-25)
LOCAL["ours_4b_p2e_nomie"] = ("4b", FC / "eval_v1/runs/p2e_nomie_v1_4b_s0/checkpoints/step_000025/lora")   # M3 control (PREREG Amendment 9r)
for _s in ("noPF_p12", "noSK_pf", "noSO"):
    LOCAL[f"abl_{_s}"] = ("4b", FC / f"eval_v1/runs/single_v4qS_cur22s_{_s}_v20_4b_s0/checkpoints/step_000150/lora")
for _m in ("qwen3vl_8b", "internvl3_5_4b", "internvl3_5_8b", "glm4_6v_flash", "covip_8b", "repic_7b"):
    LOCAL[f"zs_{_m}"] = (_m, None)   # open baselines, same eval.sh protocol as Table 2 (owner 2026-10-22)
API = {"gpt61sol_medium": "api:openai:gpt-6.1-sol@medium", "gpt6luna_high": "api:openai:gpt-6-luna@high",
       "opus55_medium_cap1024": "api:anthropic:claude-opus-5-5@medium"}
SYSTEMS = list(LOCAL) + list(API)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def manifest() -> dict:
    m = json.loads((RELEASE / "MANIFEST.json").read_text())
    for p, k in ((HOLDOUT, "holdout_sha256"), (KEY_ACTIONS, "key_actions_sha256"), (INVENTORY, "mie_inventory_sha256")):
        if sha(p) != m[k]:
            raise RuntimeError(f"{p.name} differs from MANIFEST {k}")
    return m


def run_dir(sid: str) -> Path:
    if sid not in SYSTEMS:
        raise ValueError(f"unknown system {sid}")
    return RUNS / sid / "retrieved"


def check_run(sid: str, run: Path | None = None) -> dict:
    """Completed-run check (raises): release sha, retrieved/memory_call, greedy, K4, 81 rows, system identity."""
    run = run or run_dir(sid)
    meta = json.loads((run / "eval_report.json").read_text())["meta"]
    m = manifest()
    if meta["data"]["retrieved"]["sha256"] != m["holdout_sha256"] or Path(meta["data"]["retrieved"]["path"]).resolve() != HOLDOUT.resolve():
        raise ValueError(f"{sid}: run not bound to the conflict release")
    if (meta["modes"], meta["arms"], meta.get("samples"), meta.get("online_k")) != (["retrieved"], ["memory_call"], 0, 4):
        raise ValueError(f"{sid}: protocol mismatch {meta['modes']} {meta['arms']} samples={meta.get('samples')} k={meta.get('online_k')}")
    if Path(meta.get("online_query_emb", "")).resolve() != QUERY_EMB.resolve():
        raise ValueError(f"{sid}: query embedding mismatch")
    if sid in LOCAL:
        size, lora = LOCAL[sid]
        if meta["model"] != size or meta.get("backend") not in (("vllm", "vllm_text") if sid.startswith("zs_") else ("vllm",)):
            raise ValueError(f"{sid}: model/backend mismatch")
        if (meta.get("lora") is None) != (lora is None) or (lora and Path(meta["lora"]).resolve() != lora.resolve()):
            raise ValueError(f"{sid}: lora mismatch {meta.get('lora')}")
    else:
        if meta["model"] != API[sid]:
            raise ValueError(f"{sid}: model mismatch {meta['model']}")
        if sid.endswith("cap1024") and (meta.get("claude_overrides") or {}).get("visible_cap_stage2") != 1024:
            raise ValueError(f"{sid}: visible cap 1024 override missing")
    recs = jl(run / "records.jsonl")
    if len(recs) != N_ROWS or len({r["row_id"] for r in recs}) != N_ROWS or len(jl(run / "texts.jsonl")) != N_ROWS:
        raise ValueError(f"{sid}: expected {N_ROWS} records/texts")
    return meta
