"""Shared paths, system registry and budget caps for the fix2 pipeline (text fixes -> delta re-eval). CPU only.

No API client is imported anywhere in this package. Frozen inputs are only read; every output goes to a NEW path:
  staging (default):  <WORK>/staging/v4_build/<name containing 'fix2'>/
  --materialize:      $FC/v4_build/<same name>/   (refuses to overwrite an existing file)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

WORK = Path(__file__).resolve().parent
FC = (Path(__file__).resolve().parents[2] / 'bv3' / 'final_construction')
V4 = FC / "v4_build"
STAGING = WORK / "staging" / "v4_build"
PY = sys.executable     # CPU tools, vLLM eval.sh driver (as conflict RUNBOOK)
PYA = sys.executable        # API evaluate_ae / claude batch (as conflict RUNBOOK)

# frozen inputs (read only)
SUB = V4 / "conflict_subset_v1"
GTFIX = V4 / "conflict_gtfix_v1"
CONFLICT_SETS = {
    "v1": {"tag": "conflict_v1", "release": SUB / "RELEASE_TEST_v4q_conflict_v1", "key_actions": SUB / "KEY_ACTIONS_conflict_v1.jsonl",
           "inventory": GTFIX / "INVENTORY_conflict_v1_gtfix1.jsonl", "new_rows": GTFIX / "NEW_ROWS_conflict_v1_gtfix1.jsonl",
           "eval": V4 / "conflict_eval_v1", "judges_merged": GTFIX / "judges_merged/conflict_v1"},
    "v2add": {"tag": "conflict_v2add", "release": SUB / "RELEASE_TEST_v4q_conflict_v2add", "key_actions": SUB / "KEY_ACTIONS_conflict_v2add.jsonl",
              "inventory": GTFIX / "INVENTORY_conflict_v2add_gtfix1.jsonl", "new_rows": GTFIX / "NEW_ROWS_conflict_v2add_gtfix1.jsonl",
              "eval": V4 / "conflict_eval_v2add", "judges_merged": GTFIX / "judges_merged/conflict_v2add"},
}
MAIN_RELEASE = FC / "wp_v9_vcm/holdout_v9_sealed/RELEASE_TEST_v4q"
MAIN_MODES = ("retrieved", "gt_retrieved", "gt_retrieved_drop")
MAIN_INVENTORY = V4 / "final_eval_assembly/medfix_v1/INVENTORY_TEST_v4q_APPROVED_medfix_v1.jsonl"
KEY_ACTIONS_CANON = V4 / "g0_redesign_v1/rebuild_v20/KEY_ACTIONS_CANONICAL_v4q.jsonl"
QUERY_EMB = V4 / "online/query_emb_test_v4i"
TRAIN_RELEASES = (FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1", FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1_p2e")

# new output names (all contain 'fix2')
CONFLICT_BUILD = "conflict_fix2_v1"
CONFLICT_EVAL = "conflict_eval_fix2_v1"
MAIN_BUILD = "release_test_v4q_fix2"
MAIN_EVAL = "main_eval_fix2_v1"
RECOMPUTE = "recompute_fix2_v1"

# hazard-type -> group (= final_eval_assembly/assembly_common.GROUPS; G1 = person down)
GROUPS = {"person_distress": "G1", "fall": "G1", "fire_heat": "G2", "electrical": "G2", "chemical": "G2", "combustion_fumes": "G2",
          "battery_overheat": "G2", "machine": "G3", "falling_object": "G3", "sharp": "G3", "animal_attack": "G4",
          "stinging_insects": "G4", "venomous_creature": "G4", "wild_or_aggressive_animal": "G4", "water_flood": "G5",
          "intruder": "G5", "choking": "G5", "structural_hazard": "G5", "storm_damage": "G5"}

# systems (conflict_eval_v2add/conflict_common.LOCAL ids; ours_9b = PReMind-9B = original v2 s25; p2f rejected 10-25)
LOCAL_SYSTEMS = ["base4b", "base9b", "ours_4b_p2e", "ours_9b", "zs_qwen3vl_8b", "zs_internvl3_5_4b", "zs_internvl3_5_8b",
                 "zs_glm4_6v_flash", "zs_covip_8b", "zs_repic_7b", "abl_noPF_p12", "abl_noSK_pf", "abl_noSO",
                 "ours_4b_p2e_nomie", "ours_4b_p2e_s1"]
OPTIONAL_LOCAL = {"ours_4b_p2e_s1"}          # seed-1 replicate: run only if its adapter exists (rebuttal only)
API_SYSTEMS = {"gpt6luna_high": "api:openai:gpt-6-luna@high", "opus55_medium_cap1024": "api:anthropic:claude-opus-5-5@medium",
               "gpt61sol_medium": "api:openai:gpt-6.1-sol@medium"}
API_ORDER = ["gpt6luna_high", "opus55_medium_cap1024", "gpt61sol_medium"]   # Luna first; owner checkpoint before Opus/Sol
PROVIDER = {"gpt6luna_high": "openai", "gpt61sol_medium": "openai", "opus55_medium_cap1024": "anthropic", "luna_judge": "openai"}
# owner caps 2026-10-27 (DO_LIST N11): OpenAI total (Luna/Sol inference + Luna judges) <= $30, Anthropic (Opus) <= $10
PROVIDER_CAP_USD = {"openai": 30.0, "anthropic": 10.0}
# main-test assembly ids (Table 2 config) for the same checkpoints
MAIN_ASSEMBLY_ID = {"ours_4b_p2e": "ours_4b", "ours_9b": "ours_9b"}
OWNER_GPUS = (4, 5, 6, 7)                    # owner preference 2026-10-14/27; 0-3 only for preemptible jobs (not used here)


def out_root(materialize: bool) -> Path:
    return V4 if materialize else STAGING


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_new(p: Path, text: str, materialize: bool) -> None:
    """Write a new file. In --materialize mode an existing file is never overwritten (write-once)."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and (materialize or not str(p).startswith(str(WORK))):
        raise FileExistsError(f"refusing to overwrite {p}")
    p.write_text(text)


def write_jl(p: Path, rows: Iterable[dict], materialize: bool) -> None:
    write_new(p, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), materialize)


def write_json(p: Path, obj, materialize: bool) -> None:
    write_new(p, json.dumps(obj, indent=1, ensure_ascii=False) + "\n", materialize)


def type_groups(types: Iterable[str]) -> list[str]:
    return sorted({GROUPS[t] for t in types})


def unseen_by_types(types: Iterable[str]) -> bool:
    """assembly_common.row_groups rule: UNSEEN iff the row has any hazard group other than G1."""
    return bool(set(type_groups(types)) - {"G1"})


def no_secrets_env() -> dict[str, str]:
    """Environment for CPU subprocesses: API keys removed, GPUs hidden."""
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env
