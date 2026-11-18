"""Paths, sys.path setup and holdout refusal shared by trainer_v2 (import first).

Import-only use of: wp_v9_vcm (render + context modes), the reward lineage (rl_sweep r6 -> v3 -> v2 + baseline_eval
parser) and trainer_v1 (mock policy / tokenizer, train.load_image). Nothing under those packages is modified.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parent                       # final_construction
V3 = FC.parent                         # package root
R6 = V3 / "rl_sweep" / "rlvr_v3_prep" / "r6"
TI = FC / "train_interface_v1"
VCM_DIR = FC / "wp_v9_vcm"
CANARY_DIR = VCM_DIR / "canary_v9"
CANARY_ROWS = CANARY_DIR / "s4" / "rows_siglip2.json"            # pre-release s4 rows (smoke fallback)
CANARY_RENDERED = CANARY_DIR / "s6" / "rendered_rows.json"     # finalized canary render (train/dev, accepted)
OPEN_SPLITS = ("train", "dev")

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # trainer_v1 config pin
MODEL_ID_9B = "Qwen/Qwen3.5-9B"
MODEL_REVISION_9B = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # baseline_eval/config.py pin
MODELS = {"4b": (MODEL_ID, MODEL_REVISION), "9b": (MODEL_ID_9B, MODEL_REVISION_9B)}
PROD_S6 = VCM_DIR / "prod_v9" / "s6"                              # production renders (when they exist)

for _p in (str(V3), str(FC), str(R6), str(TI)):
    if _p not in sys.path:
        sys.path.append(_p)  # appended: never shadow the caller's own modules


class HoldoutRefusedError(RuntimeError):
    """A holdout path or row reached trainer_v2 (never allowed)."""


def refuse_holdout_path(path: object) -> None:
    for s in (str(path), str(Path(str(path)).resolve())):
        if "holdout" in s.lower():
            raise HoldoutRefusedError(f"refused: path contains 'holdout': {path}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
