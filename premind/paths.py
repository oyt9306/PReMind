"""Model pins and the holdout guard shared by training and evaluation (formerly trainer_v2/paths_v2.py)."""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN_SPLITS = ("train", "dev")

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # trainer_v1 config pin
MODEL_ID_9B = "Qwen/Qwen3.5-9B"
MODEL_REVISION_9B = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # baseline_eval/config.py pin
MODELS = {"4b": (MODEL_ID, MODEL_REVISION), "9b": (MODEL_ID_9B, MODEL_REVISION_9B)}


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
