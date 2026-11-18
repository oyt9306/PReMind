"""Pin of a CUR22B-on-9B run (main_v4hcur22b_9b): the full pins_v4hcur22b block (v4h + aa + cur + amb + fa, built by
the frozen pins_v4hcur22b.block from the same arguments) + ``m9b`` (model size / id / revision and the memory knobs:
gen batch, train micro batch, LoRA rank, world, prompts per rank, sha256 of the 9B wrapper code).
prereg PREREG_TRAINER_V2_V4HCUR22B_9B, base_prereg PREREG_TRAINER_V2_V4HCUR22B. NEW file; pins_v4hcur22b is imported
read-only. World size = PIN_WORLD, else WORLD_SIZE (torchrun), else 1: export PIN_WORLD=<ranks> for the pin write AND
the run (the CUR23 9B launch once failed because the run-time check read the PIN_WORLD default 1).
  python -I -m trainer_v2_v4.pins_v4hcur22b_9b write --out PIN.json -- <main_v4hcur22b_9b args>  (write once)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import pins_v4hcur21 as P21  # noqa: E402
from trainer_v2_v4 import pins_v4hcur22b as P22B  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR22B_9B"
CODE = ("main_v4hcur22b_9b.py", "pins_v4hcur22b_9b.py", "scripts/v4hcur22b_9b_plan.sh")
BLOCKS = P22B.BLOCKS + ("m9b",)
MODEL = "9b"
OWN_FLAG = "--prereg-v4hcur22b-9b"


def world() -> int:
    return int(os.environ.get("PIN_WORLD") or os.environ.get("WORLD_SIZE") or "1")


def _value(argv: list[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    from trainer_v2_v4 import main_v4h as MH
    v = MH._value(argv, flag)
    return default if v is None else v


def m9b_block(run: list[str]) -> dict:
    """Model + memory knobs of the run (defaults = run_v2 / main_run defaults when a flag is absent)."""
    from trainer_v2 import paths_v2 as P
    size = _value(run, "--model", "4b")
    mid, rev = P.MODELS[size]
    return {"code_sha256": {f: P21._sha(HERE / f) for f in CODE}, "model": size, "model_id": mid, "revision": rev,
            "gen_batch": int(_value(run, "--gen-batch", "16")), "batch_rollout": "--batch-rollout" in run,
            "train_micro_batch": int(_value(run, "--train-micro-batch", "1")),
            "lora_r": int(_value(run, "--lora-r", "16")), "prompts_per_rank": int(_value(run, "--prompts-per-step")),
            "world": world()}


def block(run: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur22b as M22B
    if any(a == OWN_FLAG or a.startswith(OWN_FLAG + "=") for a in run):
        raise SystemExit(f"REFUSED: the pinned arguments must not contain {OWN_FLAG}")
    opt, rest = M22B.resolve(run)
    if opt.prereg_v4hcur22b:
        raise SystemExit("REFUSED: a 9B run is pinned by pins_v4hcur22b_9b, not --prereg-v4hcur22b")
    out = P22B.block(opt, rest)
    out["base_prereg"] = P22B.PREREG
    out["prereg"] = PREREG
    out["m9b"] = m9b_block(run)
    return json.loads(json.dumps(out, default=str))


def verify(pin: Path, run: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(run)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22b_9b pin (prereg={pre.get('prereg')!r})")
    bad = P21.diff(pre, now, BLOCKS)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22b_9b pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22b_9b.py write --out PIN.json -- <main_v4hcur22b_9b args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    blk = block(run)
    if (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur22b_9b pin needs --curriculum v22")
    if blk["m9b"]["model"] != MODEL:
        raise SystemExit(f"REFUSED: a v4hcur22b_9b pin needs --model {MODEL} (got {blk['m9b']['model']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
