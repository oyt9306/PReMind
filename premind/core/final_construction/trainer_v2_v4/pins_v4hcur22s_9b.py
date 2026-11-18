"""Pin of the CUR22S run on Qwen3.5-9B (main_v4hcur22s_9b; CUR22S_9B.md) = the CUR22S block (the frozen
pins_v4hcur22b.block + pins_v4hcur22s.s22_block, built from the run's arguments exactly as pins_v4hcur22s.block does)
+ ``m9b`` (model / id / revision, gen batch, batch rollout, train micro batch, LoRA rank, prompts per rank, world,
sha256 of the 9B wrapper code; fields as pins_v4hcur22b_9b.m9b_block).
Declared differences vs the launched 4B CUR22S pin (the write refuses anything else; every other block must be
byte-identical in its json serialisation, the 4B write-time ``cur22b_check`` record excluded):
  m9b (new)    prereg PREREG_TRAINER_V2_V4HCUR22S_9B    s22s_base_prereg PREREG_TRAINER_V2_V4HCUR22S (new)
  python -I -m trainer_v2_v4.pins_v4hcur22s_9b write --out PIN.json --cur22s-pin CUR22S_PIN.json -- <args>
(write once; export PIN_WORLD=<ranks> for the write AND the run). ``verify`` skips the write-time ``cur22s_check``.
NEW file; pins_v4hcur22b / pins_v4hcur22s / pins_v4hcur22b_9b are imported read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import pins_v4hcur21 as P21  # noqa: E402
from trainer_v2_v4 import pins_v4hcur22abl as PA  # noqa: E402
from trainer_v2_v4 import pins_v4hcur22b_9b as P9B  # noqa: E402
from trainer_v2_v4 import pins_v4hcur22s as PS  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR22S_9B"
CODE = ("main_v4hcur22s_9b.py", "pins_v4hcur22s_9b.py", "scripts/v4hcur22s_9b_plan.sh")
CHECK = "cur22s_check"
DECLARED = ("m9b", "s22s_base_prereg")       # dotted-path prefixes allowed to differ from the 4B CUR22S pin
MODEL = "9b"


def m9b_block(run: list[str]) -> dict:
    """= pins_v4hcur22b_9b.m9b_block with this wrapper's code files."""
    from trainer_v2 import paths_v2 as P
    v = P9B._value
    size = v(run, "--model", "4b")
    mid, rev = P.MODELS[size]
    return {"code_sha256": {f: P21._sha(HERE / f) for f in CODE}, "model": size, "model_id": mid, "revision": rev,
            "gen_batch": int(v(run, "--gen-batch", "16")), "batch_rollout": "--batch-rollout" in run,
            "train_micro_batch": int(v(run, "--train-micro-batch", "1")), "lora_r": int(v(run, "--lora-r", "16")),
            "prompts_per_rank": int(v(run, "--prompts-per-step")), "world": P9B.world()}


def block(run: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur22s_9b as E
    from trainer_v2_v4 import pins_v4hcur22b as PB
    pin, run = E.split(run)
    if pin is not None:
        raise SystemExit(f"REFUSED: the pinned arguments must not contain {E.FLAG}")
    o, o22b, rest22 = E.resolve(run)
    out = PB.block(o22b, rest22)                # = pins_v4hcur22s.block, with the 9B resolve
    out["s22_base_prereg"] = out.get("prereg")
    out["prereg"] = PREREG
    out["s22"] = PS.s22_block(o, rest22)
    out["s22s_base_prereg"] = PS.PREREG
    out["m9b"] = m9b_block(run)
    return json.loads(json.dumps(out, default=str))


def check(blk: dict, c22s: dict) -> dict:
    """The write-time record; SystemExit unless the pin differs from the 4B CUR22S pin only in the declared fields."""
    if c22s.get("prereg") != PS.PREREG:
        raise SystemExit(f"REFUSED: --cur22s-pin is not a CUR22S pin (prereg={c22s.get('prereg')!r})")
    ref = {k: v for k, v in c22s.items() if k != PS.CHECK}
    d = PA.diff(ref, blk, skip=())
    extra = [p for p in d if not PA.covered([p], DECLARED)]
    if extra or not any(p.startswith("m9b.") for p in d):
        raise SystemExit(f"REFUSED: CUR22S_9B differs from CUR22S 4B beyond {list(DECLARED)}: {extra} (diff {d})")
    same = sorted(k for k in ref if k != "prereg")
    bad = [k for k in same if PS.ser(ref[k]) != PS.ser(blk.get(k))]
    if bad:
        raise SystemExit(f"REFUSED: CUR22S blocks not byte-identical: {bad}")
    return {"diff_vs_CUR22S_4B": d, "byte_identical_blocks": same}


def verify(pin: Path, run: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(run)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22s_9b pin (prereg={pre.get('prereg')!r})")
    pre = {k: v for k, v in pre.items() if k != CHECK}
    bad = PA.diff(pre, now, skip=()) + [k for k in ("prereg", "base_prereg", "s22_base_prereg")
                                        if pre.get(k) != now.get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22s_9b pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 5 or argv[0] != "write" or argv[1] != "--out" or argv[3] != "--cur22s-pin" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22s_9b.py write --out PIN.json --cur22s-pin CUR22S_PIN.json -- <args>")
    out, c22s_path, run = Path(argv[2]), Path(argv[4]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    blk = block(run)
    if (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur22s_9b pin needs --curriculum v22")
    if blk["m9b"]["model"] != MODEL:
        raise SystemExit(f"REFUSED: a v4hcur22s_9b pin needs --model {MODEL} (got {blk['m9b']['model']})")
    blk[CHECK] = check(blk, json.loads(c22s_path.read_text()))
    blk[CHECK]["cur22s_pin"] = str(c22s_path)
    blk[CHECK]["cur22s_pin_sha256"] = P21._sha(c22s_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:                          # write once (race-safe)
        fh.write(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
