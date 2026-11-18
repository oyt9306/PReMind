"""Pin of the CUR22S single-stage run (main_v4hcur22s; CUR22S.md) = the CUR22B block rebuilt by the frozen
pins_v4hcur22b.block from the run's arguments (code_sha256 / constants / v4h / aa / cur / amb / fa / base_prereg) + an
``s22`` block (recall terms ws52u1 + credit_gate, stop_at_step 150, sha256 of the new / reused code).
Declared differences vs the launched CUR22B pin (the write refuses anything else, and every other CUR22B block must be
byte-identical in its json serialisation):
  constants.arm   passive -> memory_call
  s22.recall      ws52u1 + credit_gate        s22.stop_at_step   150        s22.code_sha256 (bookkeeping)
  prereg          PREREG_TRAINER_V2_V4HCUR22S (+ s22_base_prereg = PREREG_TRAINER_V2_V4HCUR22B)
  python -I -m trainer_v2_v4.pins_v4hcur22s write --out PIN.json --cur22b-pin CUR22B_PIN.json -- <main_v4hcur22s args>
(write once; PIN_WORLD=<ranks>). ``verify`` compares everything except the write-time ``cur22b_check`` record. NEW file.
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

PREREG = "PREREG_TRAINER_V2_V4HCUR22S"
C22B_PREREG = "PREREG_TRAINER_V2_V4HCUR22B"
CODE = ("main_v4hcur22s.py", "pins_v4hcur22s.py", "recall_decision.py", "recall_decision_u1.py", "abl_hooks.py",
        "main_v4hcur23r.py", "scripts/v4hcur22s_single_plan.sh")
CHECK = "cur22b_check"
DECLARED = ("constants.arm", "s22", "s22_base_prereg")      # dotted-path prefixes allowed to differ from CUR22B


def s22_block(o, rest22: list[str]) -> dict:
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import recall_decision as RD
    from trainer_v2_v4 import recall_decision_u1 as RDU
    return {"code_sha256": {f: P21._sha(HERE / f) for f in CODE},
            "recall": {"recall_decision": o.recall_decision, "crowdout_guard": o.crowdout_guard,
                       "weights": dict(RDU.WEIGHTS), "missed_call": RDU.MISSED_CALL,
                       "unneeded_call": RDU.UNNEEDED_CALL, "r": float(MH._value(rest22, "--r") or "nan"),
                       "source_preset": RD.SOURCE_PRESET, "guard_rule": RD.GUARD_RULE, "version": RDU.VERSION,
                       "base_version": RD.VERSION, "init_lora": None},
            "stop_at_step": o.abl_stop_at_step}


def block(argv: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur22s as MS
    from trainer_v2_v4 import pins_v4hcur22b as PB
    o, o22b, rest22 = MS.resolve(argv)
    out = PB.block(o22b, rest22)
    out["s22_base_prereg"] = out.get("prereg")
    out["prereg"] = PREREG
    out["s22"] = s22_block(o, rest22)
    return json.loads(json.dumps(out, default=str))


def ser(v) -> str:
    return json.dumps(v, indent=1, default=str)


def check(blk: dict, c22b: dict) -> dict:
    """The write-time record; SystemExit unless the pin differs from CUR22B only in the declared fields."""
    if c22b.get("prereg") != C22B_PREREG:
        raise SystemExit(f"REFUSED: --cur22b-pin is not a CUR22B pin (prereg={c22b.get('prereg')!r})")
    d = PA.diff(c22b, blk, skip=())
    extra = [p for p in d if not PA.covered([p], DECLARED)]
    if extra or "constants.arm" not in d or not any(p.startswith("s22.recall.") for p in d):
        raise SystemExit(f"REFUSED: CUR22S differs from CUR22B beyond {list(DECLARED)}: {extra} (diff {d})")
    same = [k for k in c22b if k not in ("prereg", "constants")]
    bad = [k for k in same if ser(c22b[k]) != ser(blk.get(k))]
    bad += [f"constants.{k}" for k in set(c22b["constants"]) | set(blk["constants"])
            if k != "arm" and ser(c22b["constants"].get(k)) != ser(blk["constants"].get(k))]
    if bad:
        raise SystemExit(f"REFUSED: CUR22B blocks not byte-identical: {bad}")
    return {"diff_vs_CUR22B": d, "byte_identical_blocks": sorted(same) + ["constants (except arm)"]}


def verify(pin: Path, argv: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(argv)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22s pin (prereg={pre.get('prereg')!r})")
    pre = {k: v for k, v in pre.items() if k != CHECK}
    bad = PA.diff(pre, now, skip=()) + [k for k in ("prereg", "base_prereg", "s22_base_prereg") if pre.get(k) != now.get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22s pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 5 or argv[0] != "write" or argv[1] != "--out" or argv[3] != "--cur22b-pin" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22s.py write --out PIN.json --cur22b-pin CUR22B_PIN.json -- <args>")
    out, c22b_path, run = Path(argv[2]), Path(argv[4]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcur22s as MS
    if any(a == MS.FLAG or a.startswith(MS.FLAG + "=") for a in run):
        raise SystemExit(f"REFUSED: the pinned arguments must not contain {MS.FLAG}")
    blk = block(run)
    if (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur22s pin needs --curriculum v22")
    blk[CHECK] = check(blk, json.loads(c22b_path.read_text()))
    blk[CHECK]["cur22b_pin"] = str(c22b_path)
    blk[CHECK]["cur22b_pin_sha256"] = P21._sha(c22b_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:                          # write once (race-safe)
        fh.write(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
