"""Pin of a v4hcur22b run (main_v4hcur22b): pins_v4hcur22.block (v4h + aa + cur, curriculum v2.2) + ``amb`` (bonus
v1.1) + ``fa`` (false-alarm penalty mode / value / base) blocks, with the sha256 of the bonus / penalty / entry code.
prereg PREREG_TRAINER_V2_V4HCUR22B, base_prereg PREREG_TRAINER_V2_V4HCUR22.
  python -I -m trainer_v2_v4.pins_v4hcur22b write --out PIN.json -- <main_v4hcur22b args>  (write once; PIN_WORLD=<ranks>)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import pins_v4hcur21 as P21  # noqa: E402
from trainer_v2_v4 import pins_v4hcur22 as P22  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR22B"
CODE = ("acute_memory_bonus.py", "acute_memory_bonus_v1_1.py", "false_alarm_pen.py", "main_v4hcurb.py",
        "main_v4hcur22b.py", "pins_v4hcur22b.py")
BLOCKS = ("v4h", "aa", "cur", "amb", "fa")


def opt_blocks(opt: argparse.Namespace) -> dict:
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import acute_memory_bonus_v1_1 as AMB
    from trainer_v2_v4 import false_alarm_pen as FA
    return {"amb": {"code_sha256": {f: P21._sha(HERE / f) for f in CODE}, "acute_memory_bonus": opt.acute_memory_bonus,
                    "version": AMB.VERSION, "bonus": AMB.BONUS, "personal_content_weight": R5.W["personal_content"],
                    "criterion": AMB.CRITERION},
            "fa": {"false_alarm_pen": opt.false_alarm_pen, "value": FA.VALUE.get(opt.false_alarm_pen, FA.BASE),
                   "base": FA.BASE, "applies_to": ["reward_v5 tag false alarm", "warning_credit effective risk",
                                                   "absent_visible_risk re-score"]}}


def block(opt: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur22 as M22
    cur, rest_c = M22.resolve(rest)
    out = P22.block(cur, rest_c)
    out["base_prereg"] = out.get("prereg")
    out["prereg"] = PREREG
    out.update(opt_blocks(opt))
    return json.loads(json.dumps(out, default=str))


def verify(pin: Path, opt: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(opt, rest)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22b pin (prereg={pre.get('prereg')!r})")
    bad = P21.diff(pre, now, BLOCKS)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22b pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22b.py write --out PIN.json -- <main_v4hcur22b args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcur22b as M22B
    opt, rest = M22B.resolve(run)
    if opt.prereg_v4hcur22b:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcur22b")
    blk = block(opt, rest)
    if blk.get("base_prereg") != P22.PREREG or (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur22b pin needs --curriculum v22")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
