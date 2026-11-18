"""Pin of a v4hcurb run (main_v4hcurb): the base entry's pin block (default pins_v4hcur.block, unchanged: v4h + aa +
cur blocks) + the ``amb`` block: sha256 of the bonus code (acute_memory_bonus.py, main_v4hcurb.py, this file), the
``--acute-memory-bonus`` value, the bonus weight, the personal_content full weight and the criterion.
  python -I -m trainer_v2_v4.pins_v4hcurb write --out PIN.json -- <main_v4hcurb args>   (write once; PIN_WORLD=<ranks>)
``verify`` (main_v4hcurb --prereg-v4hcurb PIN) refuses on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

PREREG = "PREREG_TRAINER_V2_V4HCURB"
CODE_AMB = ("acute_memory_bonus.py", "main_v4hcurb.py", "pins_v4hcurb.py")
BLOCKS = ("v4h", "aa", "cur", "amb")


def _sha(p: Path) -> Optional[str]:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _defaults(base: Optional[ModuleType], base_pins: Optional[ModuleType]) -> tuple[ModuleType, ModuleType]:
    if base is None or base_pins is None:
        from trainer_v2_v4 import main_v4hcur as MC
        from trainer_v2_v4 import pins_v4hcur as PC
        return base or MC, base_pins or PC
    return base, base_pins


def amb_block(amb: argparse.Namespace) -> dict:
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import acute_memory_bonus as AMB
    return {"code_sha256": {f: _sha(HERE / f) for f in CODE_AMB}, "acute_memory_bonus": amb.acute_memory_bonus,
            "bonus": AMB.BONUS, "personal_content_weight": R5.W["personal_content"], "criterion": AMB.CRITERION}


def block(amb: argparse.Namespace, rest: list[str], *, base: Optional[ModuleType] = None,
          base_pins: Optional[ModuleType] = None) -> dict:
    """``rest`` = main_v4hcurb.resolve(...)[1] (the base-entry argv without a pin flag)."""
    base, base_pins = _defaults(base, base_pins)
    opts, rest_b = base.resolve(rest)
    out = base_pins.block(opts, rest_b)
    out["base_prereg"] = out.get("prereg")
    out["prereg"] = PREREG
    out["amb"] = amb_block(amb)
    return out


def verify(pin: Path, amb: argparse.Namespace, rest: list[str], *, base: Optional[ModuleType] = None,
           base_pins: Optional[ModuleType] = None) -> None:
    pre = json.loads(Path(pin).read_text())
    now = json.loads(json.dumps(block(amb, rest, base=base, base_pins=base_pins), default=str))
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcurb pin (prereg={pre.get('prereg')!r})")
    bad = [k for k in sorted(set(pre) | set(now)) if k not in BLOCKS and pre.get(k) != now.get(k)]
    for blk in BLOCKS:
        a, b = pre.get(blk) or {}, now.get(blk) or {}
        bad += [f"{blk}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4hcurb pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcurb.py write --out PIN.json -- <main_v4hcurb args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcurb as MB
    amb, rest = MB.resolve(run)
    if amb.prereg_v4hcurb:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcurb")
    blk = block(amb, rest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
