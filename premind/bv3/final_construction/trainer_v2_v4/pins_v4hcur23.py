"""Pin of a v4hcur23 run (main_v4hcur23): pins_v4hcur22b.block (v4h + aa + cur (curriculum v2.2) + amb + fa) + ``h23``
(hazard judge rubric + severity cap: modes, values, frozen rubric / module sha256s, sha256 of the new code).
prereg PREREG_TRAINER_V2_V4HCUR23, base_prereg PREREG_TRAINER_V2_V4HCUR22B.
  python -I -m trainer_v2_v4.pins_v4hcur23 write --out PIN.json -- <main_v4hcur23 args>  (write once; PIN_WORLD=<ranks>)
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
from trainer_v2_v4 import pins_v4hcur22b as P22B  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR23"
CODE = ("hazard_judge_v4.py", "harm_severity_cap.py", "main_v4hcur23.py", "pins_v4hcur23.py")
BLOCKS = P22B.BLOCKS + ("h23",)


def h23_block(o23: argparse.Namespace) -> dict:
    from trainer_v2_v4 import harm_severity_cap as HSC
    from trainer_v2_v4 import hazard_judge_v4 as HJ4
    out: dict = {"code_sha256": {f: P21._sha(HERE / f) for f in CODE}, "hazard_rubric": o23.hazard_rubric,
                 "harm_severity_cap": o23.harm_severity_cap}
    if o23.hazard_rubric == "v4":
        A4 = HJ4.a4()
        out["hazard_judge"] = {"version": HJ4.VERSION, "purpose": HJ4.PURPOSE, "rubric_sha256": HJ4.RUBRIC_V4_SHA256,
                               "module_sha256": HJ4.A4_SHA256, "adequacy_v1_sha256": P21._sha(Path(A4.A1.__file__)),
                               "state_bits": {k: list(v) for k, v in HJ4.STATE_BITS.items()},
                               "max_visible": HJ4.ApiHazardJudgeV4.max_visible, "clip_words": A4.CLIP_WORDS}
    if o23.harm_severity_cap == "on":
        out["severity"] = {"cap": HSC.CAP, "component": HSC.COMPONENT, "purpose": HSC.PURPOSE, "model": HSC.MODEL,
                           "usd_cap": o23.harm_severity_usd_cap, "token_cap": HSC.TOKEN_CAP,
                           "rubric_sha256": HSC.RUBRIC_SHA256, "sha256": dict(HSC.SHA256),
                           "calibration": dict(HSC.CALIBRATION_IDS),
                           "rule": "acute & hazard JUDGE contraindicated & not degenerate: critical -> total = "
                                   "min(total, cap); minor / unjudged -> unchanged; outermost transform"}
    return out


def block(o23: argparse.Namespace, o22b: argparse.Namespace, rest: list[str]) -> dict:
    out = P22B.block(o22b, rest)
    out["base_prereg"] = P22B.PREREG
    out["prereg"] = PREREG
    out["h23"] = h23_block(o23)
    return json.loads(json.dumps(out, default=str))


def verify(pin: Path, o23: argparse.Namespace, o22b: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(o23, o22b, rest)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur23 pin (prereg={pre.get('prereg')!r})")
    bad = P21.diff(pre, now, BLOCKS)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur23 pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur23.py write --out PIN.json -- <main_v4hcur23 args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcur23 as M23
    o23, o22b, rest = M23.resolve(run)
    if o23.prereg_v4hcur23:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcur23")
    blk = block(o23, o22b, rest)
    if (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur23 pin needs --curriculum v22")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
