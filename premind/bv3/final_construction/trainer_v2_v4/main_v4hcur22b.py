#!/usr/bin/env python3
"""PReMind v2 entry point for the v14 fallback (owner-approved 2026-10-15) = main_v4hcur22 (curriculum v2.2,
``--safe-min-frac``) + acute memory bonus v1.1 (``--acute-memory-bonus on``) + the stronger false-alarm penalty
(false_alarm_pen.py, ``--false-alarm-pen fa25`` = v4wh_g with PEN false_alarm -0.25). NEW file.
  --prereg-v4hcur22b PIN  (pins_v4hcur22b.py write), verified before anything runs; older pin flags are refused.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import acute_memory_bonus_v1_1 as AMB  # noqa: E402
from trainer_v2_v4 import false_alarm_pen as FA  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4hcur22", "--prereg-v4hcur21b", "--prereg-v4hcur21", "--prereg-v4hcurb", "--prereg-v4hcur2",
         "--prereg-v4hcur", "--prereg-v4haa", "--prereg-v4h")


def opt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--acute-memory-bonus", default="off", choices=AMB.MODES)
    ap.add_argument("--false-alarm-pen", default="off", choices=FA.MODES)
    ap.add_argument("--prereg-v4hcur22b", default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import reward_v4hg as RHG
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcur22b (use --prereg-v4hcur22b)")
    opt, rest = opt_parser().parse_known_args(argv)
    if opt.acute_memory_bonus == "on" or opt.false_alarm_pen != "off":
        RHG.register()
        if MH._value(rest, "--reward-preset") != "v4wh_g":
            raise SystemExit("REFUSED: --acute-memory-bonus / --false-alarm-pen are defined over --reward-preset v4wh_g")
    return opt, rest


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4hcur22 as BASE
    from trainer_v2_v4 import main_v4hcurb as MB
    logging.basicConfig(level=logging.INFO)
    opt, rest = resolve(list(sys.argv[1:] if argv is None else argv))
    if opt.prereg_v4hcur22b:
        from trainer_v2_v4 import pins_v4hcur22b as PB
        PB.verify(Path(opt.prereg_v4hcur22b), opt, rest)
        logger.info("v4hcur22b pin verified: %s", opt.prereg_v4hcur22b)
    else:
        logger.warning("v4hcur22b run without --prereg-v4hcur22b (CPU only; the launcher requires a pin)")
    if "--print-config" in rest:
        print(json.dumps({"v4hcur22b": {"acute_memory_bonus": opt.acute_memory_bonus, "bonus_version": AMB.VERSION,
                                        "false_alarm_pen": opt.false_alarm_pen,
                                        "false_alarm_value": FA.VALUE.get(opt.false_alarm_pen, FA.BASE)}}, indent=1))
    else:
        AMB.install(opt.acute_memory_bonus)
        FA.install(opt.false_alarm_pen)
        MB._store_args(opt)                      # read-only reuse: on -> args.v4h_acute_memory_bonus = "on"
        logger.info("bonus %s %s; false alarm pen %s", AMB.VERSION, opt.acute_memory_bonus, opt.false_alarm_pen)
    return BASE.main(rest)


if __name__ == "__main__":
    sys.exit(main())
