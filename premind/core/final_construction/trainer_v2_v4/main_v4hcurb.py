#!/usr/bin/env python3
"""PReMind v2 entry point = main_v4hcur (curriculum + pilot-F flags) + the acute memory-informed response bonus
(acute_memory_bonus.py, owner request 2026-10-15). NEW file: main_v4hcur / main_v4haa / every pinned file untouched.

  reward_v4hg.score_episode -> acute_memory_bonus.apply (``--acute-memory-bonus on``; ``off`` installs nothing, i.e.
                               the run is exactly the base entry). Installed BEFORE the base entry's own hooks, so
                               acute_absent_fix / absent_visible_risk / warning_credit wrap it (warning credit stays
                               outermost; the three fixes act on disjoint rows: gt_absent vs gt_present).
Options (all others go to the base entry unchanged; GPU jobs are launched by the orchestrator only, CPU:
``--print-config``):
  --acute-memory-bonus off|on   on: + BONUS (0.06) on acute gt_present rows when hazard state specific, not
                                contraindicated AND personal_content at full weight
  --prereg-v4hcurb PIN          the run pin (pins_v4hcurb.py write), verified before anything runs
``--prereg-v4hcur`` / ``--prereg-v4haa`` / ``--prereg-v4h`` are refused (the v4hcurb pin contains all their blocks).
``on`` needs a v4wh preset (the hazard state lives in the v4h ledger block).

Composing with another curriculum entry (e.g. a future main_v4hcur2 + pins_v4hcur2 with the same resolve / main /
block / verify interface): a 3-line file ``main_v4hcur2b.py`` =
    from trainer_v2_v4 import main_v4hcur2 as BASE, pins_v4hcur2 as BASE_PINS, main_v4hcurb as B
    if __name__ == "__main__": raise SystemExit(B.run(BASE, BASE_PINS, sys.argv[1:]))
or, inside that entry's own main(), the single line ``acute_memory_bonus.install(mode)`` before its base main().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import acute_memory_bonus as AMB  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4hcur", "--prereg-v4haa", "--prereg-v4h")


def amb_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--acute-memory-bonus", default="off", choices=AMB.MODES)
    ap.add_argument("--prereg-v4hcurb", default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """(bonus options, the base-entry argv). Refuses owned options and a non-v4wh preset with the bonus on."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import reward_v4hg as RHG
    if argv[:1] == ["--"]:
        argv = argv[1:]
    amb, rest = amb_parser().parse_known_args(argv)
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in rest):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcurb (use --prereg-v4hcurb)")
    if amb.acute_memory_bonus == "on":
        RHG.register()
        if not RHG.is_hazard_preset(MH._value(rest, "--reward-preset") or ""):
            raise SystemExit("REFUSED: --acute-memory-bonus on needs a v4wh preset (hazard state in the v4h ledger)")
    return amb, rest


def _store_args(amb: argparse.Namespace) -> None:
    """``on`` joins the checkpoint args as v4h_acute_memory_bonus; ``off`` leaves them exactly as the base entry."""
    if amb.acute_memory_bonus == "off":
        return
    from trainer_v2_v4 import main_v4 as MV4
    orig = MV4.parse

    def parse(argv=None):
        args, pc = orig(argv)
        args.v4h_acute_memory_bonus = amb.acute_memory_bonus
        return args, pc
    MV4.parse = parse


def run(base: ModuleType, base_pins: ModuleType, argv: list[str]) -> int:
    """Bonus options handled here, then ``base.main(rest)`` (base = main_v4hcur, or any entry with its interface)."""
    logging.basicConfig(level=logging.INFO)
    amb, rest = resolve(list(argv))
    if amb.prereg_v4hcurb:
        from trainer_v2_v4 import pins_v4hcurb as PB
        PB.verify(Path(amb.prereg_v4hcurb), amb, rest, base=base, base_pins=base_pins)
        logger.info("v4hcurb pin verified: %s", amb.prereg_v4hcurb)
    else:
        logger.warning("v4hcurb run without --prereg-v4hcurb (allowed for smoke / CPU only; the launcher requires a pin)")
    if "--print-config" in rest:
        print(json.dumps({"v4hcurb": {"acute_memory_bonus": amb.acute_memory_bonus, "bonus": AMB.BONUS,
                                      "criterion": AMB.CRITERION, "base": base.__name__}}, indent=1))
    else:
        AMB.install(amb.acute_memory_bonus)
        _store_args(amb)
        logger.info("acute memory bonus: %s (+%s)", amb.acute_memory_bonus, AMB.BONUS)
    return base.main(rest)


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4hcur as MC
    from trainer_v2_v4 import pins_v4hcur as PC
    return run(MC, PC, list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
