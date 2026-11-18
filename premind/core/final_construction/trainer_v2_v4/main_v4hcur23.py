#!/usr/bin/env python3
"""PReMind v2 entry point of the next Stage-A run CUR23 = main_v4hcur22b's stack (curriculum v2.2 ``--safe-min-frac``,
replay budget, acute memory bonus v1.1, ``--false-alarm-pen`` fa25|off) + two reward hooks (NEW file; the CUR21B /
CUR22B entries are imported read-only):
  --hazard-rubric v3|v4          v4: the in-loop hazard judge grades with the eval's acute adequacy rubric v4
                                 (hazard_judge_v4.py); v3 (default) = unchanged
  --harm-severity-cap off|on     on: critical contraindicated acute rollouts capped at total <= -0.30
                                 (harm_severity_cap.py, severity v2 judge)
  --harm-severity-usd-cap USD    the severity judge's own run cap (default 1.0; split over DDP ranks)
  --prereg-v4hcur23 PIN          (pins_v4hcur23.py write), verified before anything runs; older pin flags are refused.
Both hooks need --reward-preset v4wh_g --hazard-backend judge --hazard-judge-provider api.
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

from trainer_v2_v4 import harm_severity_cap as HSC  # noqa: E402
from trainer_v2_v4 import hazard_judge_v4 as HJ4  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4hcur22b", "--prereg-v4hcur22", "--prereg-v4hcur21b", "--prereg-v4hcur21", "--prereg-v4hcurb",
         "--prereg-v4hcur2", "--prereg-v4hcur", "--prereg-v4haa", "--prereg-v4h")
H23_KEYS = ("hazard_rubric", "harm_severity_cap", "harm_severity_usd_cap")


def opt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--hazard-rubric", default="v3", choices=HJ4.MODES)
    ap.add_argument("--harm-severity-cap", default="off", choices=HSC.MODES)
    ap.add_argument("--harm-severity-usd-cap", type=float, default=HSC.USD_CAP)
    ap.add_argument("--prereg-v4hcur23", default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, list[str]]:
    """(cur23 options, main_v4hcur22b options, the main_v4hcur22 argv)."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4hcur22b as M22B
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcur23 (use --prereg-v4hcur23)")
    o23, rest = opt_parser().parse_known_args(argv)
    o22b, rest = M22B.resolve(rest)
    if o23.hazard_rubric != "v3" or o23.harm_severity_cap != "off":
        need = {"--reward-preset": "v4wh_g", "--hazard-backend": "judge", "--hazard-judge-provider": "api"}
        bad = [f"{k} {v}" for k, v in need.items() if (MH._value(rest, k) or ("api" if k.endswith("provider")
                                                                                else None)) != v]
        if bad:
            raise SystemExit(f"REFUSED: --hazard-rubric v4 / --harm-severity-cap on need {', '.join(bad)}")
    if o23.harm_severity_cap == "on" and not o23.harm_severity_usd_cap > 0:
        raise SystemExit("REFUSED: --harm-severity-usd-cap must be positive")
    return o23, o22b, rest


def _store_args(o23: argparse.Namespace) -> None:
    """Non-default values join the checkpoint args (v4h_<key>); defaults leave them exactly as main_v4hcur22b."""
    on = {}
    if o23.hazard_rubric != "v3":
        on["hazard_rubric"] = o23.hazard_rubric
    if o23.harm_severity_cap != "off":
        on.update(harm_severity_cap=o23.harm_severity_cap, harm_severity_usd_cap=o23.harm_severity_usd_cap)
    if not on:
        return
    from trainer_v2_v4 import main_v4 as MV4
    orig = MV4.parse

    def parse(argv=None):
        args, pc = orig(argv)
        for k, v in on.items():
            setattr(args, f"v4h_{k}", v)
        return args, pc
    MV4.parse = parse


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import acute_memory_bonus_v1_1 as AMB
    from trainer_v2_v4 import false_alarm_pen as FA
    from trainer_v2_v4 import main_v4hcur22 as BASE
    from trainer_v2_v4 import main_v4hcurb as MB
    logging.basicConfig(level=logging.INFO)
    o23, o22b, rest = resolve(list(sys.argv[1:] if argv is None else argv))
    if o23.prereg_v4hcur23:
        from trainer_v2_v4 import pins_v4hcur23 as P23
        P23.verify(Path(o23.prereg_v4hcur23), o23, o22b, rest)
        logger.info("v4hcur23 pin verified: %s", o23.prereg_v4hcur23)
    else:
        logger.warning("v4hcur23 run without --prereg-v4hcur23 (CPU only; the launcher requires a pin)")
    if "--print-config" in rest:
        print(json.dumps({"v4hcur23": {**{k: getattr(o23, k) for k in H23_KEYS},
                                       "hazard_rubric_sha256": HJ4.RUBRIC_V4_SHA256 if o23.hazard_rubric == "v4"
                                       else None, "severity_cap": HSC.CAP,
                                       "acute_memory_bonus": o22b.acute_memory_bonus,
                                       "false_alarm_pen": o22b.false_alarm_pen}}, indent=1))
    else:
        HJ4.install(o23.hazard_rubric)                 # before main_v4h.install builds the hazard judge
        HSC.hook_main_v4h(o23.harm_severity_cap, o23.harm_severity_usd_cap)   # outermost, after warning credit
        AMB.install(o22b.acute_memory_bonus)
        FA.install(o22b.false_alarm_pen)
        MB._store_args(o22b)                           # read-only reuse: on -> args.v4h_acute_memory_bonus
        _store_args(o23)
        logger.info("cur23: hazard rubric %s, severity cap %s, bonus %s %s, false alarm pen %s", o23.hazard_rubric,
                    o23.harm_severity_cap, AMB.VERSION, o22b.acute_memory_bonus, o22b.false_alarm_pen)
    return BASE.main(rest)


if __name__ == "__main__":
    sys.exit(main())
