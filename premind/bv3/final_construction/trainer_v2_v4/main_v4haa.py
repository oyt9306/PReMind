#!/usr/bin/env python3
"""PReMind v2 entry point + the acute target-absent clarify fix (owner 2026-10-14, pilot F = pilot C2 (ABSENT_SHARE
0.20), ACUTE_ABSENT keep, ``--acute-absent-clarify accept``, ``--absent-visible-risk on``, ``--warning-credit judge``).

= trainer_v2_v4.main_v4h, unchanged, with in-process only (no pinned file is edited):
  reward_v4hg.score_episode -> acute_absent_fix.apply (``accept``) and absent_visible_risk.apply (``on``) on top;
                               ``off`` (both defaults) installs nothing.
  grpo_v2.rollout / train_step + reward_v4hg.score_episode -> warning_credit (``--warning-credit judge``).
Options (all others are main_v4h's; GPU jobs are launched by the orchestrator only, CPU: ``--print-config``):
  --acute-absent-clarify off|accept   accept: on acute gt_absent withhold rows a clarify label is action-correct iff
                                      the v4h hazard state is "specific" (acute_absent_fix.py)
  --absent-visible-risk off|on       on: risk / false_alarm / missed_warning stay ON on gt_absent rows whose image
                                      shows a visible non-acute hazard (absent_visible_risk.py)
  --warning-credit off|judge         judge: risk credit = tag unsafe OR a specific warning (Luna judge, cue
                                      fallback; warning_credit.py / warning_reward.py)
  --warning-usd-cap USD              the warning judge's own run cap (default 3; independent of --hazard-usd-cap)
  --prereg-v4haa PIN                  the run pin (pins_v4haa.py write), verified before anything runs
``--prereg-v4h`` is refused (the v4haa pin contains the v4h pin block). accept / on need a v4wh preset (the hazard state / the hook)
and ``--acute-absent keep`` (mask removes exactly these rows from the training pool).
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

from trainer_v2_v4 import absent_visible_risk as AVR  # noqa: E402
from trainer_v2_v4 import acute_absent_fix as AF  # noqa: E402
from trainer_v2_v4 import warning_credit as WC  # noqa: E402

logger = logging.getLogger(__name__)
AA_KEYS = ("acute_absent_clarify", "absent_visible_risk", "warning_credit")
OWNED = ("--prereg-v4h",)


def aa_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--acute-absent-clarify", default="off", choices=AF.MODES)
    ap.add_argument("--absent-visible-risk", default="off", choices=AVR.MODES)
    ap.add_argument("--warning-credit", default="off", choices=WC.MODES)
    ap.add_argument("--warning-usd-cap", type=float, default=WC.USD_CAP)
    ap.add_argument("--prereg-v4haa", default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """(v4haa options, the main_v4h argv). Refuses options owned here and inconsistent combinations."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import reward_v4hg as RHG
    if argv[:1] == ["--"]:
        argv = argv[1:]
    aa, rest = aa_parser().parse_known_args(argv)
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in rest):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4haa (use --prereg-v4haa)")
    if aa.acute_absent_clarify == "accept" or aa.absent_visible_risk == "on" or aa.warning_credit != "off":
        RHG.register()
        if not RHG.is_hazard_preset(MH._value(rest, "--reward-preset") or ""):
            raise SystemExit("REFUSED: --acute-absent-clarify / --absent-visible-risk / --warning-credit need a v4wh "
                             "preset (patch_h scores through reward_v4hg)")
    if aa.warning_credit != "off" and not aa.warning_usd_cap > 0:
        raise SystemExit("REFUSED: --warning-usd-cap must be positive")
    if aa.acute_absent_clarify == "accept":
        if (MH._value(rest, "--acute-absent") or "keep") != "keep":
            raise SystemExit("REFUSED: --acute-absent-clarify accept needs --acute-absent keep")
    return aa, rest


def _store_args(aa: argparse.Namespace) -> None:
    """Non-default values join the checkpoint args (v4h_<key>); off leaves them exactly as main_v4h."""
    on = {k: getattr(aa, k) for k in AA_KEYS if getattr(aa, k) != "off"}
    if aa.warning_credit != "off":
        on["warning_usd_cap"] = aa.warning_usd_cap
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
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    aa, rest = resolve(argv)
    from trainer_v2_v4 import main_v4h as MH
    MH.no_abbrev()
    MH.release_pin_presets()
    if aa.prereg_v4haa:
        from trainer_v2_v4 import pins_v4haa as PAA
        PAA.verify(Path(aa.prereg_v4haa), aa, rest)
        logger.info("v4haa pin verified: %s", aa.prereg_v4haa)
    else:
        logger.warning("v4haa run without --prereg-v4haa (allowed for smoke / CPU only; the launcher requires a pin)")
    if "--print-config" in rest:
        print(json.dumps({"v4haa": {k: getattr(aa, k) for k in AA_KEYS}}, indent=1))
    else:
        AF.install(aa.acute_absent_clarify)
        AVR.install(aa.absent_visible_risk)
        if aa.warning_credit != "off":       # outermost: every fix above scores the effective <risk> tag
            WC.install(aa.warning_credit, release=Path(MH._value(rest, "--release")),
                       out=Path(MH._value(rest, "--out") or "."), usd_cap=aa.warning_usd_cap)
        _store_args(aa)
    return MH.main(rest)


if __name__ == "__main__":
    sys.exit(main())
