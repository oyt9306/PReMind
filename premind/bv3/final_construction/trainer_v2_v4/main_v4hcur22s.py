#!/usr/bin/env python3
"""Entry of CUR22S (CUR22S.md; prepared 2026-10-16, NOT LAUNCHED): ONE stage that learns grounding and selective recall
together = CUR22B 4B (main_v4hcur22b.main, imported read-only: curriculum v2.2 with the same 200-step schedule, replay 80,
safe 0.30, ladder 2,2, bonus v1.1, fa25, AA / AVR, v4wh_g, hazard judge API + warning credit judge) with only:
  --arm memory_call (from the base model; --init-lora refused)
  --recall-decision ws52u1 --crowdout-guard credit_gate   the CUR22R3 recall terms (recall_decision_u1.py), installed
                                                           exactly as main_v4hcur22r3 does (innermost, + _store_args)
  --abl-stop-at-step 150   abl_hooks horizon stop (--steps stays 200 = the curriculum horizon); ckpt 150 is due anyway
  --prereg-v4hcur22s PIN   (pins_v4hcur22s.py write) verified before anything runs
= the judge-free ablation A6 (main_v4hcur22abl) with the judges kept. NEW file.
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

from trainer_v2_v4 import abl_hooks as AH  # noqa: E402
from trainer_v2_v4 import main_v4hcur23r as MR23  # noqa: E402
from trainer_v2_v4 import recall_decision_u1 as RDU  # noqa: E402

logger = logging.getLogger(__name__)
FLAG = "--prereg-v4hcur22s"
STOP_AT = 150
HORIZON = 200
GUARD = "credit_gate"
REFUSED = ("--prereg-v4hcur22b", "--prereg-v4hcur22b-9b", "--prereg-v4hcur22abl", "--prereg-v4hcur22r",
           "--prereg-v4hcur22r2", "--prereg-v4hcur22r3", "--prereg-v4hcur22r4", "--prereg-v4hcur23",
           "--prereg-v4hcur23-9b", "--prereg-v4hcur23r", "--init-lora", "--recall-diag-features", "--hazard-rubric",
           "--harm-severity-cap", "--harm-severity-usd-cap", "--abl-warning-credit", "--abl-hazard-first")
# CUR22B values this entry pins (any other value is a different recipe)
FIXED = {"--arm": "memory_call", "--model": "4b", "--steps": str(HORIZON), "--reward-preset": "v4wh_g",
         "--curriculum": "v22", "--hazard-backend": "judge", "--hazard-judge-provider": "api",
         "--hazard-judge-model": "gpt-6-luna@medium", "--warning-credit": "judge"}


def _has(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def opt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--recall-decision", default="off", choices=RDU.DECISION_MODES)
    ap.add_argument("--crowdout-guard", default="off", choices=RDU.GUARD_MODES)
    ap.add_argument("--abl-stop-at-step", type=int, default=0)
    ap.add_argument(FLAG, default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, list[str]]:
    """(CUR22S options, main_v4hcur22b options, the main_v4hcur22 argv); refuses anything that is not CUR22S."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4hcur22b as M22B
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in REFUSED:
        if _has(argv, flag):
            raise SystemExit(f"REFUSED: {flag} is not accepted by main_v4hcur22s (single stage from the base model, "
                             f"pinned by {FLAG} only)")
    o, rest = opt_parser().parse_known_args(argv)
    if (o.recall_decision, o.crowdout_guard) != (RDU.MODE, GUARD):
        raise SystemExit(f"REFUSED: CUR22S needs --recall-decision {RDU.MODE} --crowdout-guard {GUARD}")
    if o.abl_stop_at_step != STOP_AT:
        raise SystemExit(f"REFUSED: CUR22S needs --abl-stop-at-step {STOP_AT} (owner rule)")
    for flag, want in FIXED.items():
        if MH._value(rest, flag) != want:
            raise SystemExit(f"REFUSED: CUR22S needs {flag} {want} (got {MH._value(rest, flag)!r})")
    o22b, rest22 = M22B.resolve(rest)
    if o22b.prereg_v4hcur22b:
        raise SystemExit(f"REFUSED: a CUR22S run is pinned by {FLAG}")
    return o, o22b, rest22


def strip_pin(argv: list[str]) -> list[str]:
    return [a for i, a in enumerate(argv) if not (a == FLAG or a.startswith(FLAG + "=") or (i and argv[i - 1] == FLAG))]


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4hcur22b as M22B
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    o, _, _ = resolve(argv)
    run = strip_pin(argv[1:] if argv[:1] == ["--"] else argv)
    _, rest = opt_parser().parse_known_args(run)        # main_v4hcur22b argv (its own options left in)
    if o.prereg_v4hcur22s:
        from trainer_v2_v4 import pins_v4hcur22s as PS
        PS.verify(Path(o.prereg_v4hcur22s), run)
        logger.info("v4hcur22s pin verified: %s", o.prereg_v4hcur22s)
    else:
        logger.warning("v4hcur22s run without %s (CPU only; the launcher requires a pin)", FLAG)
    if "--print-config" in rest:
        print(json.dumps({"v4hcur22s": {"recall_decision": o.recall_decision, "weights": RDU.WEIGHTS,
                                        "crowdout_guard": o.crowdout_guard, "stop_at_step": o.abl_stop_at_step,
                                        "init_lora": None}}, indent=1))
    else:
        RDU.install(o.recall_decision, o.crowdout_guard)   # innermost, before main_v4hcur22b installs (= CUR22R3)
        MR23._store_args(o)                                # v4h_recall_decision / v4h_crowdout_guard (= CUR22R3)
        AH.install_stop_at(o.abl_stop_at_step)
        AH.store_args({"abl_stop_at_step": o.abl_stop_at_step})
        logger.info("cur22s: recall decision %s (missed %sr, unneeded %sr), crowd-out guard %s, stop at %d of %d, "
                    "no init LoRA", o.recall_decision, RDU.MISSED_CALL, RDU.UNNEEDED_CALL, o.crowdout_guard,
                    o.abl_stop_at_step, HORIZON)
    return M22B.main(rest)


if __name__ == "__main__":
    sys.exit(main())
