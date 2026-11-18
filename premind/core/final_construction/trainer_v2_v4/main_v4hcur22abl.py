#!/usr/bin/env python3
"""Entry of the judge-free CUR22B ablations A0-A6 (CUR22_ABLATIONS.md; prepared 2026-10-16, NOT LAUNCHED) = the frozen
main_v4hcur22b.main (curriculum v2.2, bonus v1.1, fa25, AA / AVR, v4wh_g), imported read-only, with:
  --abl-warning-credit off|cue   cue: the warning credit with the deterministic cue detector, no judge (abl_hooks.py);
                                 needs --warning-credit off (the paid judge mode is refused)
  --abl-hazard-first on|off      off: reward_v5 hazard_first (R_early) off, = the old hf-off preset
  --abl-stop-at-step N           hard stop after step N; --steps keeps the curriculum horizon (0 = none)
  --recall-decision / --crowdout-guard   the CUR22R recall terms (recall_decision.py, read-only), memory_call arm only
  --prereg-v4hcur22abl PIN       (pins_v4hcur22abl.py write) verified before anything runs
Judge-free: --hazard-backend must be cue and --warning-credit off, so no API client is ever built. No warm start
(--init-lora refused): A6 is the single-stage memory_call run from the base model. NEW file.
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
from trainer_v2_v4 import recall_decision as RD  # noqa: E402

logger = logging.getLogger(__name__)
FLAG = "--prereg-v4hcur22abl"
REFUSED = ("--prereg-v4hcur22b", "--prereg-v4hcur22b-9b", "--prereg-v4hcur22r", "--prereg-v4hcur23",
           "--prereg-v4hcur23-9b", "--prereg-v4hcur23r", "--init-lora", "--hazard-rubric", "--harm-severity-cap",
           "--harm-severity-usd-cap", "--hazard-judge-provider", "--hazard-judge-url", "--hazard-judge-model")


def _has(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def opt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--abl-warning-credit", default="off", choices=AH.WARNING_MODES)
    ap.add_argument("--abl-hazard-first", default="on", choices=AH.HF_MODES)
    ap.add_argument("--abl-stop-at-step", type=int, default=0)
    ap.add_argument("--recall-decision", default="off", choices=RD.DECISION_MODES)
    ap.add_argument("--crowdout-guard", default="off", choices=RD.GUARD_MODES)
    ap.add_argument(FLAG, default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, list[str]]:
    """(ablation options, main_v4hcur22b options, the main_v4hcur22 argv). Refuses anything that is not judge-free."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4hcur22b as M22B
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in REFUSED:
        if _has(argv, flag):
            raise SystemExit(f"REFUSED: {flag} is not accepted by main_v4hcur22abl (judge-free, no warm start, "
                             f"pinned by {FLAG} only)")
    o, rest = opt_parser().parse_known_args(argv)
    if (MH._value(rest, "--hazard-backend") or "cue") != "cue":
        raise SystemExit("REFUSED: judge-free ablation: --hazard-backend must be cue")
    if (MH._value(rest, "--warning-credit") or "off") != "off":
        raise SystemExit("REFUSED: judge-free ablation: --warning-credit must be off (use --abl-warning-credit cue)")
    if MH._value(rest, "--reward-preset") != "v4wh_g":
        raise SystemExit("REFUSED: the ablations are defined over --reward-preset v4wh_g (the CUR22B stack)")
    steps = int(MH._value(rest, "--steps") or 1)
    if o.abl_stop_at_step < 0 or (o.abl_stop_at_step and o.abl_stop_at_step >= steps):
        raise SystemExit(f"REFUSED: --abl-stop-at-step {o.abl_stop_at_step} must be in [1, --steps {steps})")
    recall = o.recall_decision != "off" or o.crowdout_guard != "off"
    if recall and MH._value(rest, "--arm") != "memory_call":
        raise SystemExit("REFUSED: --recall-decision / --crowdout-guard act on the memory_call arm only")
    o22b, rest22 = M22B.resolve(rest)
    if o22b.prereg_v4hcur22b:
        raise SystemExit(f"REFUSED: an ablation run is pinned by {FLAG}")
    return o, o22b, rest22


def stored(o: argparse.Namespace) -> dict:
    out = {}
    if o.abl_warning_credit != "off":
        out["warning_credit"] = o.abl_warning_credit
    if o.abl_hazard_first != "on":
        out["abl_hazard_first"] = o.abl_hazard_first
    if o.abl_stop_at_step:
        out["abl_stop_at_step"] = o.abl_stop_at_step
    out.update({k: getattr(o, k) for k in ("recall_decision", "crowdout_guard") if getattr(o, k) != "off"})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4hcur22b as M22B
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    o, _, _ = resolve(argv)
    run = [a for i, a in enumerate(argv) if not (a == FLAG or a.startswith(FLAG + "=")
                                                  or (i and argv[i - 1] == FLAG))]
    _, rest = opt_parser().parse_known_args(run)        # main_v4hcur22b argv (its own options left in)
    if o.prereg_v4hcur22abl:
        from trainer_v2_v4 import pins_v4hcur22abl as PA
        PA.verify(Path(o.prereg_v4hcur22abl), run)
        logger.info("v4hcur22abl pin verified: %s", o.prereg_v4hcur22abl)
    else:
        logger.warning("v4hcur22abl run without %s (CPU only; the launcher requires a pin)", FLAG)
    if "--print-config" in rest:
        print(json.dumps({"v4hcur22abl": {k: getattr(o, k) for k in ("abl_warning_credit", "abl_hazard_first",
                                                                     "abl_stop_at_step", "recall_decision",
                                                                     "crowdout_guard")}}, indent=1))
    else:
        RD.install(o.recall_decision, o.crowdout_guard)          # innermost, as main_v4hcur22r
        AH.install_hazard_first(o.abl_hazard_first)
        AH.install_warning_cue(o.abl_warning_credit, MH._value(rest, "--release"))
        AH.install_stop_at(o.abl_stop_at_step)
        AH.store_args(stored(o))
        logger.info("cur22abl: warning credit %s, hazard first %s, stop at %s, recall %s / %s",
                    o.abl_warning_credit, o.abl_hazard_first, o.abl_stop_at_step, o.recall_decision, o.crowdout_guard)
    return M22B.main(rest)


if __name__ == "__main__":
    sys.exit(main())
