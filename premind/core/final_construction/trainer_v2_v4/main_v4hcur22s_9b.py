#!/usr/bin/env python3
"""Entry of CUR22S on Qwen3.5-9B (CUR22S_9B.md; prepared 2026-10-16, NOT LAUNCHED) = main_v4hcur22s (frozen, imported
read-only: same refusals, same fixed CUR22B values, same hook install order) with only:
  --model 9b                   (main_v4hcur22s pins --model 4b; everything else in its FIXED table is kept)
  --prereg-v4hcur22s-9b PIN    (pins_v4hcur22s_9b.py write) = the CUR22S block + m9b, verified before anything runs
The 4B pin flag --prereg-v4hcur22s is refused. main_v4hcur22b.main then logs "run without --prereg-v4hcur22b":
expected, the 9B pin covers every CUR22B block. NEW file.
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
from trainer_v2_v4 import main_v4hcur22s as MS  # noqa: E402
from trainer_v2_v4 import main_v4hcur23r as MR23  # noqa: E402
from trainer_v2_v4 import recall_decision_u1 as RDU  # noqa: E402

logger = logging.getLogger(__name__)
FLAG = "--prereg-v4hcur22s-9b"
MODEL = "9b"
REFUSED = MS.REFUSED + (MS.FLAG,)
FIXED = {**MS.FIXED, "--model": MODEL}


def split(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """(9B pin path or None, the argv without the 9B pin flag)."""
    if argv[:1] == ["--"]:
        argv = argv[1:]
    pin, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == FLAG and i + 1 < len(argv):
            pin, i = argv[i + 1], i + 2
            continue
        if a.startswith(FLAG + "="):
            pin, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    return pin, rest


def resolve(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, list[str]]:
    """= main_v4hcur22s.resolve with --model 9b (argv without the 9B pin flag)."""
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4hcur22b as M22B
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in REFUSED + (FLAG,):
        if MS._has(argv, flag):
            raise SystemExit(f"REFUSED: {flag} is not accepted by main_v4hcur22s_9b (pinned args; 4B pin flag; "
                             f"warm start / judge-free flags)")
    o, rest = MS.opt_parser().parse_known_args(argv)
    if (o.recall_decision, o.crowdout_guard) != (RDU.MODE, MS.GUARD):
        raise SystemExit(f"REFUSED: CUR22S needs --recall-decision {RDU.MODE} --crowdout-guard {MS.GUARD}")
    if o.abl_stop_at_step != MS.STOP_AT:
        raise SystemExit(f"REFUSED: CUR22S needs --abl-stop-at-step {MS.STOP_AT} (owner rule)")
    for flag, want in FIXED.items():
        if MH._value(rest, flag) != want:
            raise SystemExit(f"REFUSED: CUR22S_9B needs {flag} {want} (got {MH._value(rest, flag)!r})")
    o22b, rest22 = M22B.resolve(rest)
    if o22b.prereg_v4hcur22b:
        raise SystemExit(f"REFUSED: a CUR22S_9B run is pinned by {FLAG}")
    return o, o22b, rest22


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4hcur22b as M22B
    logging.basicConfig(level=logging.INFO)
    pin, run = split(list(sys.argv[1:] if argv is None else argv))
    o, _, _ = resolve(run)
    _, rest = MS.opt_parser().parse_known_args(run)    # main_v4hcur22b argv (its own options left in)
    if pin:
        from trainer_v2_v4 import pins_v4hcur22s_9b as P9S
        P9S.verify(Path(pin), run)
        logger.info("v4hcur22s_9b pin verified (CUR22S blocks + m9b): %s", pin)
    else:
        logger.warning("v4hcur22s_9b run without %s (CPU only; the launcher requires a pin)", FLAG)
    if "--print-config" in rest:
        print(json.dumps({"v4hcur22s_9b": {"recall_decision": o.recall_decision, "weights": RDU.WEIGHTS,
                                           "crowdout_guard": o.crowdout_guard, "stop_at_step": o.abl_stop_at_step,
                                           "init_lora": None, "model": MODEL}}, indent=1))
    else:
        RDU.install(o.recall_decision, o.crowdout_guard)   # innermost, before main_v4hcur22b installs (= CUR22S)
        MR23._store_args(o)
        AH.install_stop_at(o.abl_stop_at_step)
        AH.store_args({"abl_stop_at_step": o.abl_stop_at_step})
        logger.info("cur22s_9b: recall decision %s (missed %sr, unneeded %sr), crowd-out guard %s, stop at %d of %d, "
                    "no init LoRA, model %s", o.recall_decision, RDU.MISSED_CALL, RDU.UNNEEDED_CALL, o.crowdout_guard,
                    o.abl_stop_at_step, MS.HORIZON, MODEL)
    return M22B.main(rest)


if __name__ == "__main__":
    sys.exit(main())
