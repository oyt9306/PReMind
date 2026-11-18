"""Pin of a v4hcur21 run (main_v4hcur21, curriculum v2.1): the v4haa pin block (pins_v4haa.block, unchanged) + the
``cur`` block (as pins_v4hcur2: schedule sha256, quotas, seed, horizon, re-queue rule, replay rule, release / key-action
sha256; plan Plan21: acute x1 + ``--replay-budget``, pinned in quotas and replay.budget) + the v2.1 runtime rule (main_v4hcur21.runtime_rule) and the sha256 of the curriculum code (curriculum_v1.py,
curriculum_v2.py, curriculum_v2_1.py, main_v4hcur.py, main_v4hcur21.py, this file).
  python -I -m trainer_v2_v4.pins_v4hcur21 write --out PIN.json -- <main_v4hcur21 args>   (write once; PIN_WORLD=<ranks>)
``verify`` (main_v4hcur21 --prereg-v4hcur21 PIN) refuses on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

PREREG = "PREREG_TRAINER_V2_V4HCUR21"
CODE_CUR = ("curriculum_v1.py", "curriculum_v2.py", "curriculum_v2_1.py", "main_v4hcur.py", "main_v4hcur21.py",
            "pins_v4hcur21.py")
BLOCKS = ("v4h", "aa", "cur")


def _sha(p: Path) -> Optional[str]:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def cur_block(cur: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import curriculum_v1 as CV
    from trainer_v2_v4 import curriculum_v2 as C2
    from trainer_v2_v4 import main_v4hcur21 as M21
    out: dict = {"code_sha256": {f: _sha(HERE / f) for f in CODE_CUR}, "curriculum": cur.curriculum}
    if cur.curriculum == "off":
        return out
    plan, ids, data, _ = M21.build_plan(rest, cur.replay_budget)
    plan.check(ids)                                      # a pin is never written for a schedule failing a check
    out.update(schedule_sha256=plan.sha, quotas=plan.quotas, seed=plan.seed, horizon_steps=plan.steps,
               p_global=plan.p_global, n_slots=len(plan.schedule), requeue=CV.REQUEUE, min_gap_steps=CV.MIN_GAP_STEPS,
               replay={**C2.REPLAY, "mastery": C2.MASTERY, "max_visits": C2.MAX_VISITS,
                       "start_step": C2.REPLAY_START_STEP, "budget": cur.replay_budget},
               runtime=M21.runtime_rule(), data={k: v for k, v in data.items() if k != "out"})
    return out


def block(cur: argparse.Namespace, rest: list[str]) -> dict:
    """``rest`` = main_v4hcur21.resolve(...)[1] (main_v4haa args without a pin flag)."""
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import pins_v4haa as PAA
    aa, rest_aa = MA.resolve(rest)
    out = PAA.block(aa, rest_aa)
    out["prereg"] = PREREG
    out["cur"] = json.loads(json.dumps(cur_block(cur, rest), default=str))
    return out


def diff(pre: dict, now: dict, blocks: tuple = BLOCKS) -> list[str]:
    bad = [k for k in sorted(set(pre) | set(now)) if k not in blocks and pre.get(k) != now.get(k)]
    for blk in blocks:
        a, b = pre.get(blk) or {}, now.get(blk) or {}
        bad += [f"{blk}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return bad


def verify(pin: Path, cur: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), json.loads(json.dumps(block(cur, rest), default=str))
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur21 pin (prereg={pre.get('prereg')!r})")
    bad = diff(pre, now)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur21 pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur21.py write --out PIN.json -- <main_v4hcur21 args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcur21 as M21
    cur, rest = M21.resolve(run)
    if cur.prereg_v4hcur21:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcur21")
    blk = block(cur, rest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
