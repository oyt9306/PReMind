"""Pin of a v4hcur22 run (main_v4hcur22, curriculum v2.2): pins_v4hcur21's layout (v4haa block + ``cur`` block) with
the Plan22 quotas (incl. safe_min_frac and the moved slots), the runtime rule and the sha256 of the curriculum code
(curriculum_v1/v2/v2_1/v2_2, main_v4hcur, main_v4hcur21 (its hooks are reused), main_v4hcur22, this file).
  python -I -m trainer_v2_v4.pins_v4hcur22 write --out PIN.json -- <main_v4hcur22 args>   (write once; PIN_WORLD=<ranks>)
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

PREREG = "PREREG_TRAINER_V2_V4HCUR22"
CODE_CUR = ("curriculum_v1.py", "curriculum_v2.py", "curriculum_v2_1.py", "curriculum_v2_2.py", "main_v4hcur.py",
            "main_v4hcur21.py", "main_v4hcur22.py", "pins_v4hcur22.py")


def cur_block(cur: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import curriculum_v1 as CV
    from trainer_v2_v4 import curriculum_v2 as C2
    from trainer_v2_v4 import main_v4hcur22 as M22
    out: dict = {"code_sha256": {f: P21._sha(HERE / f) for f in CODE_CUR}, "curriculum": cur.curriculum}
    if cur.curriculum == "off":
        return out
    plan, ids, data, _ = M22.build_plan(rest, cur)
    plan.check(ids)
    out.update(schedule_sha256=plan.sha, quotas=plan.quotas, seed=plan.seed, horizon_steps=plan.steps,
               p_global=plan.p_global, n_slots=len(plan.schedule), requeue=CV.REQUEUE, min_gap_steps=CV.MIN_GAP_STEPS,
               replay={**C2.REPLAY, "mastery": C2.MASTERY, "max_visits": C2.MAX_VISITS,
                       "start_step": C2.REPLAY_START_STEP, "budget": plan.quotas["replay_budget"]},
               runtime=M22.runtime_rule(cur), data={k: v for k, v in data.items() if k != "out"})
    return out


def block(cur: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import pins_v4haa as PAA
    aa, rest_aa = MA.resolve(rest)
    out = PAA.block(aa, rest_aa)
    out["prereg"] = PREREG
    out["cur"] = json.loads(json.dumps(cur_block(cur, rest), default=str))
    return out


def verify(pin: Path, cur: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), json.loads(json.dumps(block(cur, rest), default=str))
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22 pin (prereg={pre.get('prereg')!r})")
    bad = P21.diff(pre, now)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22 pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22.py write --out PIN.json -- <main_v4hcur22 args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4hcur22 as M22
    cur, rest = M22.resolve(run)
    if cur.prereg_v4hcur22:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcur22")
    blk = block(cur, rest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
