"""Pin of a v4haa run (main_v4haa): the v4h pin block (pins_v4h.block, unchanged) + the ``aa`` block: sha256 of the fix
code (acute_absent_fix.py, absent_visible_risk.py, main_v4haa.py, this file), the ``--acute-absent-clarify`` and
``--absent-visible-risk`` values, the action weight paid,
the acceptance criterion, the ``--warning-credit`` block (backend, model, own USD cap,
warning_reward / wd_* sha256), and the run's ``--target-absent-share`` / ``--acute-absent`` values (the base v4 pin does not
record the absent share).
  python -I -m trainer_v2_v4.pins_v4haa write --out PIN.json -- <main_v4haa args>   (write once; PIN_WORLD=<ranks>)
``verify`` (main_v4haa --prereg-v4haa PIN) refuses on any difference.
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

PREREG = "PREREG_TRAINER_V2_V4HAA"
CODE_AA = ("acute_absent_fix.py", "absent_visible_risk.py", "warning_credit.py", "warning_reward.py", "main_v4haa.py", "pins_v4haa.py")
CRITERION = {"acute_absent_clarify": "gt_absent & withhold target & image-acute & clarify & v4h.state == specific & not "
                                     "contraindicated"
                                     " -> action = W[action]",
             "absent_visible_risk": "gt_absent & risk group masked (missing fact) & image severity minor|acute or "
                                    "evidence -> risk / false_alarm / missed_warning scored as reward_v5._terms"}


def _sha(p: Path) -> Optional[str]:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


WD_FILES = ("warning_detector.py", "wd_cue.py", "wd_judge.py", "wd_reference.py")


def warning_block(aa: argparse.Namespace, base_rest: list[str]) -> Optional[dict]:
    """--warning-credit: backend, model, own USD cap, per-step limits, detector versions and code / reference sha256
    (None when off, so an off pin refuses an on run and vice versa)."""
    if aa.warning_credit == "off":
        return None
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import warning_credit as WC
    from trainer_v2_v4 import warning_reward as WR
    J = WR.J
    rel = Path(MH._value(base_rest, "--release") or "")
    return {"mode": aa.warning_credit, "backend": "judge", "fallback": "cue", "model": WC.MODEL,
            "usd_cap": aa.warning_usd_cap, "max_judge_calls_per_step": WR.MAX_JUDGE_CALLS_PER_STEP,
            "step_budget_s": WR.STEP_BUDGET_S, "judge_version": J.JUDGE_VERSION, "rubric_sha256": J.RUBRIC_SHA256,
            "wd_sha256": {f: _sha(WR._WJ / f) for f in WD_FILES},
            "refs_sha256": {f: _sha(rel / f) for f in ("train.jsonl", "dev.jsonl")}}


def block(aa: argparse.Namespace, rest: list[str]) -> dict:
    """``rest`` = main_v4haa.resolve(...)[1]."""
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import pins_v4h as PV
    MH.no_abbrev()
    h, base_rest, _ = MH.resolve(rest)
    out = PV.block(h, base_rest)
    out["prereg"] = PREREG
    out["aa"] = {"code_sha256": {f: _sha(HERE / f) for f in CODE_AA},
                 "config": {k: getattr(aa, k) for k in MA.AA_KEYS},
                 "target_absent_share": MH._value(base_rest, "--target-absent-share"), "acute_absent": h.acute_absent,
                 "action_weight": R5.W["action"], "criterion": CRITERION, "warning": warning_block(aa, base_rest)}
    return out


def verify(pin: Path, aa: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), json.loads(json.dumps(block(aa, rest), default=str))
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4haa pin (prereg={pre.get('prereg')!r})")
    bad = [k for k in sorted(set(pre) | set(now)) if k not in ("v4h", "aa") and pre.get(k) != now.get(k)]
    for blk in ("v4h", "aa"):
        a, b = pre.get(blk) or {}, now.get(blk) or {}
        bad += [f"{blk}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4haa pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4haa.py write --out PIN.json -- <main_v4haa args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4haa as MA
    aa, rest = MA.resolve(run)
    if aa.prereg_v4haa:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4haa")
    blk = block(aa, rest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
