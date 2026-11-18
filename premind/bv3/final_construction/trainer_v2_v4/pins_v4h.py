"""Pin of a v2 (main_v4h) run: the base pin block (pins_v4.block for --base v4, pins_v4b.block for --base v4b; both
unchanged) + the v4h block: sha256 of every v4h code file, the pinned guard module (v4wh_g), the training rubric and
the frozen eval rubric / shared eval files, the key-action file, the data release pin (--prereg), and the reward /
judge configuration (preset, backend, provider, judge model spec, caps, text log; acute_absent only when mask, so a
keep pin is unchanged and refuses a mask run, and vice versa).
  python -I pins_v4h.py write --out PIN.json -- <main_v4h args>     (write once; PIN_WORLD=<ranks> for --base v4b)
``verify`` (main_v4h --prereg-v4h PIN) refuses on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

CODE_H = ("hazard_rubric.py", "key_actions_v4h.py", "hazard_cue.py", "reward_v4h.py", "reward_v4hg.py",
          "hazard_judge.py", "hazard_judge_api.py", "hazard_monitor.py", "patch_h.py", "main_v4h.py", "pins_v4h.py")
H_CONFIG = ("base", "hazard_backend", "hazard_judge_provider", "hazard_judge_model", "hazard_usd_cap",
            "hazard_token_cap", "hazard_judge_concurrency", "hazard_judge_timeout", "hazard_monitor",
            "log_rollout_texts", "rollout_text_frac")


def _sha(p: Path) -> Optional[str]:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def base_parts(h: argparse.Namespace, rest: list[str]) -> tuple[Any, dict, Any]:
    """(main_v4 / main_v4b args, aux manifest, batch plan) exactly as the run builds them."""
    from trainer_v2_v4 import main_v4h as MH
    MH.no_abbrev()
    if h.base == "v4b":
        from trainer_v2_v4 import main_v4b as MB
        args, _ = MB.parse(rest)
        items, man = MB.pool(args) if args.aux_batch_ratio > 0 else ([], {})
        plan = MB.make_plan(args, len(items), int(os.environ.get("PIN_WORLD", os.environ.get("WORLD_SIZE", "1")))) \
            if items else None
        return args, man, plan
    from trainer_v2_v4 import main_v4 as MV4
    args, _ = MV4.parse(rest)
    _, man = MV4.aux_pool(args)
    return args, man, None


def block(h: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import hazard_rubric as HR
    from trainer_v2_v4 import key_actions_v4h as KA
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import reward_v4hg as RHG
    args, man, plan = base_parts(h, rest)
    if h.base == "v4b":
        from trainer_v2_v4 import pins_v4b as PB
        out = PB.block(args, man, plan)
    else:
        from trainer_v2_v4 import pins_v4 as P4
        out = P4.block(args, man)
    kp = MH.key_actions_path(h)
    out["prereg"] = "PREREG_TRAINER_V2_V4H"
    out["v4h"] = {"code_sha256": {f: _sha(HERE / f) for f in CODE_H},
                  "guard_sha256": RHG.GUARD_SHA256 if RHG.is_guarded(args.reward_preset) else None,
                  "rubric_sha256": HR.RUBRIC_SHA256, "eval_rubric_sha256": HR.EVAL_RUBRIC_SHA256,
                  "shared_sha256": KA.shared_sha256(), "key_actions": str(kp), "key_actions_sha256": _sha(kp),
                  "release": str(args.release), "release_pin": getattr(args, "prereg", None),
                  "release_pin_sha256": _sha(args.prereg) if getattr(args, "prereg", None) else None,
                  "config": {k: getattr(h, k) for k in H_CONFIG + tuple(k for k in MH.h_keys(h) if k not in MH.H_KEYS)}}
    return out


def verify(pin: Path, h: argparse.Namespace, rest: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(h, rest)
    if pre.get("prereg") != "PREREG_TRAINER_V2_V4H":
        raise SystemExit(f"REFUSED: {pin} is not a v4h pin (prereg={pre.get('prereg')!r})")
    bad = [k for k in sorted(set(pre) | set(now)) if k != "v4h" and pre.get(k) != now.get(k)]
    bad += [f"v4h.{k}" for k in sorted(set(pre["v4h"]) | set(now["v4h"])) if pre["v4h"].get(k) != now["v4h"].get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4h pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4h.py write --out PIN.json -- <main_v4h args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    from trainer_v2_v4 import main_v4h as MH
    h, rest, _ = MH.resolve(run)
    if h.prereg_v4h:
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4h")
    blk = block(h, rest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
