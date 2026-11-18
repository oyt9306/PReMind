"""v4 recipe pin (complements the release PREREG that eval_v1.main_run verifies with trainer_v2.pins).

``write`` (CPU, orchestrator) records: sha256 of the trainer_v2_v4 code, of the CoViP release sources the port copies,
of the Faker name pool, of the CoViP data file and the overlap exclude list; the recipe constants (reward preset,
kl beta / estimator, aux ratio / methods / prompts per step). ``verify`` refuses a run (SystemExit) whose code, data or
constants differ. Usage: python -I pins_v4.py write --out PIN.json -- <the main_v4 arguments of the run>
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

CODE = ("covip_aux.py", "kl_ref.py", "mix.py", "patch.py", "reward_v4.py", "main_v4.py", "pins_v4.py",
        "monitor_slice.py")
COVIP_SRC = ("grpo_pmllm.py", "vlm_modules/qwen_module.py")
CONSTANTS = ("reward_preset", "arm", "steps", "prompts_per_step", "kl_beta", "kl_estimator", "aux_ratio", "aux_methods",
             "aux_prompts_per_step", "monitor_frac", "monitor_seed")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def block(args: argparse.Namespace, aux_manifest: dict) -> dict:
    from trainer_v2_v4 import covip_aux as CA
    out = {"prereg": "PREREG_TRAINER_V2_V4", "code_sha256": {f: _sha(HERE / f) for f in CODE},
           "constants": {k: getattr(args, k) for k in CONSTANTS}}
    if args.monitor_frac > 0:
        from trainer_v2_v4 import monitor_slice as MS
        info = MS.slice_ids(Path(args.release), args.mode, args.monitor_frac, args.monitor_seed)[1]
        out["monitor_slice"] = {k: info[k] for k in ("n_rows", "n_slice", "ids_sha256")}
    if args.aux_ratio > 0:
        out["aux"] = {"data_sha256": aux_manifest.get("sha256"), "n_items": aux_manifest.get("n_items"),
                      "exclude_sha256": _sha(Path(args.aux_exclude)) if args.aux_exclude else None,
                      "covip_src_sha256": {f: _sha(CA.COVIP / f) for f in COVIP_SRC},
                      "names_sha256": _sha(CA.NAMES) if "vision" in args.aux_methods else None}
    return out


def verify(pin: Path, args: argparse.Namespace, aux_manifest: dict) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(args, aux_manifest)
    bad = [f"code:{f}" for f, s in now["code_sha256"].items() if pre["code_sha256"].get(f) != s]
    bad += [f"const:{k}" for k, v in now["constants"].items() if pre["constants"].get(k) != v]
    for k in ("aux", "monitor_slice"):
        if pre.get(k) != now.get(k):
            bad.append(k)
    if bad:
        raise SystemExit(f"REFUSED: v4 pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4.py write --out PIN.json -- <main_v4 args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    from trainer_v2_v4 import main_v4 as M
    args, _ = M.parse(run)
    _, man = M.aux_pool(args)
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    out.write_text(json.dumps(block(args, man), indent=1) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
