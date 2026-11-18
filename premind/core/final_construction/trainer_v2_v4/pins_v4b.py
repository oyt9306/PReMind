"""Pin of a batch-mixing run (main_v4b): the v4 pin block (pins_v4.block, unchanged) + sha256 of the batch-mixing code,
the CoViP pool / exclude / CoViP sources and the per-step allocation of the plan. ``write`` / ``verify`` as pins_v4.
Usage: python -I pins_v4b.py write --out PIN.json -- <main_v4b args>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

CODE_B = ("batch_mix.py", "patch_b.py", "leak.py", "main_v4b.py", "pins_v4b.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def block(args: Any, man: dict, plan: Any) -> dict:
    from trainer_v2_v4 import covip_aux as CA
    from trainer_v2_v4 import pins_v4 as P4
    out = P4.block(args, man)
    out["prereg"] = "PREREG_TRAINER_V2_V4B"
    out["code_sha256_b"] = {f: _sha(HERE / f) for f in CODE_B}
    out["batch"] = {"aux_mode": args.aux_mode, "ratio": args.aux_batch_ratio}
    if plan is not None:
        out["batch"].update(total=plan.total, q=str(plan.q), slots_per_step=plan.slots_per_step,
                            counts_sha256=hashlib.sha256(json.dumps(list(plan.counts)).encode()).hexdigest(),
                            pool_n=plan.n_pool, data_sha256=man.get("sha256"),
                            exclude_sha256=_sha(Path(args.aux_exclude)) if args.aux_exclude else None,
                            covip_src_sha256={f: _sha(CA.COVIP / f) for f in ("grpo_pmllm.py", "vlm_modules/qwen_module.py")})
    return out


def verify(pin: Path, args: Any, man: dict, plan: Any) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(args, man, plan)
    bad = [k for k in ("code_sha256", "code_sha256_b", "constants", "batch", "monitor_slice") if pre.get(k) != now.get(k)]
    if bad:
        raise SystemExit(f"REFUSED: v4b pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    import os
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4b.py write --out PIN.json -- <main_v4b args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    from trainer_v2_v4 import main_v4b as M
    args, _ = M.parse(run)
    world = int(os.environ.get("PIN_WORLD", "1"))
    items, man = M.pool(args) if args.aux_batch_ratio > 0 else ([], {})
    plan = M.make_plan(args, len(items), world) if items else None
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    out.write_text(json.dumps(block(args, man, plan), indent=1) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
