#!/usr/bin/env python3
"""PReMind v4 entry point with BATCH-LEVEL CoViP mixing (P1, owner 2026-10-13). GPU jobs are launched by the
orchestrator only; CPU tests use ``--print-config``.

= trainer_v2_v4.main_v4 (same arguments, same KL / reward / monitor options; aux_ratio stays 0 there) +
  --aux-mode batch|step     batch (default): CoViP prompts inside each optimizer step (batch_mix.py, patch_b.py);
                            step: the earlier interleaved aux steps (delegates to main_v4 with --aux-ratio)
  --aux-batch-ratio r       share of CoViP prompts among ALL prompts over the run (P1: 0.25)
  --prereg-v4b PIN          the batch pin (pins_v4b.py); --prereg-v4 is not used in batch mode
--steps = PReMind steps (P1 Stage A: 250, as P3); --prompts-per-step = PReMind prompts per rank (as P3).
The aux pool options (--aux-data / --aux-methods / --aux-exclude) are main_v4's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import covip_aux as CA  # noqa: E402
from trainer_v2_v4 import main_v4 as MV4  # noqa: E402

B_KEYS = ("aux_mode", "aux_batch_ratio", "prereg_v4b")


def b_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--aux-mode", default="batch", choices=("batch", "step"))
    ap.add_argument("--aux-batch-ratio", type=float, default=0.0)
    ap.add_argument("--prereg-v4b", default=None)
    return ap


def parse(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, bool]:
    b, rest = b_parser().parse_known_args(argv)
    if b.aux_mode == "step":
        args, pc = MV4.parse(rest)
    else:
        args, pc = MV4.parse(rest)
        if args.aux_ratio != 0:
            raise SystemExit("REFUSED: batch mode takes --aux-batch-ratio; --aux-ratio (step mode) must stay 0")
        if not 0 <= b.aux_batch_ratio < 1:
            raise SystemExit("REFUSED: --aux-batch-ratio must be in [0, 1)")
        if b.aux_batch_ratio > 0 and args.target_absent_share is None:
            raise SystemExit("REFUSED: batch mixing needs --target-absent-share (the PReMind sampler stream)")
        if b.aux_batch_ratio > 0 and args.monitor_frac > 0:
            raise SystemExit("REFUSED: batch mixing is defined for the dev-split recipe (--monitor-frac 0)")
    for k in B_KEYS:
        setattr(args, k, getattr(b, k))
    return args, pc


def pool(args: argparse.Namespace) -> tuple[list, dict]:
    ex = set(json.loads(Path(args.aux_exclude).read_text())) if args.aux_exclude else set()
    items, man = CA.load_items(Path(args.aux_data), tuple(args.aux_methods.split(",")), exclude_images=ex,
                               seed=args.seed)
    if not items:
        raise SystemExit("REFUSED: the CoViP pool is empty after the overlap exclusion")
    return items, {**man, "exclude": args.aux_exclude}


def make_plan(args: argparse.Namespace, n_pool: int, world: int):
    from trainer_v2_v4 import batch_mix as BM
    return BM.AuxBatchPlan(args.aux_batch_ratio, args.steps, args.prompts_per_step * world, world, args.seed, n_pool,
                           getattr(args, "dapo_max_prompts_factor", 3.0))


def main(argv: Optional[list[str]] = None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    b, _ = b_parser().parse_known_args(argv)
    if b.aux_mode == "step":                               # interleaved aux steps: the main_v4 path, unchanged
        rest = [a for a in argv if a not in ("--aux-mode", "step")]
        return MV4.main(rest)
    args, print_config = parse(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    items, man, plan = [], {}, None
    if args.aux_batch_ratio > 0:
        items, man = pool(args)
        plan = make_plan(args, len(items), world)
    if args.prereg_v4b:
        from trainer_v2_v4 import pins_v4b
        pins_v4b.verify(Path(args.prereg_v4b), args, man, plan)
    if print_config:
        print(json.dumps({"v4": {k: getattr(args, k) for k in MV4.V4_KEYS}, "v4b": {k: getattr(args, k) for k in B_KEYS},
                          "reward_preset": args.reward_preset, "arm": args.arm, "steps": args.steps,
                          "prompts_per_step": args.prompts_per_step, "world": world, "aux_manifest": man,
                          "plan": None if plan is None else {"total": plan.total, "q": str(plan.q),
                                                             "slots_per_step": plan.slots_per_step,
                                                             "first_counts": list(plan.counts[:12])}},
                         indent=1, default=str))
        return 0
    from trainer_v2_v4 import patch as PT
    PT.install(kl_beta=args.kl_beta, aux_items=[], aux_ratio=0.0, world=world, seed=args.seed,
               monitor=(args.monitor_frac, args.monitor_seed) if args.monitor_frac > 0 else None)
    if plan is not None:
        from trainer_v2_v4 import patch_b as PB
        PB.install(plan, items)
    if world >= 2:                                         # pragma: no cover - torchrun GPU job
        from eval_v1 import ddp_main as DM
        from trainer_v2 import ddp_v2 as DV
        dist = DV.init_from_env()
        DM._setup_logging(dist)
        try:
            return DM.gpu_main(args, dist)
        finally:
            DV.shutdown()
    from eval_v1 import main_run as MR                     # pragma: no cover - single-GPU job
    return MR.gpu_main(args)


if __name__ == "__main__":
    sys.exit(main())
