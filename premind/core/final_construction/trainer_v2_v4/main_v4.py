#!/usr/bin/env python3
"""PReMind v4 RLVR entry point (GPU jobs launched by the orchestrator only; CPU tests use ``--print-config``).

= eval_v1.main_run (single GPU) / eval_v1.ddp_main (torchrun, WORLD_SIZE >= 2) + the v4 pieces, installed in-process
by ``patch.install`` (frozen trainer_v2 / eval_v1 files are not edited):
  --kl-beta B              KL(pi || base) k3 term (kl_ref.py); 0 = off (default)
  --reward-preset v4wf     withholding-fair reward (reward_v4.py); every reward_v5 preset still works unchanged
  --aux-ratio r            fraction of optimizer steps on CoViP samples (mix.py); 0 = off (default), 1 = CoViP only
  --aux-data / --aux-methods / --aux-prompts-per-step / --aux-exclude   CoViP pool, methods (multi_vision[,vision]),
                           prompts per aux step PER RANK, overlap exclusion list (covip_overlap.py)
  --monitor-frac F         held-out monitor slice of the train release (monitor_slice.py; default 0.08): never
                           trained on; the in-loop "dev" eval (over-abstention monitor, confirmed stop rule) runs on it
  --prereg-v4 PIN          verify the v4 pin (pins_v4.py: trainer_v2_v4 code sha256, aux data / exclude sha256,
                           kl / aux / preset constants); the release PREREG (--prereg) is verified by main_run as usual
Every other flag is main_run's (``--release`` is a parameter, e.g. RELEASE_TRAIN_DEV_v8 once WP-A has built it).
The v4 values are stored in the checkpoint args (resume refuses a different configuration).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import covip_aux as CA  # noqa: E402
from trainer_v2_v4 import reward_v4 as RW  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_EXCLUDE = HERE / "out" / "covip_overlap_v4.reported.exclude.json"
V4_KEYS = ("kl_beta", "kl_estimator", "aux_ratio", "aux_data", "aux_methods", "aux_prompts_per_step", "aux_exclude",
           "monitor_frac", "monitor_seed", "prereg_v4")


def v4_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--kl-estimator", default="k3", choices=("k3",))
    ap.add_argument("--aux-ratio", type=float, default=0.0)
    ap.add_argument("--aux-data", default=str(CA.DEFAULT_DATA))
    ap.add_argument("--aux-methods", default="multi_vision")
    ap.add_argument("--aux-prompts-per-step", type=int, default=0)
    ap.add_argument("--aux-exclude", default=str(DEFAULT_EXCLUDE))
    ap.add_argument("--monitor-frac", type=float, default=0.08,
                    help="held-out monitor slice of TRAIN (no dev split, owner 2026-10-13); 0 = off (needs a dev file)")
    ap.add_argument("--monitor-seed", type=int, default=0)
    ap.add_argument("--prereg-v4", default=None)
    ap.add_argument("--print-config", action="store_true", help="CPU: parse, load the aux pool, print, exit")
    return ap


def parse(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, bool]:
    """(main_run args + the v4 keys, print_config). print_config stays OUT of args: the checkpoint stores args."""
    RW.register()                                   # v4 preset names become valid --reward-preset choices
    from eval_v1 import main_run as MR
    v4, rest = v4_parser().parse_known_args(argv)
    args = MR.parse(rest)
    for k in V4_KEYS:
        setattr(args, k, getattr(v4, k))
    if not 0.0 <= args.aux_ratio <= 1.0:
        raise SystemExit("REFUSED: --aux-ratio must be in [0, 1] (1 = CoViP-only ablation)")
    if args.aux_ratio > 0 and args.aux_prompts_per_step < 1:
        raise SystemExit("REFUSED: --aux-ratio > 0 needs --aux-prompts-per-step >= 1")
    if not (args.monitor_frac == 0 or 0 < args.monitor_frac < 0.5):
        raise SystemExit("REFUSED: --monitor-frac must be 0 (off) or in (0, 0.5)")
    if args.kl_beta < 0:
        raise SystemExit("REFUSED: --kl-beta must be >= 0")
    return args, v4.print_config


def aux_pool(args: argparse.Namespace) -> tuple[list, dict]:
    if args.aux_ratio <= 0:
        return [], {}
    ex = set(json.loads(Path(args.aux_exclude).read_text())) if args.aux_exclude else set()
    items, man = CA.load_items(Path(args.aux_data), tuple(args.aux_methods.split(",")), exclude_images=ex,
                               seed=args.seed)
    if not items:
        raise SystemExit("REFUSED: the CoViP pool is empty after the overlap exclusion")
    return items, {**man, "exclude": args.aux_exclude}


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args, print_config = parse(argv)
    items, man = aux_pool(args)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.prereg_v4:
        from trainer_v2_v4 import pins_v4
        pins_v4.verify(Path(args.prereg_v4), args, man)
    if print_config:
        mon = None
        if args.monitor_frac > 0:
            from trainer_v2_v4 import monitor_slice as MS
            mon = {k: v for k, v in MS.slice_ids(Path(args.release), args.mode, args.monitor_frac,
                                                 args.monitor_seed)[1].items() if k != "slice_components"}
        print(json.dumps({"v4": {k: getattr(args, k) for k in V4_KEYS}, "reward_preset": args.reward_preset,
                          "arm": args.arm, "aux_manifest": man, "monitor_slice": mon, "world": world},
                         indent=1, default=str))
        return 0
    from trainer_v2_v4 import patch
    patch.install(kl_beta=args.kl_beta, aux_items=items, aux_ratio=args.aux_ratio, p_aux=args.aux_prompts_per_step,
                  world=world, dapo_factor=getattr(args, "dapo_max_prompts_factor", 3.0), seed=args.seed,
                  monitor=(args.monitor_frac, args.monitor_seed) if args.monitor_frac > 0 else None)
    if world >= 2:                                   # pragma: no cover - torchrun GPU job
        from eval_v1 import ddp_main as DM
        from trainer_v2 import ddp_v2 as DV
        dist = DV.init_from_env()
        DM._setup_logging(dist)
        try:
            return DM.gpu_main(args, dist)
        finally:
            DV.shutdown()
    from eval_v1 import main_run as MR             # pragma: no cover - single-GPU job
    return MR.gpu_main(args)


if __name__ == "__main__":
    sys.exit(main())
