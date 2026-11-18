#!/usr/bin/env python3
"""PReMind v2 entry point + curriculum v2.2 (curriculum_v2_2.Plan22 = Plan21 + ``--safe-min-frac``; v14 fallback,
owner-approved 2026-10-15). NEW file: main_v4hcur21.py (pinned by the live CUR21B run) is imported read-only; its
in-process hooks (install, outcome / DDP step hooks, suspect logging) are reused unchanged, only the plan differs.
Options: --curriculum off|v22, --replay-budget N (default 80), --safe-min-frac F (default off; fallback 0.30),
--prereg-v4hcur22 PIN (pins_v4hcur22.py). Logs as main_v4hcur21 (report.json says curriculum v2.2).
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

from trainer_v2_v4 import curriculum_v1 as CV  # noqa: E402
from trainer_v2_v4 import curriculum_v2_1 as C21  # noqa: E402
from trainer_v2_v4 import curriculum_v2_2 as C22  # noqa: E402
from trainer_v2_v4 import main_v4hcur as MC  # noqa: E402
from trainer_v2_v4 import main_v4hcur21 as M21  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4haa", "--prereg-v4h", "--prereg-v4hcur", "--prereg-v4hcur2", "--prereg-v4hcur21")


def cur_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--curriculum", default="off", choices=("off", "v22"))
    ap.add_argument("--prereg-v4hcur22", default=None)
    ap.add_argument("--replay-budget", type=int, default=C21.REPLAY_BUDGET)
    ap.add_argument("--safe-min-frac", type=float, default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcur22 (use --prereg-v4hcur22)")
    cur, rest = cur_parser().parse_known_args(argv)
    if cur.curriculum == "v22":
        M21.resolve(["--curriculum", "v21"] + rest)          # the same requirements as v21 (share, keep, dapo)
    return cur, rest


def build_plan(rest: list[str], cur: argparse.Namespace) -> tuple[C22.Plan22, set, dict, list]:
    import hashlib

    from eval_v1 import release_data as D
    from trainer_v2_v4 import key_actions_v4h as KA
    from trainer_v2_v4 import main_v4h as MH
    args, h = MC.run_args(rest)
    rel = Path(args.release)
    kp = MH.key_actions_path(h)
    acute_ids = set(KA.load(kp, rel, args.mode)) if kp.is_file() else set()
    items, _ = D.load_release(rel, "train", args.mode, arms=(args.arm,))
    path = D.release_file(rel, "train", args.mode)
    ids = {json.loads(x)["sample_id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    w = MC.world()
    plan = C22.Plan22(items, acute_ids, args.steps, args.prompts_per_step * w, args.target_absent_share, args.seed,
                      cur.replay_budget, cur.safe_min_frac)
    data = {"release_train": str(path), "release_train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "key_actions_sha256": hashlib.sha256(kp.read_bytes()).hexdigest() if kp.is_file() else None,
            "world": w, "prompts_per_rank": args.prompts_per_step, "out": args.out}
    return plan, ids, data, items


def runtime_rule(cur: argparse.Namespace) -> dict:
    return {**M21.runtime_rule(), "safe_min_frac": cur.safe_min_frac, "replay_budget_requested": cur.replay_budget}


def prepare(rest: list[str], rank: int, cur: argparse.Namespace) -> tuple[C22.Plan22, Path]:
    plan, ids, data, _ = build_plan(rest, cur)
    checks = plan.check(ids)
    d = MC.cur_dir(data["out"])
    rep = {**plan.report(), "checks": checks, "data": data, "seed": plan.seed, "requeue": CV.REQUEUE,
           **runtime_rule(cur)}
    if rank == 0:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "report.json"
        if p.exists() and json.loads(p.read_text()).get("schedule_sha256") != plan.sha:
            raise SystemExit(f"REFUSED: {p} holds another schedule")
        p.write_text(json.dumps(rep, indent=1) + "\n")
    logger.info("curriculum v2.2: sha %s slots %s checks %s", plan.sha[:12], plan.quotas["slots"], checks)
    return plan, d / "steps.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    cur, rest = resolve(list(sys.argv[1:] if argv is None else argv))
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import main_v4h as MH
    MH.no_abbrev()
    MH.release_pin_presets()
    if cur.prereg_v4hcur22:
        from trainer_v2_v4 import pins_v4hcur22 as P22
        P22.verify(Path(cur.prereg_v4hcur22), cur, rest)
        logger.info("v4hcur22 pin verified: %s", cur.prereg_v4hcur22)
    else:
        logger.warning("v4hcur22 run without --prereg-v4hcur22 (CPU only; the launcher requires a pin)")
    if cur.curriculum == "v22":
        if "--print-config" in rest:
            plan, ids, _, _ = build_plan(rest, cur)
            print(json.dumps({"v4hcur22": {"schedule_sha256": plan.sha, "quotas": plan.quotas,
                                           "checks": plan.check(ids), **runtime_rule(cur)}}, indent=1, default=str))
        else:
            if MC.world() < 2:
                raise SystemExit("REFUSED: --curriculum v22 runs on the DDP path only (WORLD_SIZE >= 2)")
            plan, log_path = prepare(rest, int(os.environ.get("RANK", "0")), cur)
            M21.install(plan, log_path)
    return MA.main(rest)


if __name__ == "__main__":
    sys.exit(main())
