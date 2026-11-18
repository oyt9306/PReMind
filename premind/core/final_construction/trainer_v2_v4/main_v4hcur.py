#!/usr/bin/env python3
"""PReMind v2 entry point + the coverage curriculum (owner 2026-10-15, pilot CUR = pilot F + curriculum_v1).

= trainer_v2_v4.main_v4haa, unchanged, with in-process only (no pinned file is edited), when ``--curriculum v1``:
  eval_v1.absent_sampler.AbsentShareSampler -> curriculum_v1.CurriculumSampler (same draw / skip / payload / summary;
                                               ddp_main builds it because --target-absent-share is set)
  eval_v1.ddp_main.ddp_train_one_step       -> set the step, refund refill positions no rank rolled out, all-gather
                                               the DAPO-filtered groups, re-queue all-wrong ones once, log the step
  trainer_v2.algo_v2.dynamic_rollout / shaped_rewards / informative -> record filtered groups + reason (no change)
Before training (rank 0) the plan is built from the release, the 5 sanity checks run (CurriculumCheckFailed stops the
run) and OUT.curriculum/report.json (coverage report) is written; per-step log OUT.curriculum/steps.jsonl.
Options (all others are main_v4haa's; GPU jobs are launched by the orchestrator only, CPU: ``--print-config``):
  --curriculum off|v1        off = exactly main_v4haa
  --prereg-v4hcur PIN        the run pin (pins_v4hcur.py write), verified before anything runs
``--prereg-v4haa`` / ``--prereg-v4h`` are refused (the v4hcur pin contains both blocks). v1 needs --target-absent-share
(the non-acute absent share), --algo dapo, --acute-absent keep and the DDP path (WORLD_SIZE >= 2).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import curriculum_v1 as CV  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4haa", "--prereg-v4h")
FILTERED: list = []                                  # this rank's filtered groups of the current step
_CTX: dict = {"active": False, "eps": None, "j": 0}
ORIG: dict = {}


def cur_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--curriculum", default="off", choices=("off", "v1"))
    ap.add_argument("--prereg-v4hcur", default=None)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """(curriculum options, the main_v4haa argv). Refuses owned options and inconsistent combinations."""
    from trainer_v2_v4 import main_v4h as MH
    if argv[:1] == ["--"]:
        argv = argv[1:]
    cur, rest = cur_parser().parse_known_args(argv)
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in rest):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcur (use --prereg-v4hcur)")
    if cur.curriculum == "v1":
        if MH._value(rest, "--target-absent-share") is None:
            raise SystemExit("REFUSED: --curriculum v1 needs --target-absent-share (the non-acute absent share)")
        if (MH._value(rest, "--acute-absent") or "keep") != "keep":
            raise SystemExit("REFUSED: --curriculum v1 needs --acute-absent keep (every acute row is scheduled)")
        if MH._value(rest, "--algo") != "dapo":
            raise SystemExit("REFUSED: --curriculum v1 is defined for --algo dapo")
    return cur, rest


def world() -> int:
    return int(os.environ.get("PIN_WORLD", os.environ.get("WORLD_SIZE", "1")))


def run_args(rest: list[str]) -> tuple[Any, Any]:
    """(main_v4 args, v4h options) exactly as the run parses them."""
    from trainer_v2_v4 import main_v4 as MV4
    from trainer_v2_v4 import main_v4h as MH
    from trainer_v2_v4 import main_v4haa as MA
    MH.no_abbrev()
    _, rest_aa = MA.resolve(rest)
    h, base_rest, _ = MH.resolve(rest_aa)
    if h.base != "v4":
        raise SystemExit("REFUSED: --curriculum v1 runs on --base v4 (no batch mixing)")
    args, _ = MV4.parse(base_rest)
    return args, h


def build_plan(rest: list[str]) -> tuple[CV.Plan, set, dict]:
    """(plan, release train row ids, data block) from the release, as ddp_main loads it (passive arm items)."""
    import hashlib

    from eval_v1 import release_data as D
    from trainer_v2_v4 import key_actions_v4h as KA
    from trainer_v2_v4 import main_v4h as MH
    args, h = run_args(rest)
    rel = Path(args.release)
    kp = MH.key_actions_path(h)
    acute_ids = set(KA.load(kp, rel, args.mode)) if kp.is_file() else set()
    items, _ = D.load_release(rel, "train", args.mode, arms=(args.arm,))
    path = D.release_file(rel, "train", args.mode)
    ids = {json.loads(x)["sample_id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    w = world()
    plan = CV.Plan(items, acute_ids, args.steps, args.prompts_per_step * w, args.target_absent_share, args.seed)
    data = {"release_train": str(path), "release_train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "key_actions_sha256": hashlib.sha256(kp.read_bytes()).hexdigest() if kp.is_file() else None,
            "world": w, "prompts_per_rank": args.prompts_per_step, "out": args.out}
    return plan, ids, data


def cur_dir(out: str) -> Path:
    o = Path(out)
    return o.parent / (o.name + ".curriculum")       # outside OUT: main_run refuses a non-empty fresh OUT


# ----------------------------------------------------------------------------- in-process hooks
def _dynamic_rollout(*a: Any, **kw: Any) -> Any:
    _CTX.update(active=True, eps=None, j=0)
    try:
        return ORIG["AL.dynamic_rollout"](*a, **kw)
    finally:
        _CTX.update(active=False, eps=None, j=0)


def _shaped_rewards(eps: list, cfg: Any) -> list:
    out = ORIG["AL.shaped_rewards"](eps, cfg)
    if _CTX["active"]:
        _CTX.update(eps=eps, j=0)
    return out


def _success(ep: Any) -> Optional[bool]:
    from trainer_v2 import opsd as OP
    try:
        return not OP.failure_reasons(ep.ledger)
    except (KeyError, TypeError, AttributeError):
        return None


def _informative(rewards: list) -> bool:
    ok = ORIG["AL.informative"](rewards)
    if _CTX["active"] and _CTX["eps"] is not None:
        g, j = len(rewards), _CTX["j"]
        grp = _CTX["eps"][j * g:(j + 1) * g]
        _CTX["j"] = j + 1
        if not ok and grp:
            succ = [_success(ep) for ep in grp]
            FILTERED.append((grp[0].item.row_id, "unknown" if None in succ else CV.classify(succ)))
    return ok


def unused_positions(n_draws: int, p_global: int, taken: list[int]) -> list[int]:
    """Indices (into the step's draws) of refill-stream positions no rank took (rank r owns r, r + W, ...)."""
    w = len(taken)
    return [p_global + i for i in range(n_draws - p_global) if i // w >= taken[i % w]]


def _ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler,
                        cursor, *a, **kw):
    orig = ORIG["DM.ddp_train_one_step"]
    if not isinstance(sampler, CV.CurriculumSampler):
        return orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler, cursor,
                    *a, **kw)
    sampler.set_step(step)
    FILTERED.clear()
    rec, used = orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler, cursor,
                     *a, **kw)
    pg = args.prompts_per_step * dist.world
    taken = [int(t) for t in rec.get("ddp", {}).get("refills_per_rank", [0] * dist.world)]
    refunded = sampler.refund(unused_positions(len(sampler.step_draws), pg, taken))
    filtered = [tuple(x) for part in dist.gather(list(FILTERED)) for x in part]
    requeued = sampler.requeue(filtered)
    if dist.is_main:
        draws = sampler.step_draws
        line = {"step": step, "schedule_sha256": sampler.plan.sha,
                "scheduled": [[r, k] for r, k, _ in draws[:pg]], "refills": [[r, k] for r, k, _ in draws[pg:]],
                "refunded": refunded, "filtered": [list(x) for x in filtered], "requeued": requeued,
                "requeue_drawn": [r for r, k, _ in draws if k == "requeue"], "cursor_after": cursor + used,
                "state": sampler.state()}
        with Path(sampler.log_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    return rec, used


def install(plan: CV.Plan, log_path: Path) -> None:
    from eval_v1 import absent_sampler as AS
    from eval_v1 import ddp_main as DM
    from trainer_v2 import algo_v2 as AL
    if ORIG:
        raise RuntimeError("main_v4hcur hooks already installed")

    def factory(items: list, target: float, seed: int) -> CV.CurriculumSampler:
        if abs(target - plan.quotas["absent_share"]) > 1e-12 or seed != plan.seed:
            raise SystemExit("REFUSED: curriculum plan built for another absent share / seed")
        if {it.row_id for it in items} != set(plan.of_rows()):
            raise SystemExit("REFUSED: the training items differ from the curriculum plan rows")
        return CV.CurriculumSampler(items, plan, log_path)
    targets = {"AS.AbsentShareSampler": (AS, "AbsentShareSampler", factory),
               "DM.ddp_train_one_step": (DM, "ddp_train_one_step", _ddp_train_one_step),
               "AL.dynamic_rollout": (AL, "dynamic_rollout", _dynamic_rollout),
               "AL.shaped_rewards": (AL, "shaped_rewards", _shaped_rewards),
               "AL.informative": (AL, "informative", _informative)}
    for key, (mod, name, fn) in targets.items():
        ORIG[key] = getattr(mod, name)
        setattr(mod, name, fn)


def prepare(rest: list[str], rank: int) -> tuple[CV.Plan, Path]:
    """Plan + sanity checks (every rank; any failure stops the run) + coverage report (rank 0)."""
    plan, ids, data = build_plan(rest)
    checks = plan.check(ids)
    d = cur_dir(data["out"])
    rep = {**plan.report(), "checks": checks, "data": data, "seed": plan.seed, "requeue": CV.REQUEUE,
           "min_gap_steps": CV.MIN_GAP_STEPS}
    if rank == 0:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "report.json"
        if p.exists() and json.loads(p.read_text()).get("schedule_sha256") != plan.sha:
            raise SystemExit(f"REFUSED: {p} holds another schedule")
        p.write_text(json.dumps(rep, indent=1) + "\n")
    logger.info("curriculum v1: sha %s slots %s absent %.3f acute/step %s checks %s", plan.sha[:12],
                plan.quotas["slots"], rep["non_acute_absent_share"], rep["acute_per_step"], checks)
    return plan, d / "steps.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    cur, rest = resolve(argv)
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import main_v4h as MH
    MH.no_abbrev()
    MH.release_pin_presets()
    if cur.prereg_v4hcur:
        from trainer_v2_v4 import pins_v4hcur as PC
        PC.verify(Path(cur.prereg_v4hcur), cur, rest)
        logger.info("v4hcur pin verified: %s", cur.prereg_v4hcur)
    else:
        logger.warning("v4hcur run without --prereg-v4hcur (allowed for smoke / CPU only; the launcher requires a pin)")
    if cur.curriculum == "v1":
        if "--print-config" in rest:
            plan, ids, _ = build_plan(rest)
            rep = plan.report()
            print(json.dumps({"v4hcur": {"curriculum": cur.curriculum, "schedule_sha256": plan.sha,
                                         "quotas": plan.quotas, "checks": plan.check(ids),
                                         "non_acute_absent_share": rep["non_acute_absent_share"],
                                         "acute_per_step": rep["acute_per_step"]}}, indent=1))
        else:
            if world() < 2:
                raise SystemExit("REFUSED: --curriculum v1 runs on the DDP path only (WORLD_SIZE >= 2)")
            plan, log_path = prepare(rest, int(os.environ.get("RANK", "0")))
            install(plan, log_path)
    return MA.main(rest)


if __name__ == "__main__":
    sys.exit(main())
