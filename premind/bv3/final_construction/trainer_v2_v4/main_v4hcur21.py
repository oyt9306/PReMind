#!/usr/bin/env python3
"""PReMind v2 entry point + curriculum v2.1 (curriculum_v2_1.py: every emitted draw gated, withhold-aware mastery,
truncation-tolerant resume; Codex review K 2026-10-15). NEW file, same hooks as main_v4hcur2 (which, like
curriculum_v2.py / main_v4hcur.py, is imported or mirrored, never edited):
  eval_v1.absent_sampler.AbsentShareSampler -> curriculum_v2_1.CurriculumSamplerV21
  eval_v1.ddp_main.ddp_train_one_step       -> set step, refund, all-gather FILTERED + OUTCOMES, requeue, observe, log
  trainer_v2.algo_v2.dynamic_rollout / shaped_rewards / informative -> every rolled-out group's outcome
                                               (curriculum_v2_1.group_outcome on the in-process ledgers)
Logs: OUT.curriculum/report.json, steps.jsonl (per step as v2 + realized non-acute-absent [a, n]), summary.json.
Options: --curriculum off|v21, --replay-budget N (default curriculum_v2_1.REPLAY_BUDGET = 80; acute replay slots and
the global cap on acute repeat draws; pinned), --prereg-v4hcur21 PIN (pins_v4hcur21.py). v21 needs what v2 needs.
Plan: curriculum_v2_1.Plan21 (every acute row once + the replay budget + v1's non-acute rule).
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
from trainer_v2_v4 import curriculum_v2_1 as C21  # noqa: E402
from trainer_v2_v4 import main_v4hcur as MC  # noqa: E402

logger = logging.getLogger(__name__)
OWNED = ("--prereg-v4haa", "--prereg-v4h", "--prereg-v4hcur", "--prereg-v4hcur2")
FILTERED: list = []
OUTCOMES: list = []
_CTX: dict = {"active": False, "eps": None, "j": 0, "acute": {}}
ORIG: dict = {}


def cur_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--curriculum", default="off", choices=("off", "v21"))
    ap.add_argument("--prereg-v4hcur21", default=None)
    ap.add_argument("--replay-budget", type=int, default=C21.REPLAY_BUDGET)
    return ap


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    from trainer_v2_v4 import main_v4h as MH
    if argv[:1] == ["--"]:
        argv = argv[1:]
    cur, rest = cur_parser().parse_known_args(argv)
    for flag in OWNED:
        if any(a == flag or a.startswith(flag + "=") for a in rest):
            raise SystemExit(f"REFUSED: {flag} is set by main_v4hcur21 (use --prereg-v4hcur21)")
    if cur.curriculum == "v21":
        if MH._value(rest, "--target-absent-share") is None:
            raise SystemExit("REFUSED: --curriculum v21 needs --target-absent-share (the non-acute absent share)")
        if (MH._value(rest, "--acute-absent") or "keep") != "keep":
            raise SystemExit("REFUSED: --curriculum v21 needs --acute-absent keep (every acute row is scheduled)")
        if MH._value(rest, "--algo") != "dapo":
            raise SystemExit("REFUSED: --curriculum v21 is defined for --algo dapo")
    return cur, rest


def build_plan(rest: list[str], replay_budget: int = C21.REPLAY_BUDGET) -> tuple[C21.Plan21, set, dict, list]:
    """(plan, release train row ids, data block, items), loaded exactly as main_v4hcur2.build_plan loads them."""
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
    plan = C21.Plan21(items, acute_ids, args.steps, args.prompts_per_step * w, args.target_absent_share, args.seed,
                      replay_budget)
    data = {"release_train": str(path), "release_train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "key_actions_sha256": hashlib.sha256(kp.read_bytes()).hexdigest() if kp.is_file() else None,
            "world": w, "prompts_per_rank": args.prompts_per_step, "out": args.out}
    return plan, ids, data, items


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


def _informative(rewards: list) -> bool:
    ok = ORIG["AL.informative"](rewards)
    if _CTX["active"] and _CTX["eps"] is not None:
        g, j = len(rewards), _CTX["j"]
        grp = _CTX["eps"][j * g:(j + 1) * g]
        _CTX["j"] = j + 1
        if grp:
            rid = grp[0].item.row_id
            try:
                OUTCOMES.append(C21.group_outcome(rid, [ep.ledger for ep in grp], _CTX["acute"].get(rid, False)))
            except (KeyError, TypeError, AttributeError) as e:      # never breaks training; logged per step
                logger.warning("curriculum v2.1: no outcome for %s (%s)", rid, e)
                OUTCOMES.append({"row_id": rid, "mastery": None, "pc_rate": None, "n": len(grp), "error": str(e)})
            if not ok:
                succ = [MC._success(ep) for ep in grp]
                FILTERED.append((rid, "unknown" if None in succ else CV.classify(succ)))
    return ok


def _ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler,
                        cursor, *a, **kw):
    orig = ORIG["DM.ddp_train_one_step"]
    if not isinstance(sampler, C21.CurriculumSamplerV21):
        return orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler, cursor,
                    *a, **kw)
    sampler.set_step(step)
    FILTERED.clear()
    OUTCOMES.clear()
    rec, used = orig(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist, sampler, cursor,
                     *a, **kw)
    pg = args.prompts_per_step * dist.world
    taken = [int(t) for t in rec.get("ddp", {}).get("refills_per_rank", [0] * dist.world)]
    refunded = sampler.refund(MC.unused_positions(len(sampler.step_draws), pg, taken))
    filtered = [tuple(x) for part in dist.gather(list(FILTERED)) for x in part]
    outcomes = [dict(x) for part in dist.gather(list(OUTCOMES)) for x in part]
    requeued = sampler.requeue(filtered)
    bad = [o for o in outcomes if o.get("mastery") is None]
    added = sampler.observe([o for o in outcomes if o.get("mastery") is not None])
    if dist.is_main:
        draws = sampler.step_draws
        drawn = sorted({r for r, _, _ in draws})
        line = {"step": step, "curriculum": "v2.1", "schedule_sha256": sampler.plan.sha,
                "scheduled": [[r, k] for r, k, _ in draws[:pg]], "refills": [[r, k] for r, k, _ in draws[pg:]],
                "refunded": refunded, "filtered": [list(x) for x in filtered], "requeued": requeued,
                "requeue_drawn": [r for r, k, _ in draws if k == "requeue"],
                "replay_drawn": [r for r, k, _ in draws if k == "replay"],
                "fill_drawn": [r for r, k, _ in draws if k == "fill"],
                "stuck_new": [x["row_id"] for x in sampler.new_suspects],
                "realized_non_acute_absent": [sampler.n_naa, sampler.n],
                "outcomes": outcomes, "outcome_errors": len(bad), "replay_added": added,
                "replay_events": sampler.events, "replay_queue": sampler.queue_state(),
                "mastery_history": {r: sampler.hist.get(r, []) for r in drawn},
                "cursor_after": cursor + used, "state": sampler.state()}
        d = Path(sampler.log_path).parent
        with Path(sampler.log_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        (d / "summary.json").write_text(json.dumps({"step": step, **sampler.summary()}, indent=1) + "\n")
        if sampler.new_suspects:
            with (d / "suspect_rows.jsonl").open("a", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x) + "\n" for x in sampler.new_suspects)
        (d / "suspect_images.json").write_text(json.dumps(sampler.image_suspects(), indent=1) + "\n")
    return rec, used


def install(plan: C21.Plan2, log_path: Path) -> None:
    from eval_v1 import absent_sampler as AS
    from eval_v1 import ddp_main as DM
    from trainer_v2 import algo_v2 as AL
    if ORIG:
        raise RuntimeError("main_v4hcur21 hooks already installed")
    _CTX["acute"] = {r: s.startswith("acute") for s, rows in plan.strata.items() for r in rows}

    def factory(items: list, target: float, seed: int) -> C21.CurriculumSamplerV21:
        if abs(target - plan.quotas["absent_share"]) > 1e-12 or seed != plan.seed:
            raise SystemExit("REFUSED: curriculum plan built for another absent share / seed")
        if {it.row_id for it in items} != set(plan.of_rows()):
            raise SystemExit("REFUSED: the training items differ from the curriculum plan rows")
        return C21.CurriculumSamplerV21(items, plan, log_path)
    targets = {"AS.AbsentShareSampler": (AS, "AbsentShareSampler", factory),
               "DM.ddp_train_one_step": (DM, "ddp_train_one_step", _ddp_train_one_step),
               "AL.dynamic_rollout": (AL, "dynamic_rollout", _dynamic_rollout),
               "AL.shaped_rewards": (AL, "shaped_rewards", _shaped_rewards),
               "AL.informative": (AL, "informative", _informative)}
    for key, (mod, name, fn) in targets.items():
        ORIG[key] = getattr(mod, name)
        setattr(mod, name, fn)


def runtime_rule() -> dict:
    return {"acute_heavy_frac": C21.ACUTE_HEAVY_FRAC, "acute_repeat_cap": "replay_budget (replay + re-queue + fill repeats of acute rows)",
            "replay": C21.REPLAY, "mastery": C21.MASTERY, "max_visits": C21.MAX_VISITS,
            "non_acute_cap": "max(non_acute, cycle-0 scheduled visits)", "share_tol_draws": C21.SHARE_TOL,
            "share_denominator": "emitted draws net of refunds, all strata", "continuation": "none (fill pool)",
            "mastery_criterion": "v2.1 (withhold full; acute withhold rows: no wrong_concept/distractor penalty)",
            "min_gap_steps": CV.MIN_GAP_STEPS,
            "stuck": {"acute_zero_mastery_visits": C21.STUCK_ZERO_VISITS, "report": "suspect_rows.jsonl"},
            "image_suspects": {"min_rows": C21.IMG_MIN_ROWS, "max_mean_mastery": C21.IMG_MAX_MASTERY,
                               "min_run_acute_mastery": C21.IMG_RUN_MASTERY, "action": "log only"}}


def prepare(rest: list[str], rank: int, replay_budget: int = C21.REPLAY_BUDGET) -> tuple[C21.Plan21, Path]:
    plan, ids, data, _ = build_plan(rest, replay_budget)
    checks = plan.check(ids)
    d = MC.cur_dir(data["out"])
    rep = {**plan.report(), "curriculum": "v2.1", "checks": checks, "data": data, "seed": plan.seed,
           "requeue": CV.REQUEUE, **runtime_rule()}
    if rank == 0:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "report.json"
        if p.exists() and json.loads(p.read_text()).get("schedule_sha256") != plan.sha:
            raise SystemExit(f"REFUSED: {p} holds another schedule")
        p.write_text(json.dumps(rep, indent=1) + "\n")
    logger.info("curriculum v2.1: sha %s slots %s absent %.3f checks %s", plan.sha[:12], plan.quotas["slots"],
                rep["non_acute_absent_share"], checks)
    return plan, d / "steps.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    cur, rest = resolve(argv)
    from trainer_v2_v4 import main_v4haa as MA
    from trainer_v2_v4 import main_v4h as MH
    MH.no_abbrev()
    MH.release_pin_presets()
    if cur.prereg_v4hcur21:
        from trainer_v2_v4 import pins_v4hcur21 as P21
        P21.verify(Path(cur.prereg_v4hcur21), cur, rest)
        logger.info("v4hcur21 pin verified: %s", cur.prereg_v4hcur21)
    else:
        logger.warning("v4hcur21 run without --prereg-v4hcur21 (CPU only; the launcher requires a pin)")
    if cur.curriculum == "v21":
        if "--print-config" in rest:
            plan, ids, _, _ = build_plan(rest, cur.replay_budget)
            rep = plan.report()
            print(json.dumps({"v4hcur21": {"curriculum": cur.curriculum, "schedule_sha256": plan.sha,
                                           "quotas": plan.quotas, "checks": plan.check(ids),
                                           "non_acute_absent_share": rep["non_acute_absent_share"],
                                           "replay_slots": rep["replay_slots"], "replay_budget": cur.replay_budget,
                                           **runtime_rule()}}, indent=1))
        else:
            if MC.world() < 2:
                raise SystemExit("REFUSED: --curriculum v21 runs on the DDP path only (WORLD_SIZE >= 2)")
            plan, log_path = prepare(rest, int(os.environ.get("RANK", "0")), cur.replay_budget)
            install(plan, log_path)
    return MA.main(rest)


if __name__ == "__main__":
    sys.exit(main())
