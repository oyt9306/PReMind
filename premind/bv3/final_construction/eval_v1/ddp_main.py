#!/usr/bin/env python3
"""Data-parallel (torchrun) release RLVR runner: the eval_v1.main_run recipe on W GPUs (rlvr_main_ddp.sh).

Same CLI as main_run (``main_run.parse``); ``--prompts-per-step`` is PER RANK, the global batch is P x W prompts
per step. Single-GPU runs keep using main_run (this module refuses WORLD_SIZE < 2).

Data: every rank builds the same seeded item order / absent-share sampler and computes the step's GLOBAL batch
exactly as a single-GPU run with P_global = P x W would (rotation from the global cursor, or P_global sampler draws);
rank r trains on slice [r P, (r + 1) P). The checkpoint ``data_cursor`` is global, so resume replays identically
(sampler.skip(cursor)), and a single-GPU checkpoint with P_global prompts per step resumes here with W x P = P_global.
DAPO dynamic sampling, rank-aware ("each rank fills its share"): rank r refills from positions r, r + W, ... of the
stream that follows the global batch (ddp_v2.StridedStream; a private fork of the sampler for the absent-share
order), no communication during the rollout; afterwards the stream advances by W x max_r(refills_r) positions
(unused positions are skipped). Each rank keeps at most P informative groups; the gradient is normalised by the
global episode count, so ranks with fewer kept groups are weighted correctly.
Rank 0 alone writes steps.jsonl, checkpoints (LoRA + optimizer, identical on all ranks; every rank's RNG state in
``ddp.rank_rng``), dev evals, wandb and main_report.json; the others wait at barriers. The stop decision (rank 0's
StopRule, which alone sees dev evals) is broadcast. OPSD (frozen / self teacher) runs per rank on its episodes.
VEF presets: every rank judges its own groups (cache ``vef_cache.rank<r>.jsonl``) with QAs drawn for the GLOBAL step;
the merged step record (identical on every rank) carries the ``vef`` summary, so the judge-down stop (checkpoint, then
SystemExit "VEF JUDGE DOWN") is taken by every rank together.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from eval_v1 import absent_sampler as AS  # noqa: E402
from eval_v1 import main_run as MR  # noqa: E402
from trainer_v2 import ddp_v2 as DV  # noqa: E402
from trainer_v2 import run_v2 as RV  # noqa: E402
from trainer_v2 import vcm_data as VD  # noqa: E402

logger = logging.getLogger(__name__)
MARKER = "TRAINER V2 RLVR MAIN DDP DONE"


def require_ddp_gpus(dist: DV.DistCtx) -> None:
    """Refuse unless CUDA_VISIBLE_DEVICES is a comma list of WORLD_SIZE distinct integer ids (PCI order, HF offline)."""
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not all(i.isdigit() for i in ids) or len(set(ids)) != len(ids) or len(ids) != dist.world:
        raise SystemExit(f"REFUSED: CUDA_VISIBLE_DEVICES must list {dist.world} distinct integer GPU ids")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("REFUSED: need CUDA_DEVICE_ORDER=PCI_BUS_ID and HF_HUB_OFFLINE=1")


def fork_sampler(s: AS.AbsentShareSampler) -> AS.AbsentShareSampler:
    """Independent copy of the sampler state (items shared, RNG / pools / counters copied)."""
    c = copy.copy(s)
    c.rng = copy.deepcopy(s.rng)
    c.pools = {k: list(v) for k, v in s.pools.items()}
    c.counts = Counter(s.counts)
    return c


def global_batch(items: list, cursor: int, p_global: int, sampler: Optional[AS.AbsentShareSampler]) -> list:
    """The step's P_global prompts, as a single-GPU run would draw them (advances ``sampler`` by P_global)."""
    if sampler is not None:
        return sampler.draw(p_global)
    return [items[(cursor + j) % len(items)] for j in range(p_global)]


def refill_stream(items: list, cursor: int, p_global: int, sampler: Optional[AS.AbsentShareSampler],
                  dist: DV.DistCtx) -> DV.StridedStream:
    """Rank's DAPO refill view of the prompts after the global batch (call after ``global_batch``)."""
    if sampler is not None:
        return DV.StridedStream(fork_sampler(sampler).draw, dist.rank, dist.world)
    base = cursor + p_global
    pos = [0]

    def nxt(k: int) -> list:
        got = [items[(base + pos[0] + j) % len(items)] for j in range(k)]
        pos[0] += k
        return got
    return DV.StridedStream(nxt, dist.rank, dist.world)


def ddp_train_one_step(items: list, step: int, args: argparse.Namespace, backend: Any, optimizer: Any, params: list,
                       cfg: Any, opsd: Any, monitor: Any, out: Path, dist: DV.DistCtx,
                       sampler: Optional[AS.AbsentShareSampler], cursor: int, loader: Any = VD.item_images,
                       cuda: bool = True, vef: Any = None) -> tuple[dict, int]:
    """One global step; returns (global step record, prompts consumed from the global stream). ``loader`` / ``cuda``
    are injectable for the CPU tests."""
    p, p_global = args.prompts_per_step, args.prompts_per_step * dist.world
    batch = global_batch(items, cursor, p_global, sampler)
    stream = refill_stream(items, cursor, p_global, sampler, dist)
    mine = DV.rank_slice(batch, dist, p)
    rec = RV.run_steps(mine, loader, backend, optimizer, params, cfg, steps=1, prompts_per_step=p, opsd=opsd,
                       cuda=cuda, monitor=monitor, refill=stream.take, dist=dist,
                       **({} if vef is None else {"start_step": step, "vef": vef}))[0]
    taken = [int(t) for t in dist.gather(stream.taken)]
    extra = DV.StridedStream.consumed(max(taken), dist.world)
    refills = stream.taken_items(taken) if sampler is not None and dist.is_main else []
    if sampler is not None:
        sampler.skip(extra)
    rec["step"] = step
    rec["ddp"] = {"world_size": dist.world, "prompts_per_rank": p, "refills_per_rank": taken}
    if "peak_mem_alloc_gb" in rec:                                     # rank 0's numbers above; every rank's here
        rec["ddp"]["peak_mem_alloc_gb_per_rank"] = dist.gather(rec["peak_mem_alloc_gb"])
    if "dapo_prompts_drawn" in rec:
        rec["n_prompts_drawn"] = p_global + extra
    if sampler is not None and dist.is_main:
        drawn = batch + refills
        rec["data"] = sampler.payload(drawn)
        if "dapo_prompts_drawn" in rec:
            cond = {it.row_id: AS.condition(it) for it in drawn}
            rows = rec.get("rows") or []
            rec["data"]["train/data/kept_absent_share"] = (sum(cond.get(r) == AS.ABSENT for r in rows) / len(rows)
                                                           if rows else None)
    if dist.is_main:
        with (out / "steps.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    return rec, p_global + extra


def decide(dist: DV.DistCtx, value: Any) -> bool:
    """Rank 0's decision (checkpoint now / stop) on every rank."""
    return bool(dist.broadcast(bool(value) if dist.is_main else None))


def save_ddp(out: Path, step: int, model: Any, optimizer: Any, args: argparse.Namespace, monitor: Any, cursor: int,
             dist: DV.DistCtx) -> None:
    """Rank 0 writes the checkpoint (main_run layout + ``ddp`` block with every rank's RNG); all ranks wait."""
    params = [q for q in model.parameters() if q.requires_grad]
    dist.sync_check(params)
    rngs = dist.gather(DV.rng_state())
    if dist.is_main:
        MR.save_checkpoint(out, step, model, optimizer, args, monitor, cursor,
                           extra={"ddp": {"world_size": dist.world, "prompts_per_rank": args.prompts_per_step,
                                          "global_prompts_per_step": args.prompts_per_step * dist.world,
                                          "rank_rng": rngs}})
    dist.barrier()


def restore_ddp(path: Path, model: Any, optimizer: Any, args: argparse.Namespace, monitor: Any,
                dist: DV.DistCtx) -> int:
    """Resume a DDP checkpoint, or a single-GPU one whose P equals the global P here. Rank r restores its own RNG
    when the checkpoint has one for it (same world size); otherwise rank 0 keeps the checkpoint's (single-GPU) RNG
    and rank r > 0 reseeds from (seed, rank, step)."""
    import torch
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    ddp = state.get("ddp") or {}
    world_ck, p_ck = int(ddp.get("world_size", 1)), int(state["args"]["prompts_per_step"])
    if p_ck * world_ck != args.prompts_per_step * dist.world:
        raise ValueError(f"global prompts per step differ: checkpoint {p_ck} x {world_ck}, "
                         f"run {args.prompts_per_step} x {dist.world}")
    shadow = argparse.Namespace(**{**vars(args), "prompts_per_step": p_ck})
    start = MR.restore_checkpoint(path, model, optimizer, shadow, monitor)
    rank_rng = ddp.get("rank_rng") or []
    if world_ck == dist.world and dist.rank < len(rank_rng):
        DV.set_rng_state(rank_rng[dist.rank])
    elif dist.rank > 0:
        DV.reseed_rank(args.seed + start, dist.rank)
    return start


def _setup_logging(dist: DV.DistCtx) -> None:
    level = logging.INFO if dist.is_main else logging.WARNING
    fmt = ("" if dist.is_main else f"[rank{dist.rank}] ") + "%(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def gpu_main(args: argparse.Namespace, dist: DV.DistCtx) -> int:  # pragma: no cover - scheduler GPU job
    import torch

    from eval_v1 import release_data as D
    from eval_v1 import stop_rule as SR
    from eval_v1 import wandb_layout as WL
    from eval_v1 import wandb_run as WR
    from trainer_v2 import grpo_v2 as G
    from trainer_v2 import metrics_v9 as MV
    from trainer_v2 import paths_v2 as P
    from trainer_v2.hf_backend import HFBackend
    require_ddp_gpus(dist)
    out = Path(args.out)
    for path in (args.out, args.teacher_table, args.prereg):
        if path:
            P.refuse_holdout_path(path)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"existing output requires --resume: {out}")
    dist.barrier()                                              # every rank checked before rank 0 writes
    if dist.is_main:
        out.mkdir(parents=True, exist_ok=True)
    ws = MR.init_lora_meta(args, out)                           # warm start (every rank checks the same files)
    if args.prereg:
        from trainer_v2 import pins
        pins.verify(Path(args.prereg), rows=D.release_file(Path(args.release), "train", args.mode))
        pins.verify_ws(Path(args.prereg), getattr(args, "init_lora", None), getattr(args, "init_lora_sha256", None))
    MR.ag_check(args)                                           # v52ws_ag: vef2 prompt schema + pin ag block
    if ws and not args.resume and dist.is_main:
        (out / MR.INIT_LORA_FILE).write_text(json.dumps(ws, indent=2) + "\n")
    RV._seed(args.seed)
    vef = MR.vef_context(args, out, dist.rank)
    items, manifest = D.load_release(Path(args.release), "train", args.mode, arms=(args.arm,),
                                     vef=vef is not None or MR.needs_match_src(args))
    if not items:
        raise ValueError("no accepted training items")
    random.Random(args.seed).shuffle(items)
    opsd = MR._opsd_context(args)
    model, proc = RV.load_model(lora=True, size=args.model, fast_conv=args.fast_conv, lora_r=args.lora_r)
    if ws and not args.resume:                                  # warm start: step 0 from the init LoRA on every rank
        MR.load_init_lora(model, ws["path"], ws["adapter_sha256"])
    backend = HFBackend(model, proc, temperature=1.0)
    params = [q for q in model.parameters() if q.requires_grad]
    cfg = G.LoopConfig(num_generations=args.num_generations, r=args.r, reward_preset=args.reward_preset,
                       opsd_lambda=args.opsd_lambda if opsd else 0.0, **RV.algo_kwargs(args),
                       **RV.rollout_kwargs(args), **RV.train_kwargs(args), **RV.CL.dapo_kwargs(args),
                       **RV.CL.opsd_mode_kwargs(args))
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    monitor = MV.OverAbstainMonitor(window=args.monitor_window, min_precision=args.min_withhold_precision,
                                    hard_stop=args.hard_stop and args.stop_rule == "legacy")
    stopper = SR.StopRule(args.stop_rule, args.hard_stop, args.stop_warmup, args.monitor_window)
    dist.barrier()
    start, cursor = 0, 0
    if args.resume:
        ck = MR.latest_checkpoint(out)
        start = restore_ddp(ck, model, optimizer, args, monitor, dist)
        cursor = MR.data_cursor(ck, start, args.prompts_per_step * dist.world)
    else:
        DV.reseed_rank(args.seed, dist.rank)
    dist.broadcast_params(params)                               # replicas start bit-identical
    sampler = None
    if args.target_absent_share is not None:
        sampler = AS.AbsentShareSampler(items, args.target_absent_share, args.seed)
        sampler.skip(cursor)
    p_global = args.prompts_per_step * dist.world
    wargs = argparse.Namespace(**{**vars(args), "prompts_per_step": p_global})       # wandb tags / config: global P
    wb, acc, best, hints = None, WL.Accumulator(), WL.BestTracker(), None
    if dist.is_main:
        if args.resume:
            for s, payload in SR.dev_history(out, start, WR.read_records,
                                             lambda r, k: WR.safe_payload(WR.dev_val_payload, r, k)):
                stopper.add_dev(s, payload)
        wb = WR.init(wargs, {**WR.run_config(wargs, cfg, manifest, start), "ddp_world_size": dist.world,
                             "P_per_rank": args.prompts_per_step}, out)
        if wb and args.resume:
            WR.safe_payload(WR.replay, out, start, acc, best)
        hints = MR._hints_by_row(items, opsd) if wb else None
        if args.resume and args.dev_every and start % args.dev_every == 0 and not (
                out / "dev_eval" / f"step_{start:06d}" / "eval_report.json").exists():
            payload = MR._dev_eval(args, start, backend, cfg, out)
            MR._log_val(wb, best, payload)
            stopper.add_dev(start, payload)
    dist.barrier()
    last = start
    for step in range(start, args.steps):
        t0 = time.time()
        rec, used = ddp_train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, dist,
                                       sampler, cursor, vef=vef)
        cursor += used
        last = step + 1
        if wb:
            WR.safe_log(wb, {**WR.safe_payload(
                WL.train_payload, rec, step=last, g=cfg.num_generations, acc=acc,
                lr=optimizer.param_groups[0]["lr"], opsd_lambda=cfg.opsd_lambda, hints_by_row=hints,
                step_s=time.time() - t0), **rec.get("data", {}), **MR.vef_wandb(rec)})
        saved = decide(dist, (args.checkpoint_every and last % args.checkpoint_every == 0) or last == args.steps
                       or monitor.stopped)
        if saved:
            save_ddp(out, last, model, optimizer, args, monitor, cursor, dist)
        if MR.vef_down(rec):                    # merged record: the same decision on every rank
            if not saved:
                save_ddp(out, last, model, optimizer, args, monitor, cursor, dist)
            raise SystemExit(f"{MR.VEF_DOWN} at step {last}: {rec['vef']}")
        if args.dev_every and last % args.dev_every == 0:
            if dist.is_main:
                payload = MR._dev_eval(args, last, backend, cfg, out)
                MR._log_val(wb, best, payload)
                stopper.add_dev(last, payload)
            dist.barrier()
        if dist.is_main:
            stop_payload = stopper.update(last, rec.get("monitor"), monitor.stopped)
            if wb:
                WR.safe_log(wb, stop_payload)
        if decide(dist, stopper.stopped):
            if not saved:
                save_ddp(out, last, model, optimizer, args, monitor, cursor, dist)
            logger.warning("over-abstain hard stop (%s) at step %d", args.stop_rule, last)
            break
    if dist.is_main:
        summary = {"step": last, "requested_steps": args.steps, "hard_stop": stopper.stopped, **stopper.summary(),
                   "absent_sampler": sampler.summary() if sampler else None, "data": manifest,
                   "config": cfg.__dict__, "ddp": {"world_size": dist.world, "prompts_per_rank": args.prompts_per_step,
                                                   "global_prompts_per_step": p_global},
                   "checkpoint": str(MR.latest_checkpoint(out)), **({"init_lora": ws} if ws else {})}
        (out / "main_report.json").write_text(json.dumps(summary, indent=2))
        if wb:
            WR.safe_summary(wb, {**summary, **best.summary()})
            try:
                wb.finish()
            except Exception as exc:  # noqa: BLE001 - logging is optional
                logger.warning("wandb finish failed: %s", exc)
        logger.info("%s arm=%s steps=%d world=%d P_global=%d out=%s%s", MARKER, args.arm, last, dist.world, p_global,
                    out, f" stopped={SR.REASON}" if stopper.stopped else "")
    dist.barrier()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = MR.parse(argv)
    dist = DV.init_from_env()
    if dist is None:
        raise SystemExit("REFUSED: launch with torchrun --nproc_per_node >= 2 (single GPU: eval_v1.main_run)")
    _setup_logging(dist)
    try:
        return gpu_main(args, dist)
    finally:
        DV.shutdown()


if __name__ == "__main__":
    sys.exit(main())
