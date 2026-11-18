#!/usr/bin/env python3
"""trainer_v2 entry point. ``dry-run`` is CPU (mock policy). ``pass6`` / ``opsd`` are GPU jobs that the scheduler runs
through pass6_smoke.sh / opsd_smoke.sh only (single GPU, CUDA_VISIBLE_DEVICES = one integer, HF offline).

pass6: ONE real optimizer step per requested arm (RLVR, reward v5) on Qwen3.5-4B + LoRA (trainer_v1 LoraSpec) with
the 9-image VCM prompts; reports peak GPU memory, tokens per prompt / episode and step time (rollout vs train).
opsd:  a few steps with L = L_RLVR + lambda * L_distill (teacher = same model, LoRA disabled, hinted prompt).
Micro-batch: 1 episode per forward/backward by default (9-image sequences are ~4-6k tokens; --train-micro-batch
groups episodes, see train_mb.py), gradient accumulation over the
step's P x G episodes (default P=1, G=8: the SPEC M5 "per_device_bs 2 x grad_accum 4" budget, made more
conservative), gradient checkpointing on, logits only at model-token positions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2 import cli_v2 as CL  # noqa: E402
from trainer_v2 import grpo_v2 as G  # noqa: E402
from trainer_v2.cli_v2 import flat_scalars, wandb_init  # noqa: E402,F401 (re-exported)
from trainer_v2 import metrics_v9 as MV  # noqa: E402
from trainer_v2 import opsd as OP  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402
from trainer_v2 import reward_v5 as R5  # noqa: E402
from trainer_v2 import vcm_data as VD  # noqa: E402

logger = logging.getLogger(__name__)
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "task_type": "CAUSAL_LM",  # = trainer_v1 config.LoraSpec
        "target_modules": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"}
EPISODE_KEYS = ("reward", "advantage", "success", "truncated", "entropy", "clip_frac", "is_cap_frac", "is_ratio_mean",
                "ratio_mean", "n_prompt", "n_completion", "n_model_tokens", "kl", "n_distill_tokens", "opsd_gate", "action_pred")
STEP_EXTRA_KEYS = ("dapo_filtered_groups", "dapo_resample_rounds", "dapo_prompts_drawn", "dapo_kept_groups",
                   "dapo_no_informative", "overlong_penalty_mean", "opsd_self_teacher", "t_teacher_s")
MARKERS = {"pass6": "TRAINER V2 PASS6 DONE", "opsd": "TRAINER V2 OPSD SMOKE DONE", "dry-run": "TRAINER V2 DRY RUN OK",
           "mini": "TRAINER V2 RLVR MINI DONE"}


def require_single_gpu() -> None:
    """Refuse unless exactly one GPU is pinned by integer id in PCI bus order and the hub is offline."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must be a single integer GPU id")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("REFUSED: need CUDA_DEVICE_ORDER=PCI_BUS_ID and HF_HUB_OFFLINE=1")
    import torch
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"REFUSED: {torch.cuda.device_count()} visible GPUs (need exactly 1)")


def load_model(lora: bool, size: str = "4b", fast_conv: bool = False,
               lora_r: int = 16) -> tuple[Any, Any]:  # pragma: no cover - GPU
    """``fast_conv``: Gated DeltaNet causal conv on fla's triton kernels (fast_conv.py); False = transformers' torch
    fallback (unchanged). ``lora_r``: LoRA rank, alpha = 2 * rank (16 = the LORA default, unchanged)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid, rev = P.MODELS[size]
    proc = AutoProcessor.from_pretrained(mid, revision=rev)
    model = AutoModelForImageTextToText.from_pretrained(mid, revision=rev, dtype=torch.bfloat16).to("cuda")
    if fast_conv:
        from trainer_v2 import fast_conv as FC
        FC.install(model=model)
    if not lora:
        return model.eval().requires_grad_(False), proc
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**lora_spec(lora_r)))
    return model.train(), proc


def lora_spec(rank: int = 16) -> dict:
    """LORA with rank ``rank`` and alpha = 2 * rank (rank 16 returns LORA exactly)."""
    return {**LORA, "r": rank, "lora_alpha": 2 * rank}


def subset(items: list, limit: Optional[int]) -> list:
    """Per arm, at most ``limit`` items, round-robin over row types (keeps H1/H2/H3-eligible rows in tiny subsets)."""
    if not limit:
        return items
    out = []
    for arm in dict.fromkeys(it.arm for it in items):
        by_type: dict[str, list] = {}
        for it in items:
            if it.arm == arm:
                by_type.setdefault(str(it.row.get("row_type")), []).append(it)
        picked, k = [], 0
        while len(picked) < limit and any(len(v) > k for v in by_type.values()):
            picked += [v[k] for v in by_type.values() if len(v) > k][:limit - len(picked)]
            k += 1
        out += picked
    return out


def longest_first(items: list) -> list:
    """Worst case first (memory smoke): order by rendered prompt + env text length, ties by row id."""
    size = {id(it): len(json.dumps([it.messages, it.env_call])) for it in items}
    return sorted(items, key=lambda it: (-size[id(it)], it.row_id))


def run_steps(items: list, loader: Any, backend: Any, optimizer: Any, params: list, cfg: G.LoopConfig, *,
              steps: int, prompts_per_step: int, opsd: Optional[OP.OpsdContext] = None, cuda: bool = False,
              monitor: Optional[MV.OverAbstainMonitor] = None, step_log: Optional[Path] = None,
              on_step: Optional[Callable[[dict], None]] = None,
              refill: Optional[Callable[[int], list]] = None, dist: Optional[Any] = None,
              start_step: int = 0, vef: Optional[Any] = None) -> list:
    """Fixed-order steps over ``items`` (one arm); images come lazily from the set_transform dataset. ``monitor``
    (WP-T5) logs withhold metrics per step and stops the run on a hard stop; ``step_log`` appends one JSON per step;
    ``on_step`` receives each step record (wandb). DAPO dynamic sampling draws extra prompts from ``refill(k)`` (e.g.
    the absent-share sampler), else from the fixed order after the step's batch (the next step continues after
    them); ``n_prompts_drawn`` then records the prompts the step consumed. ``dist`` (ddp_v2.DistCtx, torchrun) is
    passed to train_step: ``items`` are then this rank's prompts and the step record is merged over ranks.
    ``start_step``: global index of the first step (eval_v1 runs one step per call); ``vef`` (vef.VefContext, VEF
    presets): passed to train_step with the global step; the record then carries the ``vef`` step summary."""
    ds = VD.lazy_dataset(items, loader=loader)
    log = []
    cursor = 0
    for step in range(steps):
        idx = [(cursor + j) % len(items) for j in range(prompts_per_step)]
        batch = [items[i] for i in idx]
        images = [ds[i:i + 1]["images"][0] for i in idx]       # lazy: loaded for this batch only
        extra: list = []

        def draw(k: int, base: int = cursor + prompts_per_step) -> tuple[list, list]:
            got = refill(k) if refill is not None else [items[(base + len(extra) + j) % len(items)] for j in range(k)]
            extra.extend(got)
            return got, [loader(it) for it in got]
        if cuda:  # pragma: no cover - GPU
            import torch
            torch.cuda.reset_peak_memory_stats()
        res = G.train_step(batch, images, backend, optimizer, params, cfg, opsd=opsd, refill=draw,
                           **({} if dist is None else {"dist": dist}),
                           **({} if vef is None else {"vef": vef, "step": start_step + step}))
        cursor += prompts_per_step + (len(extra) if refill is None else 0)
        batch = res.pop("used_items", batch)                   # DAPO: the kept prompts
        eps = res["episodes"]
        rec = {"step": step, "arm": batch[0].arm, "rows": [it.row_id for it in batch],
               "n_images_per_prompt": [len(it.image_paths) for it in batch],
               "prompt_tokens": sorted({e["n_prompt"] for e in eps}),
               "completion_tokens_max": max(e["n_completion"] for e in eps),
               "episode_tokens_max": max(e["n_prompt"] + e["n_completion"] for e in eps),
               "model_tokens_mean": sum(e["n_model_tokens"] for e in eps) / len(eps),
               **{k: res[k] for k in ("reward_mean", "rewards", "grad_norm", "opsd_gate_rate", "t_rollout_s",
                                      "t_train_s")},
               **{k: res[k] for k in ("grad_norms", "updates_per_batch") if k in res},      # mu > 1 only
               **{k: res[k] for k in STEP_EXTRA_KEYS if k in res},                          # DAPO / self teacher
               **({"n_prompts_drawn": prompts_per_step + len(extra)} if "dapo_prompts_drawn" in res else {}),
               "kl_mean": _mean([e["kl"] for e in eps if e["kl"] is not None]),
               "distill_tokens_mean": _mean([e["n_distill_tokens"] for e in eps]),
               "decisions": [e["decision"] for e in eps], "gate_reasons": [e["gate_reasons"] for e in eps],
               "components": res["components"], "metrics": MV.summarize(res["records"]),
               "episodes": [{k: e.get(k) for k in EPISODE_KEYS} for e in eps],      # wandb training dynamics
               **({} if vef is None else {"vef": vef.observe(res["vef_raw"])}),
               **({"ag": _ag_summary(res["records"])} if any("ag" in r for r in res["records"]) else {})}  # v52ws_ag
        if monitor is not None:
            rec["monitor"] = monitor.update(res["records"])
        if cuda:  # pragma: no cover - GPU
            import torch
            rec["peak_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            rec["peak_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        logger.info("STEP %s", json.dumps({k: v for k, v in rec.items()
                                           if k not in ("rewards", "gate_reasons", "components", "metrics",
                                                        "episodes")}))
        if step_log is not None:
            with open(step_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        if on_step is not None:
            on_step(rec)
        log.append(rec)
        if monitor is not None and monitor.stopped:
            logger.warning("OVER-ABSTAIN HARD STOP at step %d: %s", step, rec["monitor"])
            break
    return log


def _mean(xs: list) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def _ag_summary(records: list[dict]) -> dict:
    """v52ws_ag step summary (trainer_v2/withhold_ag.py): grounded-withhold rate, Match accuracy, unneeded-call rate."""
    from trainer_v2.withhold_ag import summary
    return summary(records)


def algo_kwargs(args: argparse.Namespace) -> dict:
    """LoopConfig policy-loss fields from CLI args (also used by eval_v1.main_run)."""
    return {"algo": args.algo, "updates_per_batch": args.updates_per_batch, "clip_eps_low": args.clip_eps_low,
            "clip_eps_high": args.clip_eps_high}


def add_algo_args(ap: argparse.ArgumentParser) -> None:
    """--algo / --updates-per-batch / --clip-eps-low / --clip-eps-high (defaults = the pre-GSPO GRPO loop)."""
    ap.add_argument("--algo", default="grpo", choices=G.PL.ALGOS, help="grpo: token ratio; gspo: sequence ratio")
    ap.add_argument("--updates-per-batch", type=int, default=1, help="mu: optimizer updates per rollout batch")
    ap.add_argument("--clip-eps-low", type=float, default=None, help="default: grpo 0.2, gspo 3e-4")
    ap.add_argument("--clip-eps-high", type=float, default=None, help="default: grpo 0.2, gspo 4e-4")


def rollout_kwargs(args: argparse.Namespace) -> dict:
    """LoopConfig rollout-batching fields from CLI args (also used by eval_v1.main_run); absent args = defaults."""
    return {"batch_rollout": bool(getattr(args, "batch_rollout", False)), "gen_batch": getattr(args, "gen_batch", 16)}


def add_rollout_args(ap: argparse.ArgumentParser) -> None:
    """--batch-rollout / --gen-batch (defaults = the sequential rollout). 9B on one A40: keep --gen-batch small."""
    ap.add_argument("--batch-rollout", action="store_true",
                    help="batched Stage-1 / Stage-2 / passive generation (left padding); same distribution, "
                         "different RNG consumption order than the sequential rollout")
    ap.add_argument("--gen-batch", type=int, default=16, help="batched rollout: max output rows per generate call")


def train_kwargs(args: argparse.Namespace) -> dict:
    """LoopConfig training-forward field from CLI args (also used by eval_v1.main_run); absent arg = 1."""
    return {"train_micro_batch": getattr(args, "train_micro_batch", 1)}


def add_train_args(ap: argparse.ArgumentParser) -> None:
    """--train-micro-batch (default 1 = one episode per forward/backward, the pre-micro-batching loop)."""
    ap.add_argument("--train-micro-batch", type=int, default=1,
                    help="episodes per policy/teacher forward (length-grouped, left-padded); same loss weighting")
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank (alpha = 2 * rank); 16 = previous runs")


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)


def _items(args: argparse.Namespace, arms: tuple[str, ...]) -> tuple[list, dict]:
    items, manifest = VD.load_items(Path(args.rows), arms=arms, mode=args.mode, drop_p=args.drop_p, order=args.order,
                                    require_accepted=not args.allow_unaccepted, splits=tuple(args.splits.split(",")))
    return items, manifest


def gpu_main(args: argparse.Namespace) -> int:  # pragma: no cover - GPU job
    import torch
    from trainer_v2.hf_backend import HFBackend
    require_single_gpu()
    out = Path(args.out)
    P.refuse_holdout_path(out)
    out.mkdir(parents=True, exist_ok=True)
    pins = None
    if args.prereg:
        from trainer_v2 import pins as PN
        pins = PN.verify(Path(args.prereg), rows=Path(args.rows))
    _seed(args.seed)
    arms = tuple(args.arms.split(","))
    items, manifest = _items(args, arms)
    opsd = None
    if args.cmd == "opsd":
        opsd = OP.OpsdContext.load(Path(args.teacher_table), args.hint_set, force=args.force_gate)
        keep = (lambda e: e is not None) if args.force_gate else (lambda e: bool(e and e.get("teacher_ok")))
        items = [it for it in items if keep(opsd.table.get(OP.table_key(it))) and OP.build_hint(it, opsd.hint_ids)[0]]
        logger.info("opsd items (table %s, force_gate=%s): %d", args.teacher_table, args.force_gate, len(items))
    elif args.cmd == "mini" and args.teacher_table:
        opsd = OP.OpsdContext.load(Path(args.teacher_table), args.hint_set)       # real selective gate
    items = subset(items, args.limit)
    if args.cmd == "mini":
        random.Random(args.seed).shuffle(items)                                   # seeded order over row types
    if args.pick == "longest":
        items = longest_first(items)
    t0 = time.time()
    model, proc = load_model(lora=True, fast_conv=args.fast_conv)
    backend = HFBackend(model, proc, temperature=1.0)
    params = [p for p in model.parameters() if p.requires_grad]
    cfg = G.LoopConfig(num_generations=args.num_generations, r=args.r, reward_preset=args.reward_preset,
                       opsd_lambda=args.opsd_lambda if opsd is not None else 0.0, **algo_kwargs(args),
                       **rollout_kwargs(args), **train_kwargs(args), **CL.dapo_kwargs(args),
                       **CL.opsd_mode_kwargs(args))
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    report = {"cmd": args.cmd, "config": cfg.__dict__, "data": manifest, "lora": LORA, "gpu": torch.cuda.get_device_name(0),
              "model_load_s": round(time.time() - t0, 1), "load_peak_mem_gb":
              round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "arms": {}, "prereg_verified": pins is not None,
              "opsd": {"table": args.teacher_table, "hint_set": args.hint_set} if opsd is not None else None}
    wb = wandb_init(args, {k: v for k, v in report.items() if k != "arms"}, out)
    ok = True
    for arm in arms:
        mine = [it for it in items if it.arm == arm]
        on_step = None
        if wb is not None:
            def on_step(rec: dict, arm: str = arm) -> None:
                wb.log({f"{arm}/{k}": v for k, v in flat_scalars(rec).items()})
        try:
            mon = MV.OverAbstainMonitor(window=args.monitor_window, min_precision=args.min_withhold_precision,
                                        hard_stop=args.hard_stop)
            report["arms"][arm] = run_steps(mine, VD.item_images, backend, opt, params, cfg, steps=args.steps,
                                            prompts_per_step=args.prompts_per_step, opsd=opsd, cuda=True,
                                            monitor=mon, step_log=out / f"steps_{arm}.jsonl", on_step=on_step)
            report.setdefault("hard_stop", {})[arm] = mon.stopped
        except Exception as exc:  # noqa: BLE001 - report per arm, then fail the job
            logger.exception("arm %s failed", arm)
            report["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
            ok = False
        torch.cuda.empty_cache()
    if args.cmd == "mini" and ok:
        model.save_pretrained(str(out / "lora_final"))
        torch.save({"optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(), "steps": args.steps,
                    "config": cfg.__dict__}, out / "trainer_state.pt")
        report["checkpoint"] = str(out / "lora_final")
    (out / f"{args.cmd}_report.json").write_text(json.dumps(report, indent=1, default=str))
    if wb is not None:
        wb.summary.update({"ok": ok, "hard_stop": report.get("hard_stop"), "checkpoint": report.get("checkpoint")})
        wb.finish(exit_code=0 if ok else 1)
    for arm, log in report["arms"].items():
        if isinstance(log, list):
            print(f"ARM {arm}: peak_alloc_gb={max(r['peak_mem_alloc_gb'] for r in log)} "
                  f"prompt_tokens={log[0]['prompt_tokens']} episode_tokens_max={max(r['episode_tokens_max'] for r in log)}"
                  f" step_s={[round(r['t_rollout_s'] + r['t_train_s'], 1) for r in log]} "
                  f"gate={[r['opsd_gate_rate'] for r in log]} reward={[round(r['reward_mean'], 3) for r in log]} "
                  f"monitor_last={log[-1].get('monitor')}", flush=True)
    if not ok:
        print("TRAINER V2 FAILED (see report)", flush=True)
        return 1
    print(f"{MARKERS[args.cmd]} arms={','.join(arms)} steps={args.steps} report={out / f'{args.cmd}_report.json'}",
          flush=True)
    return 0


def dry_run(args: argparse.Namespace) -> dict:
    """CPU end to end with the mock policy: one RLVR step per arm, then a mock teacher table and one OPSD step."""
    import torch

    from trainer_v2 import mock_v2 as M
    from trainer_v2 import teacher_table as TT
    _seed(args.seed)
    items, manifest = _items(args, ("passive", "memory_call"))
    items = subset(items, args.limit or 4)
    good = M.completion("unsafe", "direct_answer", "Close the window first.", "[1]")
    backend = M.MockBackend([good, M.completion("safe", "clarify", "Which one is it?"), "garbage " * 3])
    params = list(backend.policy.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    cfg = G.LoopConfig(num_generations=4, r=0.1, opsd_lambda=OP.LAMBDA)
    rl_cfg = G.LoopConfig(num_generations=4, r=0.1)
    out = {"data": manifest, "arms": {}}
    for arm in ("passive", "memory_call"):
        mine = [it for it in items if it.arm == arm]
        out["arms"][arm] = run_steps(mine, M.fake_images, backend, opt, params, rl_cfg, steps=1, prompts_per_step=1)
    table = TT.build_table(items, M.fake_images, backend, cfg, "main", {"mock": True})
    ctx = OP.OpsdContext(OP.HINT_SETS["main"], table["entries"], force=True)
    out["teacher_table"] = table["summary"]
    hinted = [it for it in items if OP.build_hint(it, ctx.hint_ids)[0] is not None]
    for arm in ("passive", "memory_call"):
        out[f"opsd_{arm}"] = run_steps([it for it in hinted if it.arm == arm], M.fake_images, backend, opt, params,
                                       cfg, steps=1, prompts_per_step=1, opsd=ctx)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=sorted(MARKERS))
    ap.add_argument("--rows", default=str(P.CANARY_RENDERED))
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="passive,memory_call")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--prompts-per-step", type=int, default=1)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="items per arm (round-robin over row types)")
    ap.add_argument("--r", type=float, default=None, help="memory-call decision magnitude (required for that arm)")
    ap.add_argument("--reward-preset", default="main", choices=sorted(R5.PRESETS))
    ap.add_argument("--mode", default="retrieved", choices=VD.CONTEXT_MODES)
    ap.add_argument("--drop-p", type=float, default=0.2)
    ap.add_argument("--order", default="display", choices=VD.ORDERS)
    ap.add_argument("--allow-unaccepted", action="store_true")
    ap.add_argument("--splits", default="train", help="comma list from train,dev (training uses train only)")
    ap.add_argument("--teacher-table")
    ap.add_argument("--hint-set", default="main", choices=sorted(OP.HINT_SETS))
    ap.add_argument("--opsd-lambda", type=float, default=OP.LAMBDA)
    ap.add_argument("--force-gate", action="store_true", help="smoke plumbing: gate every hinted episode")
    ap.add_argument("--pick", default="first", choices=["first", "longest"], help="item order within each arm")
    ap.add_argument("--seed", type=int, default=0)
    add_algo_args(ap)
    add_rollout_args(ap)
    add_train_args(ap)
    CL.add_dapo_args(ap)
    CL.add_opsd_mode_args(ap)
    ap.add_argument("--prereg", default=None, help="PREREG_TRAINER_V2.json: refuse if a pinned sha differs")
    ap.add_argument("--fast-conv", action="store_true", default=os.environ.get("FAST_CONV") == "1",
                    help="fla triton causal conv (fast_conv.py; env FAST_CONV=1); default torch")
    ap.add_argument("--monitor-window", type=int, default=5)
    ap.add_argument("--min-withhold-precision", type=float, default=0.7)
    ap.add_argument("--hard-stop", action="store_true", help="stop when window withhold precision < threshold")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "EAPMA"))
    ap.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-name", default=None, help="default: <cmd>_<out dir name>")
    ap.add_argument("--no-wandb", action="store_true", help="GPU jobs log to wandb unless this is set")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "dry-run":
        res = dry_run(args)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=1, default=str))
        print(f"{MARKERS['dry-run']} arms={list(res['arms'])} table={res['teacher_table']}", flush=True)
        return 0
    if args.out is None or ("memory_call" in args.arms and args.r is None):
        raise SystemExit("GPU path needs --out, and --r for the memory_call arm (no default)")
    if args.cmd == "opsd" and not args.teacher_table:
        raise SystemExit("opsd needs --teacher-table (OPSD without a table would gate every row off)")
    return gpu_main(args)


if __name__ == "__main__":
    sys.exit(main())
