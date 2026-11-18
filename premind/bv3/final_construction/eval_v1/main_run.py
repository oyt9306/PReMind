#!/usr/bin/env python3
"""Release-based RLVR main runner with resumable LoRA, optimizer and RNG state."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from eval_v1 import absent_sampler as AS  # noqa: E402
from eval_v1 import evaluate as EV  # noqa: E402
from eval_v1 import release_data as D  # noqa: E402
from eval_v1 import stop_rule as SR  # noqa: E402
from eval_v1 import wandb_layout as WL  # noqa: E402
from eval_v1 import wandb_run as WR  # noqa: E402
from trainer_v2 import grpo_v2 as G  # noqa: E402
from trainer_v2 import metrics_v9 as MV  # noqa: E402
from trainer_v2 import opsd as OP  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402
from trainer_v2 import reward_v5 as R5  # noqa: E402
from trainer_v2 import run_v2 as RV  # noqa: E402
from trainer_v2 import vcm_data as VD  # noqa: E402

logger = logging.getLogger(__name__)
MARKER = "TRAINER V2 RLVR MAIN DONE"
# stop-rule flags change when the run stops, not what it trains: a resume may switch them (pilot -> confirmed rule)
# fast_conv is a kernel choice (same math, fp-level differences), so a resume may switch it too
# the VEF judge URL (host / port of the judge server) does not change what is trained
RESUME_EXEMPT = ("resume", "steps", "stop_rule", "stop_warmup", "monitor_window", "fast_conv", "vef_judge_url")
VEF_DOWN = "VEF JUDGE DOWN"


# args added after older checkpoints were written, at the value those checkpoints trained with (pre-GSPO = GRPO, mu 1)
LEGACY_ARG_DEFAULTS = {"target_absent_share": None, "algo": "grpo", "updates_per_batch": 1, "clip_eps_low": None,
                       "clip_eps_high": None, "batch_rollout": False, "gen_batch": 16,
                       "train_micro_batch": 1, "lora_r": 16, "no_dynamic_sampling": False, "dapo_max_prompts_factor": 3.0,
                       "overlong_buffer": 128, "overlong_penalty": 1.0, "opsd_teacher": "frozen", "opsd_gate": None,
                       "opsd_only_failed": False, "vef_judge_url": None, "vef_judge_model": None}


def _comparable(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in RESUME_EXEMPT and not k.startswith("kret_")}


def latest_checkpoint(out: Path) -> Path:
    P.refuse_holdout_path(out)
    checkpoints = sorted((out / "checkpoints").glob("step_*"),
                         key=lambda p: int(p.name.removeprefix("step_")))
    valid = [p for p in checkpoints if (p / "COMPLETE").is_file()]
    if not valid:
        raise FileNotFoundError(f"no complete checkpoint under {out}")
    return valid[-1]


def save_checkpoint(out: Path, step: int, model: object, optimizer: object, args: argparse.Namespace,
                    monitor: MV.OverAbstainMonitor, cursor: int | None = None, extra: dict | None = None) -> Path:
    """``extra``: additional trainer_state entries (eval_v1.ddp_main: the ``ddp`` block); None = unchanged."""
    import torch
    target = out / "checkpoints" / f"step_{step:06d}"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    model.save_pretrained(str(target / "lora"))
    state = {"step": step, "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
             "monitor_hist": list(monitor.hist), "monitor_stopped": monitor.stopped, "data_cursor": cursor,
             "args": {k: v for k, v in vars(args).items() if k not in ("resume", "steps") and not k.startswith("kret_")}}
    if extra:
        state.update(extra)
    torch.save(state, target / "trainer_state.pt")
    (target / "COMPLETE").write_text(f"step={step}\n")
    return target


def restore_checkpoint(path: Path, model: object, optimizer: object, args: argparse.Namespace,
                       monitor: MV.OverAbstainMonitor) -> int:
    import torch
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {path}")
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    # checkpoints older than --target-absent-share trained with the uniform order (= None)
    if _comparable({**LEGACY_ARG_DEFAULTS, **state["args"]}) != _comparable(vars(args)):
        raise ValueError("checkpoint configuration differs from current run")
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(path / "lora" / "adapter_model.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    monitor.hist.extend(state["monitor_hist"])
    # confirmed rule: the monitor only logs, so a legacy-tripped flag from an older checkpoint is not carried over
    monitor.stopped = state["monitor_stopped"] and getattr(args, "stop_rule", "legacy") == "legacy"
    return int(state["step"])


INIT_LORA_FILE = "init_lora.json"


def check_init_lora(args: argparse.Namespace) -> dict:
    """Warm start (--init-lora = <run>/checkpoints/step_N/lora): refuse (ValueError) unless the step dir carries the
    COMPLETE marker and trainer_state.pt, and the adapter matches this run's LoRA (rank, alpha, target modules, base
    model id) and base-model revision (the source run's model size + its PREREG model pin = the current pin)."""
    import torch
    lora = Path(args.init_lora).resolve()
    P.refuse_holdout_path(lora)
    step_dir = lora.parent
    if not (step_dir / "COMPLETE").is_file() or not (step_dir / "trainer_state.pt").is_file():
        raise ValueError(f"init LoRA {lora}: {step_dir} has no COMPLETE marker / trainer_state.pt")
    if not (lora / "adapter_model.safetensors").is_file() or not (lora / "adapter_config.json").is_file():
        raise ValueError(f"init LoRA {lora}: no adapter_model.safetensors / adapter_config.json")
    cfg = json.loads((lora / "adapter_config.json").read_text())
    want = RV.lora_spec(args.lora_r)
    mid, rev = P.MODELS[args.model]
    bad = [k for k in ("r", "lora_alpha", "target_modules") if cfg.get(k) != want[k]]
    bad += ["base_model"] if cfg.get("base_model_name_or_path") != mid or cfg.get("revision") not in (None, rev) else []
    src = torch.load(step_dir / "trainer_state.pt", map_location="cpu", weights_only=False, mmap=True)["args"]
    bad += [k for k in ("model", "lora_r") if src.get(k) != getattr(args, k)]
    pin = Path(src["prereg"]) if src.get("prereg") else None
    if pin is None or not pin.is_file() or json.loads(pin.read_text())["models"].get(args.model) != {"id": mid,
                                                                                                      "revision": rev}:
        bad.append("base_model_revision")
    if bad:
        raise ValueError(f"init LoRA {lora} differs from this run's config: {bad}")
    return {"path": str(lora), "adapter_sha256": P.sha256_file(lora / "adapter_model.safetensors"),
            "source_complete": (step_dir / "COMPLETE").read_text().strip(), "source_arm": src.get("arm"),
            "source_out": src.get("out"), "lora_r": args.lora_r, "model": [mid, rev]}


def init_lora_meta(args: argparse.Namespace, out: Path) -> dict | None:
    """Fresh warm-start run: validate --init-lora; resume: the run's recorded ``init_lora.json``. Sets
    ``args.init_lora`` / ``args.init_lora_sha256`` (checkpoint args, wandb config); None (args untouched) otherwise."""
    if getattr(args, "init_lora", None):
        meta = check_init_lora(args)
    elif args.resume and (out / INIT_LORA_FILE).is_file():
        meta = json.loads((out / INIT_LORA_FILE).read_text())
    else:
        return None
    args.init_lora, args.init_lora_sha256 = meta["path"], meta["adapter_sha256"]
    return meta


def load_init_lora(model: object, lora: str, sha: str) -> None:
    """Copy the adapter weights into the (fresh) PEFT model; refuse on a changed file or a key mismatch."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    f = Path(lora) / "adapter_model.safetensors"
    if P.sha256_file(f) != sha:
        raise ValueError(f"init LoRA {f} changed since it was checked")
    res = set_peft_model_state_dict(model, load_file(str(f)))
    missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
    if missing or getattr(res, "unexpected_keys", []):
        raise ValueError(f"init LoRA key mismatch: missing {missing[:3]} unexpected {list(res.unexpected_keys)[:3]}")


def data_cursor(path: Path, start: int, prompts_per_step: int) -> int:
    """Prompts consumed before ``start`` (DAPO dynamic sampling draws a variable number per step); checkpoints without
    the field (or with None) consumed exactly ``start * P``."""
    import torch
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    cur = state.get("data_cursor")
    return start * prompts_per_step if cur is None else int(cur)


def needs_match_src(args: argparse.Namespace) -> bool:
    """v52ws_ag (V5Flags.withhold_match): items need ``row["vef_src"]`` (the Match parser's memory table)."""
    return R5.PRESETS[getattr(args, "reward_preset", "main")].withhold_match


def ag_check(args: argparse.Namespace) -> None:
    """v52ws_ag: refuse unless the prompts render the vef2 schema (the Match line) and a given pin has the ag block."""
    if not needs_match_src(args):
        return
    from final_construction.train_interface_v1 import final_prompts as FP
    from trainer_v2 import vef2 as VF2
    if FP.SCHEMA != VF2.SCHEMA:
        raise SystemExit(f"REFUSED: reward preset {args.reward_preset} needs CONFIDANT_SCHEMA={VF2.SCHEMA} "
                         f"(got {FP.SCHEMA})")
    if args.prereg:
        from trainer_v2 import pins
        pins.verify_ag(Path(args.prereg), args.reward_preset)


def _dev_eval(args: argparse.Namespace, step: int, backend: object, cfg: G.LoopConfig, out: Path) -> dict:
    """Write the dev eval; return its wandb ``val/*`` payload ({} if building it fails)."""
    dev, man = D.load_release(Path(args.release), "dev", args.mode, arms=(args.arm,), limit=args.dev_limit,
                              **({"vef": True} if needs_match_src(args) else {}))
    recs, texts = EV.generate({args.mode: dev}, backend, cfg, samples=0, temperature=.7, strata_map={})
    EV.write(out / "dev_eval" / f"step_{step:06d}", recs, texts,
             {"step": step, "data": man, "arm": args.arm}, draws=args.bootstrap, seed=args.seed)
    kret = _kret_dev_eval(args, step, backend, cfg, out) if args.kret_dev else None
    return WR.safe_payload(WR.dev_val_payload, recs, step, kret)


def _kret_dev_eval(args: argparse.Namespace, step: int, backend: object, cfg: G.LoopConfig, out: Path) -> list:
    """Optional K-retention check on the 72 open text-track dev rows (evaluation only; off by default)."""
    from eval_v1 import kret_data as KD
    from eval_v1 import kret_eval as KE
    target = out / "kret_dev" / f"step_{step:06d}"
    if (target / "kret_report.json").exists():  # resume: already written (write-once)
        return WR.read_records(target / "records.jsonl")
    items, man = KD.load_kret("dev", variants=tuple(args.kret_dev_variants.split(",")))
    recs, texts = KE.generate(items, backend, max_tokens=cfg.max_completion_length)
    KE.write(target, recs, texts, {"step": step, "data": man, "arm": args.arm},
             draws=args.bootstrap, seed=args.seed)
    return recs


def _log_val(wb: object, best: WL.BestTracker, payload: dict) -> None:
    if wb and payload:
        WR.safe_log(wb, payload)
        WR.safe_summary(wb, WR.safe_payload(best.update, payload))


def _hints_by_row(items: list[VD.VcmItem], opsd: object) -> dict | None:
    if opsd is None:
        return None
    return WR.safe_payload(lambda: {it.row_id: OP.build_hint(it, opsd.hint_ids)[1]["hints"] for it in items}) or None


def _opsd_context(args: argparse.Namespace) -> OP.OpsdContext | None:
    """OPSD context: the teacher table when given (required by the table gate), else the hint set only (hint gate)."""
    if not args.opsd:
        return None
    if args.teacher_table:
        return OP.OpsdContext.load(Path(args.teacher_table), args.hint_set)
    return OP.OpsdContext(OP.HINT_SETS[args.hint_set], {})


def vef_context(args: argparse.Namespace, out: Path, rank: int = 0) -> object | None:
    """VEF presets: judge client (cache ``<OUT>/vef_cache.rank<r>.jsonl``) + health check (+ the PREREG's VEF pins);
    None for every other preset. Refuses (SystemExit) when the judge is unreachable or fails its calibration."""
    flags = R5.PRESETS[args.reward_preset]
    if not flags.vef:
        return None
    from trainer_v2 import vef as VF
    from trainer_v2 import vef_judge as VJ
    judge = VJ.VefJudge(args.vef_judge_url, args.vef_judge_model, out / f"vef_cache.rank{rank}.jsonl")
    served = judge.health_check()
    if args.prereg:
        from trainer_v2 import pins
        pins.verify_vef(Path(args.prereg), served, flags.vef)
    if flags.vef not in VF.PRESETS:                       # VEF v2 (general_mix_v1/DESIGN_v2.md): scene + Match part
        from trainer_v2 import vef2 as VF2
        return VF2.Vef2Context(flags.vef, args.seed, judge)
    return VF.VefContext(flags.vef, args.seed, judge)


def vef_down(rec: dict) -> bool:
    return bool((rec.get("vef") or {}).get("judge_down"))


def vef_wandb(rec: dict) -> dict:
    if not rec.get("vef"):
        return {}
    from trainer_v2 import vef as VF
    return VF.wandb_scalars(rec["vef"])


def train_one_step(items: list[VD.VcmItem], step: int, args: argparse.Namespace, backend: object,
                   optimizer: object, params: list, cfg: G.LoopConfig, opsd: object,
                   monitor: MV.OverAbstainMonitor, out: Path, sampler: AS.AbsentShareSampler | None = None,
                   cursor: int | None = None, vef: object | None = None) -> dict:
    """Use the frozen trainer loop with a rotated input to retain global step order; with ``sampler`` the step's
    P prompts are drawn by the absent-share sampler instead (run_steps(steps=1) takes the first P items).
    ``cursor`` = prompts consumed so far (default step * P). DAPO dynamic sampling: extra prompts continue the
    rotation, or are drawn from the SAME sampler stream (so its error diffusion covers every drawn prompt); the
    payload then reports the share over all drawn prompts and ``kept_absent_share`` over the prompts trained on.
    ``vef`` (VEF presets): passed with the global ``step`` (QA sampling key); the record carries the ``vef`` summary."""
    cursor = step * args.prompts_per_step if cursor is None else cursor
    drawn: list = []
    refill = None
    if sampler is None:
        offset = cursor % len(items)
        ordered = items[offset:] + items[:offset]
    else:
        ordered = sampler.draw(args.prompts_per_step)
        drawn = list(ordered)

        def refill(k: int) -> list:
            got = sampler.draw(k)
            drawn.extend(got)
            return got
    rec = RV.run_steps(ordered, VD.item_images, backend, optimizer, params, cfg, steps=1,
                       prompts_per_step=args.prompts_per_step, opsd=opsd, cuda=True,
                       monitor=monitor, refill=refill, **({} if vef is None else {"start_step": step, "vef": vef}))[0]
    rec["step"] = step
    if sampler is not None:
        rec["data"] = sampler.payload(drawn)
        if "dapo_prompts_drawn" in rec:
            cond = {it.row_id: AS.condition(it) for it in drawn}
            rows = rec.get("rows") or []
            rec["data"]["train/data/kept_absent_share"] = (sum(cond.get(r) == AS.ABSENT for r in rows) / len(rows)
                                                           if rows else None)
    with (out / "steps.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def gpu_main(args: argparse.Namespace) -> int:  # pragma: no cover - scheduler GPU job
    import torch
    from trainer_v2.hf_backend import HFBackend
    RV.require_single_gpu()
    P.refuse_holdout_path(args.out)
    out = Path(args.out)
    if args.teacher_table:
        P.refuse_holdout_path(args.teacher_table)
    if args.prereg:
        P.refuse_holdout_path(args.prereg)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"existing output requires --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    ws = init_lora_meta(args, out)
    if args.prereg:
        from trainer_v2 import pins
        pins.verify(Path(args.prereg), rows=D.release_file(Path(args.release), "train", args.mode))
        pins.verify_ws(Path(args.prereg), getattr(args, "init_lora", None), getattr(args, "init_lora_sha256", None))
    ag_check(args)
    if ws and not args.resume:
        (out / INIT_LORA_FILE).write_text(json.dumps(ws, indent=2) + "\n")
    RV._seed(args.seed)
    vef = vef_context(args, out)
    kmix = tuple(int(k) for k in args.context_k_mix.split(",")) if args.context_k_mix else ()
    items, manifest = D.load_release(Path(args.release), "train", args.mode, arms=(args.arm,), context_k_mix=kmix,
                                     mix_seed=args.seed, vef=vef is not None or needs_match_src(args))
    if not items:
        raise ValueError("no accepted training items")
    random.Random(args.seed).shuffle(items)
    opsd = _opsd_context(args)
    model, proc = RV.load_model(lora=True, size=args.model, fast_conv=args.fast_conv, lora_r=args.lora_r)
    if ws and not args.resume:                         # warm start: step 0 from the init LoRA (resume: checkpoint)
        load_init_lora(model, ws["path"], ws["adapter_sha256"])
    backend = HFBackend(model, proc, temperature=1.0)
    params = [p for p in model.parameters() if p.requires_grad]
    cfg = G.LoopConfig(num_generations=args.num_generations, r=args.r, reward_preset=args.reward_preset,
                       opsd_lambda=args.opsd_lambda if opsd else 0.0, **RV.algo_kwargs(args),
                       **RV.rollout_kwargs(args), **RV.train_kwargs(args), **RV.CL.dapo_kwargs(args),
                       **RV.CL.opsd_mode_kwargs(args))
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    monitor = MV.OverAbstainMonitor(window=args.monitor_window, min_precision=args.min_withhold_precision,
                                    hard_stop=args.hard_stop and args.stop_rule == "legacy")
    stopper = SR.StopRule(args.stop_rule, args.hard_stop, args.stop_warmup, args.monitor_window)
    start = restore_checkpoint(latest_checkpoint(out), model, optimizer, args, monitor) if args.resume else 0
    cursor = data_cursor(latest_checkpoint(out), start, args.prompts_per_step) if args.resume else 0
    sampler = None
    if args.target_absent_share is not None:
        sampler = AS.AbsentShareSampler(items, args.target_absent_share, args.seed)
        sampler.skip(cursor)                               # resume: replay the draws of steps < start
    if args.resume:
        for s, payload in SR.dev_history(out, start, WR.read_records, lambda r, k: WR.safe_payload(WR.dev_val_payload, r, k)):
            stopper.add_dev(s, payload)
    wb = WR.init(args, WR.run_config(args, cfg, manifest, start), out)
    acc, best = WL.Accumulator(), WL.BestTracker()
    if wb and args.resume:
        WR.safe_payload(WR.replay, out, start, acc, best)
    hints = _hints_by_row(items, opsd) if wb else None
    if args.resume and args.dev_every and start % args.dev_every == 0 and not (
            out / "dev_eval" / f"step_{start:06d}" / "eval_report.json").exists():
        payload = _dev_eval(args, start, backend, cfg, out)
        _log_val(wb, best, payload)
        stopper.add_dev(start, payload)
    last = start
    for step in range(start, args.steps):
        t0 = time.time()
        rec = train_one_step(items, step, args, backend, optimizer, params, cfg, opsd, monitor, out, sampler, cursor,
                             vef=vef)
        cursor += rec.get("n_prompts_drawn", args.prompts_per_step)
        last = step + 1
        if wb:
            WR.safe_log(wb, {**WR.safe_payload(
                WL.train_payload, rec, step=last, g=cfg.num_generations, acc=acc,
                lr=optimizer.param_groups[0]["lr"], opsd_lambda=cfg.opsd_lambda, hints_by_row=hints,
                step_s=time.time() - t0), **rec.get("data", {}), **vef_wandb(rec)})
        saved = (args.checkpoint_every and last % args.checkpoint_every == 0) or last == args.steps or monitor.stopped
        if saved:
            save_checkpoint(out, last, model, optimizer, args, monitor, cursor)
        if vef_down(rec):                       # fail closed: checkpoint, then stop without the DONE marker
            if not saved:
                save_checkpoint(out, last, model, optimizer, args, monitor, cursor)
            raise SystemExit(f"{VEF_DOWN} at step {last}: {rec['vef']} (checkpoint {latest_checkpoint(out)})")
        if args.dev_every and last % args.dev_every == 0:
            payload = _dev_eval(args, last, backend, cfg, out)
            _log_val(wb, best, payload)
            stopper.add_dev(last, payload)
        stop_payload = stopper.update(last, rec.get("monitor"), monitor.stopped)
        if wb:
            WR.safe_log(wb, stop_payload)
        if stopper.stopped:
            if not saved:  # checkpoint first, then report
                save_checkpoint(out, last, model, optimizer, args, monitor, cursor)
            logger.warning("over-abstain hard stop (%s) at step %d", args.stop_rule, last)
            break
    summary = {"step": last, "requested_steps": args.steps, "hard_stop": stopper.stopped,
               **stopper.summary(), "absent_sampler": sampler.summary() if sampler else None, "data": manifest, "config": cfg.__dict__,
               "checkpoint": str(latest_checkpoint(out)), **({"init_lora": ws} if ws else {})}
    (out / "main_report.json").write_text(json.dumps(summary, indent=2))
    if wb:
        WR.safe_summary(wb, {**summary, **best.summary()})
        try:
            wb.finish()
        except Exception as exc:  # noqa: BLE001 - logging is optional
            logger.warning("wandb finish failed: %s", exc)
    logger.info("%s arm=%s steps=%d out=%s%s", MARKER, args.arm, last, out,
                f" stopped={SR.REASON}" if stopper.stopped else "")
    return 0


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--mode", default="retrieved", choices=D.MODES)
    ap.add_argument("--arm", default="memory_call", choices=VD.ARMS)
    ap.add_argument("--model", default="4b", choices=sorted(P.MODELS))
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--prompts-per-step", type=int, default=1)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--r", type=float, default=.1)
    ap.add_argument("--reward-preset", default="main", choices=sorted(R5.PRESETS))
    ap.add_argument("--vef-judge-url", default=None, help="VEF presets (required): http://127.0.0.1:<port>/v1")
    ap.add_argument("--vef-judge-model", default=None, help="VEF presets: served model name (default env "
                    "VEF_JUDGE_MODEL or vef-judge)")
    ap.add_argument("--opsd", action="store_true")
    ap.add_argument("--teacher-table")
    ap.add_argument("--hint-set", default="main", choices=sorted(OP.HINT_SETS))
    ap.add_argument("--opsd-lambda", type=float, default=OP.LAMBDA)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--dev-every", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=24)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--kret-dev", action="store_true", default=os.environ.get("KRET_DEV") == "1",
                    help="also run the K-retention dev check at each dev eval (env KRET_DEV=1)")
    ap.add_argument("--kret-dev-variants", default=os.environ.get("KRET_DEV_VARIANTS", "image"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--context-k-mix", default=os.environ.get("CONTEXT_K_MIX", ""),
                    help="training only: comma list of context sizes drawn per row (e.g. 1,2,4,8); dev stays K=8")
    ap.add_argument("--seed", type=int, default=0)
    RV.add_algo_args(ap)                       # --algo / --updates-per-batch / --clip-eps-low / --clip-eps-high
    RV.add_rollout_args(ap)                    # --batch-rollout / --gen-batch (default: sequential rollout)
    RV.add_train_args(ap)                      # --train-micro-batch (default 1: one episode per forward)
    RV.CL.add_dapo_args(ap)                    # --algo dapo: dynamic sampling cap, overlong shaping
    RV.CL.add_opsd_mode_args(ap)               # --opsd-teacher frozen|self, --opsd-gate table|hint
    ap.add_argument("--prereg")
    ap.add_argument("--init-lora", default=argparse.SUPPRESS,
                    help="warm start: a COMPLETE checkpoint's lora dir as the step-0 weights of a FRESH run (fresh "
                         "optimizer / RNG from --seed); not with --resume (unset: the key is absent from the args)")
    ap.add_argument("--fast-conv", action="store_true", default=os.environ.get("FAST_CONV") == "1",
                    help="fla triton causal conv (trainer_v2/fast_conv.py; env FAST_CONV=1; resume-exempt)")
    ap.add_argument("--monitor-window", type=int, default=5)
    ap.add_argument("--min-withhold-precision", type=float, default=.7)
    ap.add_argument("--hard-stop", action="store_true")
    ap.add_argument("--target-absent-share", type=float, default=None,
                    help="seeded stratified sampler: expected share of gt_absent prompts (default: uniform order)")
    ap.add_argument("--stop-rule", default="legacy", choices=SR.RULES,
                    help="legacy: stop when the monitor trips; confirmed: warm-up + train window + 2 dev rises")
    ap.add_argument("--stop-warmup", type=int, default=50, help="confirmed rule: no stop before this global step")
    ap.add_argument("--wandb-project", default="EAPMA")
    ap.add_argument("--wandb-group")
    ap.add_argument("--wandb-name")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    args.cmd = "main"
    if args.steps < 1 or args.checkpoint_every < 0 or args.dev_every < 0 or args.dev_limit < 1:
        ap.error("steps and dev-limit must be positive; intervals must be nonnegative")
    if args.opsd and not args.teacher_table and RV.CL.resolved_gate(args) == "table":
        ap.error("--opsd with the table gate requires --teacher-table")
    if getattr(args, "init_lora", None) is not None and (args.resume or not args.init_lora):
        ap.error("--init-lora starts a fresh run: not with --resume (resume a warm-start run without it)")
    if args.updates_per_batch < 1:
        ap.error("--updates-per-batch must be >= 1")
    if args.gen_batch < 1 or args.train_micro_batch < 1:
        ap.error("--gen-batch and --train-micro-batch must be >= 1")
    if args.target_absent_share is not None and not 0 <= args.target_absent_share <= 1:
        ap.error("--target-absent-share must be in [0, 1]")
    if bool(R5.PRESETS[args.reward_preset].vef) != bool(args.vef_judge_url):
        ap.error("VEF reward presets require --vef-judge-url, and --vef-judge-url requires a VEF preset")
    if args.stop_warmup < 0 or args.monitor_window < 1:
        ap.error("stop-warmup must be nonnegative and monitor-window positive")
    if args.hard_stop and args.stop_rule == "confirmed" and not args.dev_every:
        logger.warning("--stop-rule confirmed without --dev-every: the run can never stop")
    D.release_file(Path(args.release), "train", args.mode)
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    return gpu_main(parse(argv))


if __name__ == "__main__":
    sys.exit(main())
