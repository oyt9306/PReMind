"""``evaluate.py --model api:<provider>:<model_id>``: closed-source API evaluation (EVAL ONLY, no GPU).

Three modes:
  * ``--dry-run``      NO network: renders every request of the split / modes / arms / limit through the real
                       rollout (Stage 1 assumed to call memory, Stage 1 output = max tokens: worst case), counts
                       images and approximate input tokens and prints the estimated USD cost (api_providers.PRICES).
  * ``--prereg-line``  NO network: prints the one-line registration the orchestrator must log before a paid run;
                       its trailing sha12 is the ``--prereg-ack`` value the paid run requires.
  * local run          ``api:local:<served>`` (eval_v1/local_provider.py): free OpenAI-compatible server (vLLM) at
                       ``--api-base``; no key / allow-provider / budget / prereg gates; ledger cost 0; same cache.
  * paid run           needs ``--allow-provider <provider>`` (default: none allowed), ``--budget-usd``, the matching
                       ``--prereg-ack`` and the provider key in the environment. Responses are cached write-once in
                       ``<out>.api/calls.jsonl`` (resume: a re-run with the same args re-pays nothing done) and every
                       paid call is appended to the ledger (default ``<out>.api/ledger.jsonl``).
Outputs (records / texts / report) are written by evaluate.write exactly as for the local models (write once).

OpenAI reasoning models: ``api:openai:gpt-6.1-sol`` / ``api:openai:gpt-6-luna[@high]`` (effort suffix = profile param:
enters the cache key, the prereg line and the run metadata, so medium / high caches never mix).
Sealed test split: only with the eval.sh opt-in, FINAL_HOLDOUT=1 in the environment AND --split holdout
(holdout_opt_in); --final-holdout alone is refused. Without it every holdout path is refused (default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from PIL import Image

from eval_v1 import api_backend as AB
from eval_v1 import api_providers as AP
from eval_v1 import local_provider as LP
from eval_v1 import release_data as D
from eval_v1 import topk_context as TK
from trainer_v2 import grpo_v2 as G
from trainer_v2 import paths_v2 as P

logger = logging.getLogger(__name__)


def parse_model(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3 or parts[0] != "api" or parts[1] not in AP.PROVIDERS or not parts[2]:
        raise SystemExit(f"REFUSED: --model must be api:<{'|'.join(AP.PROVIDERS)}>:<model_id>, got {spec!r}")
    provider, model = parts[1], parts[2]
    if provider == "local":
        return provider, f"local:{model}"
    try:
        prof = AP.profile(model)
    except AP.ApiUnavailableError as e:
        raise SystemExit(f"REFUSED: {e}") from e
    if prof.provider != provider:
        raise SystemExit(f"REFUSED: {model!r} is a {prof.provider} model, not {provider}")
    return provider, model


def check_args(args: argparse.Namespace) -> None:
    if args.lora or args.fast_conv:
        raise SystemExit("REFUSED: --lora / --fast-conv are not supported for API models")
    if args.samples:
        raise SystemExit("REFUSED: API models are evaluated greedy-equivalent only (--samples 0)")


def holdout_opt_in(args: argparse.Namespace) -> None:
    """Sealed TEST split = the eval.sh opt-in: FINAL_HOLDOUT=1 (environment) AND --split holdout. Sets
    args.final_holdout; any partial opt-in is refused; nothing else lifts the holdout refusal."""
    env = os.environ.get("FINAL_HOLDOUT") == "1"
    if not (env or args.final_holdout):
        return
    if not (env and args.split == "holdout"):
        raise SystemExit("REFUSED: the sealed test split needs FINAL_HOLDOUT=1 in the environment AND --split holdout "
                         f"(got FINAL_HOLDOUT={os.environ.get('FINAL_HOLDOUT')!r}, --split {args.split})")
    args.final_holdout = True


def load_items(args: argparse.Namespace) -> tuple[dict, dict]:
    from eval_v1 import evaluate as EV
    items_by_mode, manifests = {}, {}
    for mode in args.modes.split(","):
        items_by_mode[mode], manifests[mode] = D.load_release(Path(args.release), args.split, mode,
                                                               arms=tuple(args.arms.split(",")), limit=args.limit,
                                                               final_holdout=args.final_holdout,
                                                               context_k=getattr(args, "context_k", TK.FULL_K),
                                                               **EV.online_kwargs(args))
    if not any(items_by_mode.values()):
        raise ValueError("no accepted evaluation items")
    return items_by_mode, manifests


class DryCaller:
    """No network: worst-case accounting of every rendered request."""

    def __init__(self, model: str, stage1_cap: int) -> None:
        self.model, self.stage1_cap = model, stage1_cap
        self.calls = self.images = self.in_tok = self.out_tok = self.reasoning_tok = 0
        self.max_images = self.max_request_bytes = self.max_image_bytes = 0

    def __call__(self, req: AP.ApiRequest, sizes: dict) -> AP.ApiResponse:
        n_asst = sum(role == "assistant" for role, _ in req.turns)
        imgs = req.images()
        self.calls += 1
        self.images += len(imgs)
        self.max_images = max(self.max_images, len(imgs))
        sz = [Path(p).stat().st_size for p in imgs]
        self.max_request_bytes = max(self.max_request_bytes, sum(sz))
        self.max_image_bytes = max([self.max_image_bytes, *sz])
        self.in_tok += AB.estimate_input_tokens(self.model, req, sizes) + n_asst * self.stage1_cap
        self.out_tok += req.max_tokens
        self.reasoning_tok += AP.assumed_reasoning_tokens(self.model)
        return AP.ApiResponse("<memory_call>call_pdb</memory_call>", 0, req.max_tokens, True, "dry_run")

    def summary(self) -> dict:
        return {"calls": self.calls, "images": self.images, "max_images_per_request": self.max_images,
                "max_image_bytes_per_request": self.max_request_bytes, "max_single_image_bytes": self.max_image_bytes, "est_input_tokens": self.in_tok,
                "max_output_tokens": self.out_tok, "assumed_reasoning_tokens": self.reasoning_tok,
                "est_cost_usd_upper": round(AP.cost_usd(self.model, self.in_tok, self.out_tok), 2)}


def estimate(args: argparse.Namespace, model: str, items_by_mode: dict) -> dict:
    cfg = G.LoopConfig(r=args.r, reward_preset=args.reward_preset)
    per: dict = {}
    for mode, items in items_by_mode.items():
        for arm in sorted({it.arm for it in items}):
            dry = DryCaller(model, cfg.stage1_cap)
            be = AB.ApiBackend(model, dry)
            for it in (x for x in items if x.arm == arm):
                sizes = {}
                for p in dict.fromkeys(it.image_paths):
                    with Image.open(p) as im:
                        sizes[p] = im.size
                G.rollout([it], [sizes], be, cfg, greedy=True, num_generations=1)
            per[f"{mode}|{arm}"] = {"items": sum(x.arm == arm for x in items), **dry.summary()}
    total = {k: sum(v[k] for v in per.values()) for k in ("items", "calls", "images", "est_input_tokens",
                                                            "max_output_tokens", "assumed_reasoning_tokens")}
    total["est_cost_usd_upper"] = round(AP.cost_usd(model, total["est_input_tokens"], total["max_output_tokens"]), 2)
    prof = AP.profile(model)
    out = {"model": model, "provider": prof.provider, "prices_usd_per_mtok": [prof.usd_in_per_mtok,
                                                                              prof.usd_out_per_mtok],
           "image_rule": prof.image_rule, "params": dict(prof.params), "per_setting": per, "total": total,
           "note": "upper bound: every memory_call Stage 1 calls memory and every output hits max tokens; "
                   "verify current prices before launch"}
    if AP.reasoning_allowance(model):
        n_assumed = AP.assumed_reasoning_tokens(model)
        total["est_cost_usd_with_assumed_reasoning"] = round(AP.cost_usd(
            model, total["est_input_tokens"], total["max_output_tokens"] + total["assumed_reasoning_tokens"]), 4)
        total["est_cost_usd_worst_reasoning_allowance"] = round(AP.cost_usd(
            model, total["est_input_tokens"], total["max_output_tokens"] + total["calls"] * AP.reasoning_allowance(model)), 4)
        out["reasoning_note"] = (f"ASSUMPTION (not measured): {n_assumed} reasoning tokens per call at effort "
                                 f"{AP.reasoning_effort(model)!r}; image tokens ~{AP.OPENAI_IMAGE_TOKENS_1536x1024} per "
                                 "1536x1024 UNVERIFIED; ledger uses the API's reported usage")
    return out


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    api_dir = Path(args.out + ".api")
    ledger = Path(args.api_ledger) if args.api_ledger else api_dir / "ledger.jsonl"
    if not args.final_holdout:
        P.refuse_holdout_path(api_dir)
        P.refuse_holdout_path(ledger)
    return api_dir / "calls.jsonl", ledger


def prereg_line(args: argparse.Namespace, est: dict) -> str:
    cfg = G.LoopConfig(r=args.r, reward_preset=args.reward_preset)
    t = est["total"]
    _, ledger = paths(args)
    body = (f"API-PREREG v1 | model=api:{est['provider']}:{est['model']} params={json.dumps(est['params'], sort_keys=True)}"
            f" | release={args.release} split={args.split} modes={args.modes} arms={args.arms} limit={args.limit}"
            f" final_holdout={args.final_holdout} items={t['items']} | calls<={t['calls']} images={t['images']}"
            f" est_usd<={t['est_cost_usd_upper']} budget_usd={args.budget_usd} | max_tokens stage1={cfg.stage1_cap}"
            f" stage2={cfg.max_completion_length} reward={args.reward_preset} r={args.r}"
            f" | out={args.out} ledger={ledger}")
    if AP.reasoning_allowance(est["model"]):
        body += (f" | reasoning_effort={AP.reasoning_effort(est['model'])}"
                 f" max_output_tokens=cap+{AP.reasoning_allowance(est['model'])} visible_cap_counter=qwen3.5"
                 f" est_usd_assumed_reasoning={t['est_cost_usd_with_assumed_reasoning']}"
                 f" (assumed {AP.assumed_reasoning_tokens(est['model'])} reasoning tok/call)")
    return f"{body} | sha={hashlib.sha256(body.encode()).hexdigest()[:12]}"


def provider_call(provider: str, model: str) -> object:  # network (paid run only)
    if provider == "openai":   # at most 1 retry on transient 5xx / timeout / connection / 429
        AP.require_key("openai")
        return lambda req: AP.with_retries(lambda: AP.openai_call(model, req), attempts=2, base=10.0)
    if provider == "anthropic":
        AP.require_key("anthropic")
        return lambda req: AP.with_retries(lambda: AP.anthropic_call(model, req))
    AP.require_key("gemini")
    return lambda req: AP.with_retries(lambda: AP.gemini_call(model, req))


def main_api(args: argparse.Namespace, *, call_factory=provider_call) -> int:
    from eval_v1 import evaluate as EV
    provider, model = parse_model(args.model)
    check_args(args)
    holdout_opt_in(args)
    if provider == "local":
        LP.local_model(args.local_model_key)
        LP.check_api_base(args.api_base)
    elif args.api_base or args.local_model_key:
        raise SystemExit("REFUSED: --api-base / --local-model-key are only for api:local:<served-name>")
    items_by_mode, manifests = load_items(args)
    if args.dry_run or args.prereg_line:
        est = estimate(args, model, items_by_mode)
        print(prereg_line(args, est) if args.prereg_line else json.dumps(est, indent=2))
        return 0
    if provider == "local":
        return main_local(args, model, items_by_mode, manifests)
    if provider not in (args.allow_provider or []):
        raise SystemExit(f"REFUSED: provider {provider!r} not allowed (pass --allow-provider {provider}; "
                         "default denied; owner 2026-10-05: no Gemini spending)")
    if args.budget_usd is None:
        raise SystemExit("REFUSED: --budget-usd is required for a paid API run")
    expected = prereg_line(args, estimate(args, model, items_by_mode)).rsplit("sha=", 1)[1]
    if args.prereg_ack != expected:
        raise SystemExit(f"REFUSED: --prereg-ack {args.prereg_ack!r} != sha of the prereg line ({expected}); "
                         "log the --prereg-line output first")
    out = Path(args.out)
    if not args.final_holdout:
        P.refuse_holdout_path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty (write once): {out}")
    cache, ledger = paths(args)
    store = AB.CallStore(model, cache, ledger, args.budget_usd, call_factory(provider, model))
    backend = AB.ApiBackend(model, store)
    cfg = G.LoopConfig(r=args.r, reward_preset=args.reward_preset)
    t0 = time.time()
    recs, texts = EV.generate(items_by_mode, backend, cfg, samples=0, temperature=args.temperature,
                              strata_map=EV.strata(Path(args.rule_strata) if args.rule_strata else None))
    prof = AP.profile(model)
    meta = {"release": str(args.release), "split": args.split, "modes": tuple(args.modes.split(",")),
            "arms": tuple(args.arms.split(",")), "model": args.model, "model_pin": (provider, model), "lora": None,
            "reward_preset": args.reward_preset, "r": args.r, "samples": 0, "temperature": None, "data": manifests,
            "runtime_s": round(time.time() - t0, 1),
            "api": {"provider": provider, "model": model, "params": dict(prof.params), "note": prof.note,
                    "prices_usd_per_mtok": [prof.usd_in_per_mtok, prof.usd_out_per_mtok],
                    "max_tokens": [cfg.stage1_cap, cfg.max_completion_length], "prereg_sha": expected,
                    "calls_paid": store.n_paid, "calls_cached": store.n_cached,
                    "ledger": str(ledger), "ledger_total_usd": round(store.spent, 4), "budget_usd": args.budget_usd,
                    "backend_stats": backend.stats, "n_stage2_tokens": "provider visible output tokens"}}
    if AP.reasoning_allowance(model):
        meta["api"].update({"reasoning_effort": AP.reasoning_effort(model),
                            "reasoning_allowance": AP.reasoning_allowance(model),
                            "max_output_tokens": [cfg.stage1_cap + AP.reasoning_allowance(model),
                                                  cfg.max_completion_length + AP.reasoning_allowance(model)],
                            "reasoning_tokens_paid_calls": store.reasoning_tokens,
                            "n_stage2_tokens": f"visible text truncated to the cap, counted with {AP.VISIBLE_COUNTER}"})
    meta.update(EV.online_meta(args))
    EV.write(out, recs, texts, meta, draws=args.bootstrap, seed=args.seed, final_holdout=args.final_holdout)
    logger.info("%s items=%d out=%s", EV.MARKER, len(recs), out)
    return 0


def main_local(args: argparse.Namespace, model: str, items_by_mode: dict, manifests: dict) -> int:
    """api:local run: free, so no allow-provider / budget / prereg gates; cache + zero-cost ledger kept (resume)."""
    from eval_v1 import evaluate as EV
    lm = LP.local_model(args.local_model_key)
    base, served = LP.check_api_base(args.api_base), model.split(":", 1)[1]
    out = Path(args.out)
    if not args.final_holdout:
        P.refuse_holdout_path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty (write once): {out}")
    info = LP.server_info(base, served)          # fails closed when the served name is not listed
    cache, ledger = paths(args)
    store = AB.CallStore(model, cache, ledger, float("inf"), LP.make_call(base, served, lm.image_marker))
    backend = AB.ApiBackend(model, store)
    cfg = G.LoopConfig(r=args.r, reward_preset=args.reward_preset)
    t0 = time.time()
    recs, texts = EV.generate(items_by_mode, backend, cfg, samples=0, temperature=args.temperature,
                              strata_map=EV.strata(Path(args.rule_strata) if args.rule_strata else None),
                              context_k=EV.k_arg(args))
    meta = {"release": str(args.release), "split": args.split, "modes": tuple(args.modes.split(",")),
            "arms": tuple(args.arms.split(",")), "model": args.model, "model_key": args.local_model_key,
            "model_pin": (lm.hf_id, lm.revision), "params_b": lm.params_b, "lora": lm.served_as if served ==
            lm.served_as else None, "reward_preset": args.reward_preset, "r": args.r, "samples": 0,
            "temperature": None, "data": manifests, "runtime_s": round(time.time() - t0, 1), "context_k": EV.k_arg(args),
            "api": {"provider": "local", "model": model, "params": {"temperature": 0.0}, "note": lm.note,
                    "image_marker": lm.image_marker, "server": info, "cost_usd": 0.0,
                    "max_tokens": [cfg.stage1_cap, cfg.max_completion_length], "prereg_sha": None,
                    "calls_made": store.n_paid, "calls_cached": store.n_cached, "ledger": str(ledger),
                    "backend_stats": backend.stats, "n_stage2_tokens": "server completion_tokens"}}
    meta.update(EV.online_meta(args))
    EV.write(out, recs, texts, meta, draws=args.bootstrap, seed=args.seed, final_holdout=args.final_holdout)
    logger.info("%s items=%d out=%s", EV.MARKER, len(recs), out)
    return 0
