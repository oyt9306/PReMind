#!/usr/bin/env python3
"""VCM E0/E1 evaluator. GPU generation is invoked only by eval.sh under the scheduler."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from eval_v1 import hazard_match as HM  # noqa: E402
from eval_v1 import analysis as A  # noqa: E402
from eval_v1 import baselines as B  # noqa: E402
from eval_v1 import release_data as D  # noqa: E402
from eval_v1 import timed_backend as TB  # noqa: E402
from eval_v1 import topk_context as TK  # noqa: E402
from trainer_v2 import grpo_v2 as G  # noqa: E402
from trainer_v2 import metrics_v9 as MV  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402
from trainer_v2 import vcm_data as VD  # noqa: E402

logger = logging.getLogger(__name__)
MARKER = "EVAL V1 DONE"


def read_jsonl(path: Path, *, final_holdout: bool = False) -> list[dict]:
    if not final_holdout:
        P.refuse_holdout_path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    P.refuse_holdout_path(path)
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError("rule strata must be a JSON object mapping rule_id to known/gap")
    mapping = obj.get("strata", obj)
    if not isinstance(mapping, dict):
        raise ValueError("rule strata mapping must be an object")
    aliases = {"known": "known", "main": "known", "gap": "gap", "knowledge_gap": "gap",
               "unprobed": "unprobed"}
    if any(v not in aliases for v in mapping.values()):
        raise ValueError("rule strata values must be known, gap, or unprobed")
    return {str(k): aliases[v] for k, v in mapping.items()}


def tag_record(item: VD.VcmItem, ledger: dict, *, mode: str, kind: str, sample_idx: int,
               decision: str | None, strata_map: dict[str, str], stage2: str) -> dict:
    r = MV.episode_record(item, ledger, mode=mode, kind=kind, sample_idx=sample_idx, decision=decision)
    target = item.target
    rule = item.row.get("rule_id") or item.row.get("general_knowledge_rule")
    pattern = target.get("hazard_pattern")
    # owner 2026-10-04: hazard_first = the answer's first two sentences address the hazard (eval_v1/hazard_match.py);
    # the old flag (pattern anywhere in Stage 2, strict regex) is kept as hazard_mention_strict
    hazard = HM.hazard_first(pattern, HM.answer_text(stage2))
    mention = bool(re.search(pattern, stage2, re.I)) if pattern else None
    return {**r, "family_id": item.row["family_id"], "grounding": item.row.get("grounding_requirement"),
            "risk": item.row.get("risk"), "query_style": item.row.get("query_style"), "request_form": item.row.get("request_form"),
            "knowledge_stratum": strata_map.get(str(rule), "unmapped" if rule else "none"),
            "rule_id": rule, "acute_visual": target["acute_visual"], "hazard_first": hazard, "hazard_mention_strict": mention,
            "action_pred": (ledger.get("parse") or {}).get("action_pred")}


def generate(items_by_mode: dict[str, list[VD.VcmItem]], backend: object, cfg: G.LoopConfig,
             *, samples: int, temperature: float, strata_map: dict[str, str],
             context_k: int = TK.FULL_K, forced_stage1=None) -> tuple[list[dict], list[dict]]:
    """Records gain top-K fields (topk_context.record_fields) and per-episode cost fields (timed_backend).
    ``forced_stage1`` (row_id -> Stage-1 text, --stage1-from): greedy memory-call episodes reuse that Stage-1 text."""
    fs = (lambda it: forced_stage1[it.row_id]) if forced_stage1 else None
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    if getattr(backend, "llm", None) is not None:   # vLLM: batched rollout over chunks of items
        return _generate_batched(items_by_mode, backend, cfg, samples=samples, temperature=temperature,
                                 strata_map=strata_map, context_k=context_k, forced_stage1=fs)
    backend = backend if isinstance(backend, TB.TimedBackend) else TB.TimedBackend(backend)
    recs, texts = [], []
    for mode, items in items_by_mode.items():
        for item in items:
            images = VD.item_images(item)
            backend.take()
            runs = [("greedy", ep) for ep in G.rollout([item], [images], backend, cfg, greedy=True,
                                                        num_generations=1, forced_stage1=fs)]
            costs = [TB.episode_cost(backend.take(), item.arm, 0, forced_stage1=fs is not None)]
            if samples:
                runs += [("sample", ep) for ep in G.rollout([item], [images], backend, cfg, greedy=False,
                                                              num_generations=samples, temperature=temperature)]
                calls = backend.take()
                costs += [TB.episode_cost(calls, item.arm, i) for i in range(samples)]
            topk = TK.record_fields(item.row, item.target, context_k)
            counts: dict[str, int] = {}
            for (kind, ep), cost in zip(runs, costs, strict=True):
                idx = counts.get(kind, 0)
                counts[kind] = idx + 1
                led = G.score(ep, cfg)
                recs.append({**tag_record(item, led, mode=mode, kind=kind, sample_idx=idx, decision=ep.decision,
                                          strata_map=strata_map, stage2=ep.stage2_text), **topk, **cost})
                texts.append({"row_id": item.row_id, "arm": item.arm, "mode": mode, "kind": kind,
                              "sample_idx": idx, "stage1": ep.stage1_text, "stage2": ep.stage2_text,
                              "n_stage2_tokens": ep.n_stage2})
        logger.info("generated %d items for mode %s", len(items), mode)
    return recs, texts


VLLM_CHUNK = 256


def _generate_batched(items_by_mode: dict[str, list[VD.VcmItem]], backend: object, cfg: G.LoopConfig, *,
                      samples: int, temperature: float, strata_map: dict[str, str],
                      context_k: int, forced_stage1=None) -> tuple[list[dict], list[dict]]:
    """``generate`` for the vLLM backend: same records / texts, episodes from ``rollout(batched=True)`` over chunks of
    VLLM_CHUNK items. Latency is the chunk wall time amortised per item (``seconds``; stage splits not measured)."""
    recs, texts = [], []
    for mode, items in items_by_mode.items():
        for lo in range(0, len(items), VLLM_CHUNK):
            chunk = items[lo:lo + VLLM_CHUNK]
            imgs = [VD.item_images(it) for it in chunk]
            t0 = time.perf_counter()
            eps = [("greedy", ep) for ep in G.rollout(chunk, imgs, backend, cfg, greedy=True, num_generations=1,
                                                       batched=True, forced_stage1=forced_stage1)]
            if samples:
                eps += [("sample", ep) for ep in G.rollout(chunk, imgs, backend, cfg, greedy=False,
                                                            num_generations=samples, temperature=temperature,
                                                            batched=True)]
            per_item = (time.perf_counter() - t0) / len(chunk)
            by_item: dict[int, list] = {}   # keyed by item object (a row has one item per arm)
            for kind, ep in eps:
                by_item.setdefault(id(ep.item), []).append((kind, ep))
            for item in chunk:
                topk = TK.record_fields(item.row, item.target, context_k)
                counts: dict[str, int] = {}
                for kind, ep in by_item[id(item)]:
                    idx = counts.get(kind, 0)
                    counts[kind] = idx + 1
                    led = G.score(ep, cfg)
                    cost = {k: None for k in TB.COST_KEYS}
                    cost.update(stage1_gen_tokens=ep.n_stage1 if item.arm != "passive" else None,
                                stage2_gen_tokens=ep.n_stage2, input_tokens=len(ep.prompt.ids),
                                n_images=len(ep.images), seconds=round(per_item, 4), shared_call=True)
                    recs.append({**tag_record(item, led, mode=mode, kind=kind, sample_idx=idx, decision=ep.decision,
                                              strata_map=strata_map, stage2=ep.stage2_text), **topk, **cost})
                    texts.append({"row_id": item.row_id, "arm": item.arm, "mode": mode, "kind": kind,
                                  "sample_idx": idx, "stage1": ep.stage1_text, "stage2": ep.stage2_text,
                                  "n_stage2_tokens": ep.n_stage2})
            logger.info("vllm: %s %d/%d items", mode, min(lo + VLLM_CHUNK, len(items)), len(items))
    return recs, texts


def report(recs: list[dict], meta: dict, *, draws: int, seed: int) -> dict:
    out = {"meta": meta, "metrics_v9": MV.report(recs), "table": A.table(recs, draws=draws, seed=seed)}
    cases = sorted({r["absent_case"] for r in recs if r.get("absent_case")})
    if cases:                      # v4 online path only: results by target-absent case (never saved / saved but missed)
        out["by_absent_case"] = {c: {"n_records": len(sub), "summary": A.summary(sub), "metrics_v9": MV.report(sub)}
                                 for c in cases for sub in [[r for r in recs if r.get("absent_case") == c]]}
    latency = TB.latency_summary(recs)
    if latency:
        out["latency"] = latency
    return out


def write(out: Path, recs: list[dict], texts: list[dict], meta: dict, *, draws: int, seed: int,
          final_holdout: bool = False) -> dict:
    if not final_holdout:
        P.refuse_holdout_path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty (write once): {out}")
    out.mkdir(parents=True, exist_ok=True)
    rep = report(recs, meta, draws=draws, seed=seed)
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    (out / "eval_report.json").write_text(json.dumps(rep, indent=2))
    (out / "EVAL_REPORT.md").write_text(A.markdown(rep))
    return rep


def compare(a: Path, b: Path, out: Path, *, metric: str, draws: int, seed: int,
            final_holdout: bool = False) -> dict:
    if not final_holdout:
        P.refuse_holdout_path(out)
    if out.exists():
        raise FileExistsError(out)
    ra, rb = read_jsonl(a / "records.jsonl", final_holdout=final_holdout), read_jsonl(
        b / "records.jsonl", final_holdout=final_holdout)
    overall = A.paired(ra, rb, metric=metric, draws=draws, seed=seed)
    settings = sorted({f"{r['arm']}|{r['mode']}|{r['kind']}" for r in ra})
    def setting(r: dict) -> str:
        return f"{r['arm']}|{r['mode']}|{r['kind']}"
    by_setting = {name: A.paired([r for r in ra if setting(r) == name],
                                  [r for r in rb if setting(r) == name],
                                  metric=metric, draws=draws, seed=seed) for name in settings}
    result = {"overall": overall, "by_setting": by_setting}
    out.mkdir(parents=True)
    (out / "paired_comparison.json").write_text(json.dumps(result, indent=2))
    (out / "PAIRED_COMPARISON.md").write_text(
        "# Paired VCM run comparison\n\n"
        "Delta is run B minus run A; CI uses paired family-cluster resampling.\n\n"
        f"| Metric | Delta | 95% CI | Matched items | Families |\n|---|---:|---:|---:|---:|\n"
        f"| {metric} (overall) | {overall['delta_b_minus_a']} | {overall['ci95']} | "
        f"{overall['n_items']} | {overall['n_families']} |\n" +
        "".join(f"| {metric} ({name}) | {cell['delta_b_minus_a']} | {cell['ci95']} | "
                f"{cell['n_items']} | {cell['n_families']} |\n" for name, cell in by_setting.items()))
    return result


def gpu_main(args: argparse.Namespace) -> int:  # pragma: no cover - scheduler GPU job
    import torch
    from trainer_v2 import run_v2 as RV
    from trainer_v2.hf_backend import HFBackend
    RV.require_single_gpu()
    mode_names = tuple(args.modes.split(","))
    arm_names = tuple(args.arms.split(","))
    items_by_mode, manifests = {}, {}
    for mode in mode_names:
        items_by_mode[mode], manifests[mode] = D.load_release(Path(args.release), args.split, mode,
                                                               arms=arm_names, limit=args.limit,
                                                               final_holdout=args.final_holdout,
                                                               context_k=topk_arg(args), **online_kwargs(args))
    if not any(items_by_mode.values()):
        raise ValueError("no accepted evaluation items")
    check_model_args(args)
    base_meta = None
    if args.model in B.BASELINES and args.backend == "vllm" and B.FAMILIES[B.get(args.model).family].model_type in (
            __import__("eval_v1.vllm_text_backend", fromlist=["VLLM_FAMILIES"]).VLLM_FAMILIES):
        from eval_v1 import vllm_text_backend as VT   # owner 2026-10-06: baselines on vLLM where the arch is served
        backend, base_meta = VT.load(args.model, temperature=temperature_or_one(args.temperature))
        model_pin = (base_meta["model_id"], base_meta["revision"])
    elif args.model in B.BASELINES:  # external baseline (eval only): no LoRA, no fast conv
        model, proc, base_meta = B.load_baseline(args.model)
        if base_meta.get("backend") == "generic":   # non-Qwen chat template (eval_v1/generic_backend.py)
            from eval_v1.generic_backend import from_meta
            backend = from_meta(model, proc, base_meta, temperature=temperature_or_one(args.temperature))
        else:
            from eval_v1.baseline_backend import BaselineBackend
            backend = BaselineBackend(model, proc, temperature=temperature_or_one(args.temperature))
        model_pin = (base_meta["model_id"], base_meta["revision"])
    else:
        if args.lora:
            P.refuse_holdout_path(args.lora)
        if args.backend == "vllm":   # owner 2026-10-06: Qwen3.5 evals on vLLM (eval_v1/vllm_backend.py)
            from eval_v1 import vllm_backend as VB
            backend = VB.load(args.model, lora=args.lora, temperature=temperature_or_one(args.temperature))
        else:
            model, proc = RV.load_model(lora=False, size=args.model, fast_conv=args.fast_conv)
            if args.lora:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.lora).eval()
            backend = HFBackend(model, proc, temperature=temperature_or_one(args.temperature))
        model_pin = P.MODELS[args.model]
    cfg = G.LoopConfig(r=args.r, reward_preset=args.reward_preset)
    t0 = time.time()
    recs, texts = generate(items_by_mode, backend, cfg, samples=args.samples, temperature=args.temperature,
                           strata_map=strata(Path(args.rule_strata) if args.rule_strata else None),
                           context_k=k_arg(args), forced_stage1=stage1_from(args, items_by_mode))
    meta = {"release": str(args.release), "split": args.split, "modes": mode_names, "arms": arm_names,
            "model": args.model, "model_pin": model_pin, "lora": args.lora,
            "reward_preset": args.reward_preset, "r": args.r, "samples": args.samples,
            "temperature": args.temperature, "data": manifests, "runtime_s": round(time.time() - t0, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "context_k": k_arg(args), "stage1_from": getattr(args, "stage1_from", None),
            "backend": (base_meta.get("backend_impl", "hf") if base_meta is not None else getattr(args, "backend", "hf"))}
    meta.update(online_meta(args))
    if base_meta is not None:
        meta["baseline"] = base_meta
        if base_meta.get("backend") == "generic":
            meta["baseline"] = {**base_meta, "backend_stats": backend.stats}
    write(Path(args.out), recs, texts, meta, draws=args.bootstrap, seed=args.seed,
          final_holdout=args.final_holdout)
    logger.info("%s items=%d out=%s", MARKER, len(recs), args.out)
    return 0


def stage1_from(args: argparse.Namespace, items_by_mode: dict) -> Optional[dict]:
    """--stage1-from: row_id -> Stage-1 text of a reference run (greedy, memory_call, retrieved; cross-check 2026-10-07:
    vLLM greedy Stage-1 decisions flip with batch composition, so top-K runs reuse the K=8 decision). Every memory-call
    row evaluated here must exist in the reference."""
    src = getattr(args, "stage1_from", None)
    if not src:
        return None
    ref = {t["row_id"]: t["stage1"] for t in map(json.loads, (Path(src) / "texts.jsonl").read_text().splitlines())
           if (t["arm"], t["mode"], t["kind"]) == ("memory_call", "retrieved", "greedy")}
    need = {it.row_id for items in items_by_mode.values() for it in items if it.arm == "memory_call"}
    missing = need - set(ref)
    if missing:
        raise SystemExit(f"REFUSED: --stage1-from {src} lacks {len(missing)} memory-call rows")
    return ref


def k_arg(args: argparse.Namespace) -> int:
    """--online-k if set, else --context-k (namespaces built by older callers / tests carry none: the unchanged K=8
    path)."""
    ok = getattr(args, "online_k", None)
    return ok if ok is not None else getattr(args, "context_k", TK.FULL_K)


def topk_arg(args: argparse.Namespace) -> int:
    """--context-k only (stored-rank truncation; the online path passes its K through online_kwargs)."""
    return getattr(args, "context_k", TK.FULL_K)


def online_kwargs(args: argparse.Namespace) -> dict:
    """load_release kwargs of the v4 online-retrieval path ({} when --online-k is unset: unchanged call)."""
    ok = getattr(args, "online_k", None)
    if ok is None:
        return {}
    qe = getattr(args, "online_query_emb", None)
    return {"online_k": ok, "online_query_emb": Path(qe) if qe else None}


def online_meta(args: argparse.Namespace) -> dict:
    ok = getattr(args, "online_k", None)
    return {} if ok is None else {"online_k": ok, "online_query_emb": getattr(args, "online_query_emb", None)}


def check_model_args(args: argparse.Namespace) -> None:
    """External baselines are eval-only: refuse --lora and --fast-conv (Gated DeltaNet kernel) for them.
    --context-k < 8 (eval_v1/topk_context.py): retrieved mode only; API path: free api:local models only (api_eval.py)."""
    paid_api = args.model.startswith("api:") and not args.model.startswith("api:local:")
    if getattr(args, "online_k", None) is not None and getattr(args, "context_k", TK.FULL_K) != TK.FULL_K:
        raise SystemExit("REFUSED: --online-k and --context-k < 8 are exclusive")
    if getattr(args, "online_k", None) is not None and set(args.modes.split(",")) != {"retrieved"}:
        raise SystemExit("REFUSED: --online-k needs --modes retrieved")
    if topk_arg(args) != TK.FULL_K and (paid_api or set(args.modes.split(",")) != {"retrieved"}):
        raise SystemExit("REFUSED: --context-k < 8 needs --modes retrieved and a local or api:local model")
    if (args.model in B.BASELINES or args.model.startswith("api:")) and (args.lora or args.fast_conv):
        raise SystemExit(f"REFUSED: --lora / --fast-conv are not supported for baseline {args.model!r}")


def model_arg(value: str) -> str:
    """Qwen3.5 size, external baseline key, or api:<provider>:<model_id> (eval_v1/api_eval.py)."""
    if value in P.MODELS or value in B.BASELINES or value.startswith("api:"):
        return value
    raise argparse.ArgumentTypeError(f"unknown model {value!r}; known {sorted(P.MODELS) + sorted(B.BASELINES)} "
                                     "or api:<provider>:<model_id>")


def temperature_or_one(value: float) -> float:
    if value <= 0:
        raise ValueError("temperature must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--modes", default="retrieved,gt_retrieved,gt_retrieved_drop")
    ap.add_argument("--arms", default="passive,memory_call")
    ap.add_argument("--model", default="4b", type=model_arg)
    ap.add_argument("--lora")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=.7)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--r", type=float, default=.1)
    ap.add_argument("--reward-preset", default="main", choices=sorted(__import__("trainer_v2.reward_v5", fromlist=["PRESETS"]).PRESETS))
    ap.add_argument("--rule-strata")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--metric", default="reward_mean", choices=A.METRICS)
    ap.add_argument("--final-holdout", action="store_true")
    ap.add_argument("--backend", default=os.environ.get("EVAL_BACKEND", "hf"), choices=["hf", "vllm"],
                    help="Qwen3.5 models only: hf (transformers generate) or vllm (eval_v1/vllm_backend.py)")
    ap.add_argument("--context-k", type=int, default=TK.FULL_K, choices=TK.CONTEXT_KS,
                    help="top-K retrieved memories kept at test time (eval_v1/topk_context.py; env CONTEXT_K)")
    ap.add_argument("--online-k", type=int, default=None, choices=TK.CONTEXT_KS,
                    help="v4: online SigLIP2 retrieval at K (eval_v1/online_retrieval.py; env ONLINE_K; default off)")
    ap.add_argument("--online-query-emb", default=None,
                    help="v4: fresh query-embedding sidecar dir for --online-k (default: frozen s0 query views)")
    ap.add_argument("--stage1-from", default=os.environ.get("STAGE1_FROM") or None,
                    help="run dir whose greedy memory_call/retrieved Stage-1 texts are reused (top-K: remove decision noise)")
    ap.add_argument("--fast-conv", action="store_true", default=os.environ.get("FAST_CONV") == "1",
                    help="fla triton causal conv (trainer_v2/fast_conv.py; env FAST_CONV=1)")
    ap.add_argument("--dry-run", action="store_true", help="api models: render requests + cost estimate, no network")
    ap.add_argument("--prereg-line", action="store_true", help="api models: print the registration line, no network")
    ap.add_argument("--prereg-ack", help="api models: sha12 of the logged --prereg-line output (paid run)")
    ap.add_argument("--allow-provider", action="append", help="api models: provider allowed to be called (default none)")
    ap.add_argument("--budget-usd", type=float, help="api models: hard per-run USD cap (ledger-checked)")
    ap.add_argument("--api-ledger", help="api models: ledger JSONL (default <out>.api/ledger.jsonl)")
    ap.add_argument("--api-base", help="api:local only: OpenAI-compatible base URL, e.g. http://127.0.0.1:8001/v1")
    ap.add_argument("--local-model-key", help="api:local only: eval_v1/local_provider.LOCAL_MODELS key (phi4_mm)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.compare:
        compare(Path(args.release), Path(args.compare), Path(args.out), metric=args.metric,
                draws=args.bootstrap, seed=args.seed, final_holdout=args.final_holdout)
        return 0
    if not args.release:
        ap.error("--release is required")
    check_model_args(args)
    if args.model.startswith("api:"):   # closed-source API path: no GPU (eval_v1/api_eval.py)
        from eval_v1 import api_eval
        return api_eval.main_api(args)
    D.release_file(Path(args.release), args.split, args.modes.split(",")[0], final_holdout=args.final_holdout)
    return gpu_main(args)


if __name__ == "__main__":
    sys.exit(main())
