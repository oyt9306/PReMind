#!/usr/bin/env python3
"""K-retention evaluator: does VCM RL training keep the model's use of TEXT memory? GPU only via kret_eval.sh.

Each reply is scored with reward_v4.0-passive (imported read-only, unchanged; the only input change is the
documented adjudication override in ``kret_data.scoring_view``) plus direct per-row metrics:
action accuracy, support-pattern hit and contrary-pattern hit on the parsed answer (case-insensitive regex, as in
the M1 application contract), risk accuracy and strict format; plus the lenient ``decisive_fact_hit`` and
``memory_follow_lenient`` (kret_lenient.py), reported side by side with the strict metrics. Outputs are write-once.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from eval_v1 import kret_data as KD  # noqa: E402
from eval_v1 import kret_lenient as KL  # noqa: E402
from eval_v1 import kret_stats as KS  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402

logger = logging.getLogger(__name__)
MARKER = "KRET EVAL DONE"
MAX_TOKENS = 512  # = trainer_v2 LoopConfig.max_completion_length
RV4 = KD.RV4


def _hit(patterns: list[str], text: str) -> bool:
    for pat in patterns:
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            rx = re.compile(re.escape(pat), re.I)
        if rx.search(text):
            return True
    return False


def score(item: KD.KretItem, completion: str, *, kind: str = "greedy", sample_idx: int = 0) -> dict:
    """One K-retention record (reward_v4 ledger summary + direct pattern metrics)."""
    led = RV4.reward_passive(item.view, completion)
    parsed = RV4.R33.R3._p(completion)
    answer = parsed["answer"] or ""
    contract = item.view["decisive_fact"]["application_contract"]
    tgt = led["target"]
    action_acc = parsed["action"] == tgt["expected_action"]
    pc_hit = _hit(contract["support_patterns"], answer)
    contrary = _hit(contract["contrary_patterns"], answer) if contract["contrary_patterns"] else None
    pc = led["components"]["personal_content"]
    return {"row_id": item.row_id, "family_id": item.family_id, "variant": item.variant, "kind": kind,
            "sample_idx": sample_idx, **item.meta,
            "action_pred": parsed["action"], "risk_pred": parsed["risk"],
            "format_strict": bool(parsed["format_strict"]), "action_acc": action_acc,
            "risk_acc": parsed["risk"] == tgt["risk_label"], "pc_hit": pc_hit, "contrary_hit": contrary,
            "memory_follow": action_acc and pc_hit and not contrary,
            "pc_reward_hit": (pc["value"] > 0) if pc["applicable"] else None,
            "reward_total": led["total"],
            "components": {k: c["value"] for k, c in led["components"].items() if c["applicable"]},
            **KL.fields(item.view, answer, action_acc)}


def generate(items: list[KD.KretItem], backend: object, *, max_tokens: int = MAX_TOKENS
             ) -> tuple[list[dict], list[dict]]:
    """Greedy single-turn generation through an ``HFBackend``-compatible backend."""
    recs, texts = [], []
    for item in items:
        images = KD.item_images(item)
        enc = backend.encode_prompt(item.messages, images)
        ids, _ = backend.generate([enc], images, num_generations=1, max_tokens=max_tokens, greedy=True)
        text = backend.decode(ids[0])
        recs.append(score(item, text))
        texts.append({"row_id": item.row_id, "variant": item.variant, "kind": "greedy", "sample_idx": 0,
                      "completion": text, "n_tokens": len(ids[0])})
    logger.info("kret generated %d items", len(items))
    return recs, texts


def write(out: Path, recs: list[dict], texts: list[dict], meta: dict, *, draws: int, seed: int,
          final_holdout: bool = False) -> dict:
    """Write-once records.jsonl, texts.jsonl, kret_report.json, KRET_REPORT.md."""
    if not final_holdout:
        P.refuse_holdout_path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty (write once): {out}")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"meta": meta, "table": KS.table(recs, draws=draws, seed=seed)}
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    (out / "kret_report.json").write_text(json.dumps(rep, indent=2))
    (out / "KRET_REPORT.md").write_text(KS.markdown(rep))
    return rep


def _read(path: Path, *, final_holdout: bool) -> list[dict]:
    if not final_holdout:
        P.refuse_holdout_path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(a: Path, b: Path, out: Path, *, draws: int, seed: int, final_holdout: bool = False) -> dict:
    """Paired B-minus-A deltas for every metric per variant|decoding setting (write-once)."""
    if not final_holdout:
        P.refuse_holdout_path(out)
    if out.exists():
        raise FileExistsError(out)
    ra = _read(a / "records.jsonl", final_holdout=final_holdout)
    rb = _read(b / "records.jsonl", final_holdout=final_holdout)
    def setting(r: dict) -> str:
        return f"{r['variant']}|{r['kind']}"
    result = {}
    for name in sorted({setting(r) for r in ra} | {setting(r) for r in rb}):
        sa, sb = [r for r in ra if setting(r) == name], [r for r in rb if setting(r) == name]
        result[name] = {m: KS.paired(sa, sb, metric=m, draws=draws, seed=seed) for m in KS.METRICS}
    out.mkdir(parents=True)
    (out / "paired_comparison.json").write_text(json.dumps({"a": str(a), "b": str(b), "by_setting": result},
                                                           indent=2))
    (out / "PAIRED_COMPARISON.md").write_text(KS.paired_markdown(result))
    return result


def gpu_main(args: argparse.Namespace) -> int:  # pragma: no cover - scheduler GPU job
    import torch
    from trainer_v2 import run_v2 as RV
    from trainer_v2.hf_backend import HFBackend
    RV.require_single_gpu()
    items, manifest = KD.load_kret(args.split, variants=tuple(args.variants.split(",")), limit=args.limit,
                                   final_holdout=args.final_holdout)
    if not items:
        raise ValueError("no K-retention items")
    model, proc = RV.load_model(lora=False, size=args.model)
    if args.lora:
        from peft import PeftModel
        P.refuse_holdout_path(args.lora)
        model = PeftModel.from_pretrained(model, args.lora).eval()
    backend = HFBackend(model, proc, temperature=1.0)
    t0 = time.time()
    recs, texts = generate(items, backend, max_tokens=args.max_tokens)
    meta = {"split": args.split, "variants": args.variants, "model": args.model, "model_pin": P.MODELS[args.model],
            "lora": args.lora, "max_tokens": args.max_tokens, "decoding": "greedy", "data": manifest,
            "runtime_s": round(time.time() - t0, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    write(Path(args.out), recs, texts, meta, draws=args.bootstrap, seed=args.seed,
          final_holdout=args.final_holdout)
    logger.info("%s items=%d out=%s", MARKER, len(recs), args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "holdout"))
    ap.add_argument("--variants", default="image,no_image")
    ap.add_argument("--model", default="4b", choices=sorted(P.MODELS))
    ap.add_argument("--lora")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--final-holdout", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), Path(args.out), draws=args.bootstrap,
                seed=args.seed, final_holdout=args.final_holdout)
        return 0
    KD.source_file(args.split, final_holdout=args.final_holdout)  # refuse before any model load
    return gpu_main(args)


if __name__ == "__main__":
    sys.exit(main())
