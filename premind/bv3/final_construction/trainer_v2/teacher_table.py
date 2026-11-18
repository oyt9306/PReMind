#!/usr/bin/env python3
"""Teacher table for privileged OPSD (schema v2): hinted frozen base per (arm, row) -> reward v5 Stage-2 checks.

GPU path (``build``) only through teacher_table.sh (scheduler). The CPU core ``build_table`` is shared with the tests
(mock backend). Memory-call rows: Stage 1 is the row's correct decision (fixed text, never hinted, never distilled);
the hint sits in the Stage-2 env turn. Each row: greedy first; if greedy fails, ``best_of`` samples at
``temperature`` and the first passing sample is the reference. ``teacher_ok`` = some candidate passes format, risk,
action, exact citation set and the degeneration gate. The summary reports greedy vs best-of-n pass rates and the
reward checks that block the failing rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2 import grpo_v2 as G  # noqa: E402
from trainer_v2 import opsd as OP  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402
from trainer_v2 import reward_v5 as R5  # noqa: E402
from trainer_v2 import vcm_data as VD  # noqa: E402

logger = logging.getLogger(__name__)
MARKER = "TRAINER V2 TEACHER TABLE DONE"


def build_table(items: list, loader: Callable[[Any], dict], backend: Any, cfg: G.LoopConfig, hint_set: str,
                meta: dict, *, best_of: int = 4, temperature: float = 0.7) -> dict:
    """Hinted teacher episodes for every item; returns the table (schema TABLE_SCHEMA)."""
    hint_ids = OP.HINT_SETS[hint_set]
    entries: dict[str, dict] = {}
    for it in items:
        block, audit = OP.build_hint(it, hint_ids)
        if block is None:
            entries[OP.table_key(it)] = OP.table_entry(it, audit, None, [])
            continue
        imgs = loader(it)
        kw = dict(teacher=True, ctx_fn=lambda x, b=block: OP.teacher_context(x, b), forced_stage1=_tag)
        results = [_scored("greedy", G.rollout([it], [imgs], backend, cfg, greedy=True, num_generations=1, **kw)[0],
                           cfg)]
        if OP.failure_reasons(results[0][1], stage2_only=True) and best_of > 0:
            eps = G.rollout([it], [imgs], backend, cfg, greedy=False, num_generations=best_of,
                            temperature=temperature, **kw)
            results += [_scored(f"sample{j}", ep, cfg) for j, ep in enumerate(eps)]
        entries[OP.table_key(it)] = OP.table_entry(it, audit, block, results)
    return {"schema": OP.TABLE_SCHEMA, "meta": {**meta, "hint_set": hint_set, "hint_ids": list(hint_ids),
                                                "reward_v5_sha256": R5.SOURCE_SHA256,
                                                "best_of": best_of, "sample_temperature": temperature},
            "summary": summarize(entries), "entries": entries}


def _tag(item: Any) -> str:
    return f"<memory_call>{OP.correct_decision(item)}</memory_call>"


def _scored(kind: str, ep: Any, cfg: G.LoopConfig) -> tuple[str, dict, str]:
    text = ep.stage2_text if ep.stage1_text is None else ep.stage1_text + "\n[STAGE2]\n" + ep.stage2_text
    return kind, G.score(ep, cfg), text


def summarize(entries: dict) -> dict:
    hinted = [e for e in entries.values() if e["reason"] != "no_clean_hint"]
    failing = [e for e in hinted if not e["teacher_ok"]]
    return {"n": len(entries), "n_hinted": len(hinted), "teacher_ok": sum(e["teacher_ok"] for e in hinted),
            "greedy_ok": sum(e["greedy_ok"] for e in hinted),
            "best_of_n_only_ok": sum(e["teacher_ok"] and not e["greedy_ok"] for e in hinted),
            "reasons": dict(Counter(e["reason"].split(":")[0] for e in entries.values())),
            "blocking_terms_greedy": dict(Counter(t for e in hinted if not e["greedy_ok"] for t in e["greedy_reasons"])),
            "blocking_terms_unresolved": dict(Counter(t for e in failing for t in e["greedy_reasons"])),
            "blocking_terms_samples": dict(Counter(t for e in failing for r in e["sample_reasons"] for t in r)),
            "teacher_ok_by_hint": dict(Counter(h for e in hinted if e["teacher_ok"] for h in e["hints"])),
            "teacher_ok_by_arm": dict(Counter(k.split("|")[0] for k, e in entries.items() if e["teacher_ok"])),
            "dropped_hints": dict(Counter(h for e in entries.values() for h in e.get("dropped", {})))}


def gpu_build(args: argparse.Namespace) -> int:  # pragma: no cover - GPU job (teacher_table.sh)
    import torch

    from trainer_v2 import run_v2 as RV
    from trainer_v2.hf_backend import HFBackend
    RV.require_single_gpu()
    out = Path(args.out)
    P.refuse_holdout_path(out)
    arms = tuple(args.arms.split(","))
    items, manifest = VD.load_items(Path(args.rows), arms=arms, mode=args.mode, drop_p=args.drop_p,
                                    order=args.order, require_accepted=not args.allow_unaccepted,
                                    splits=tuple(args.splits.split(",")))
    items = RV.subset(items, args.limit)
    t0 = time.time()
    model, proc = RV.load_model(lora=False)
    backend = HFBackend(model, proc, temperature=1.0)
    cfg = G.LoopConfig(r=args.r)
    meta = {"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "teacher": "frozen base (no LoRA)",
            "decoding": "greedy, then best-of-n sampled if greedy fails", "data": manifest, "arms": list(arms), "created": time.strftime("%F %T")}
    table = build_table(items, VD.item_images, backend, cfg, args.hint_set, meta, best_of=args.best_of,
                        temperature=args.temperature)
    table["meta"]["runtime_s"] = round(time.time() - t0, 1)
    table["meta"]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True))
    s = table["summary"]
    print(f"{MARKER} n={s['n']} hinted={s['n_hinted']} greedy_ok={s['greedy_ok']} teacher_ok={s['teacher_ok']} "
          f"blocking={json.dumps(s['blocking_terms_unresolved'])} out={out}", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(P.CANARY_RENDERED))
    ap.add_argument("--out", required=True)
    ap.add_argument("--hint-set", default="main", choices=sorted(OP.HINT_SETS))
    ap.add_argument("--arms", default="passive,memory_call")
    ap.add_argument("--mode", default="retrieved", choices=VD.CONTEXT_MODES)
    ap.add_argument("--drop-p", type=float, default=0.2)
    ap.add_argument("--order", default="display", choices=VD.ORDERS)
    ap.add_argument("--limit", type=int, default=None, help="rows per arm (tiny subset); omit for all rows")
    ap.add_argument("--splits", default="train", help="comma list from train,dev (training uses train only)")
    ap.add_argument("--best-of", type=int, default=4, help="sampled candidates when greedy fails (0 = greedy only)")
    ap.add_argument("--temperature", type=float, default=0.7, help="best-of-n sampling temperature")
    ap.add_argument("--r", type=float, default=0.1, help="decision magnitude for scoring only (plumbing)")
    ap.add_argument("--allow-unaccepted", action="store_true", help="rendered rows without gate acceptance")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return gpu_build(args)


if __name__ == "__main__":
    sys.exit(main())
