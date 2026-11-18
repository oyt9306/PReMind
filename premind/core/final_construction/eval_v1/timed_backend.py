"""Inference-cost measurement around any rollout backend (HFBackend / BaselineBackend / GenericBackend / mocks).

``TimedBackend`` delegates every attribute to the wrapped backend and logs each ``generate`` call: prefill (input)
tokens and images of the prompt sequence, generated tokens per completion and wall-clock seconds
(``time.perf_counter``; ``torch.cuda.synchronize`` before and after when CUDA is available). hf_backend.py is not
edited. ``episode_cost`` maps the calls of one ``grpo_v2.rollout`` of ONE item to its episodes:
passive = one call (G completions); memory_call = one Stage-1 call (G completions) + one Stage-2 call per episode.
With G > 1 the passive / Stage-1 call is shared by the G episodes (``shared_call`` True; seconds are not split).
"""
from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)
COST_KEYS = ("stage1_input_tokens", "stage1_images", "stage1_gen_tokens", "stage1_seconds", "stage2_input_tokens",
             "stage2_images", "stage2_gen_tokens", "stage2_seconds", "input_tokens", "n_images", "gen_tokens",
             "seconds", "shared_call")


def _sync() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in qwen35_trl
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimedBackend:
    """Transparent proxy; ``calls`` holds one dict per generate call since the last ``take``."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.calls: list[dict] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def generate(self, seqs: list, images: Any, **kw: Any) -> Any:
        _sync()
        t0 = time.perf_counter()
        out = self._backend.generate(seqs, images, **kw)
        _sync()
        comps = out[0]
        self.calls.append({"input_tokens": len(seqs[0].ids), "images": len(seqs[0].images),
                           "gen_tokens": [len(c) for c in comps], "seconds": time.perf_counter() - t0})
        return out

    def take(self) -> list[dict]:
        calls, self.calls = self.calls, []
        return calls


def episode_cost(calls: list[dict], arm: str, idx: int, forced_stage1: bool = False) -> dict:
    """Cost fields of episode ``idx`` of one rollout (see module doc). ``forced_stage1``: Stage 1 was not generated
    (--stage1-from), so the calls hold Stage-2 generations only and the Stage-1 cost fields are None."""
    if arm != "passive" and forced_stage1:
        s1, s2, shared = None, (calls[idx], calls[idx]["gen_tokens"][0]), False
    elif arm == "passive":
        c = calls[0]
        s1, s2, shared = None, (c, c["gen_tokens"][idx]), len(c["gen_tokens"]) > 1
    else:
        c1, c2 = calls[0], calls[1 + idx]
        s1, s2, shared = (c1, c1["gen_tokens"][idx]), (c2, c2["gen_tokens"][0]), len(c1["gen_tokens"]) > 1
    out: dict = {"shared_call": shared}
    for name, part in (("stage1", s1), ("stage2", s2)):
        call, gen = part if part else ({}, None)
        out.update({f"{name}_input_tokens": call.get("input_tokens"), f"{name}_images": call.get("images"),
                    f"{name}_gen_tokens": gen,
                    f"{name}_seconds": round(call["seconds"], 4) if part else None})
    parts = [p for p in (s1, s2) if p]
    out.update(input_tokens=sum(p[0]["input_tokens"] for p in parts), n_images=s2[0]["images"],
               gen_tokens=sum(p[1] for p in parts), seconds=round(sum(p[0]["seconds"] for p in parts), 4))
    return out


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "median": None, "p90": None}
    s = sorted(xs)
    return {"mean": round(statistics.fmean(s), 4), "median": round(statistics.median(s), 4),
            "p90": round(s[min(len(s) - 1, int(round(.9 * (len(s) - 1))))], 4)}


def latency_summary(recs: list[dict]) -> dict:
    """Per setting (arm|mode|kind): seconds per item (mean/median/p90), mean input tokens / images / gen tokens."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("seconds") is not None:
            by[f"{r['arm']}|{r['mode']}|{r['kind']}"].append(r)
    out = {}
    for name, rows in sorted(by.items()):
        def mean(key: str) -> Optional[float]:
            xs = [r[key] for r in rows if r.get(key) is not None]
            return round(statistics.fmean(xs), 2) if xs else None
        out[name] = {"n": len(rows), "seconds": _stats([r["seconds"] for r in rows]),
                     "mean_input_tokens": mean("input_tokens"), "mean_stage1_input_tokens": mean("stage1_input_tokens"),
                     "mean_stage2_input_tokens": mean("stage2_input_tokens"), "mean_images": mean("n_images"),
                     "mean_gen_tokens": mean("gen_tokens"),
                     "shared_call_share": round(sum(bool(r.get("shared_call")) for r in rows) / len(rows), 4)}
    return out
