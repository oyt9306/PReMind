"""Per-stage backend dispatch (owner HYBRID split): every JSON-producing stage builds backend-neutral requests and
`run_json` sends them to the backend named in cfg.backends[stage]:
  gpt  -> guarded paid gpt-6-sol calls (gpt_client.call; one tag per request id, never re-paid);
  qwen -> one batched local Qwen3-VL-8B BF16 generate (xgrammar JSON, temperature 0) via a `generate` callable.
Request: {"id", "instructions", "payload", "schema", "name", "images": [(label, path)], "max_out", "effort"}.
"""
from __future__ import annotations

import json
from typing import Callable

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import gpt_client as gc

Generate = Callable[[list[dict]], list[str]]


def to_qwen(req: dict) -> dict:
    """Labelled images then the JSON payload; a request may instead carry its own `interleave` layout (the
    gpt_cap_gen memory blocks), to which the JSON payload is appended."""
    parts: list[tuple[str, str]] = list(req.get("interleave") or [])
    if not parts:
        for label, path in req.get("images", []):
            parts += [("text", label), ("image", path)]
    parts.append(("text", json.dumps(req["payload"], ensure_ascii=False)))
    return {"interleave": parts, "system": req["instructions"], "schema": req["schema"], "temperature": 0.0,
            "seed": cm.SEED, "max_tokens": min(int(req.get("max_out", 4000)), 8000)}


def run_json(cfg: cm.Cfg, stage: str, reqs: list[dict], gpt_fn: gc.GptFn | None = None,
             generate: Generate | None = None) -> dict[str, dict | None]:
    backend = cfg.backends[stage]
    if not reqs:
        return {}
    if backend == "gpt":
        return {r["id"]: gc.call(cfg, gpt_fn, r["instructions"], r["payload"], r["schema"], r["name"],
                                 f"{cfg.run_id}::{stage}::{r['id']}", r.get("images", []), r.get("max_out", 8000),
                                 r.get("effort", gc.GPT_EFFORT)) for r in reqs}
    if backend != "qwen":
        raise ValueError(f"unknown backend {backend!r} for stage {stage}")
    if generate is None:
        raise RuntimeError(f"stage {stage} runs on the local Qwen backend: launch it as a GPU job")
    out = {}
    for r, raw in zip(reqs, generate([to_qwen(r) for r in reqs])):
        try:
            out[r["id"]] = json.loads(raw)
        except json.JSONDecodeError:
            out[r["id"]] = None
    return out


def obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(item: dict, n: int | None = None) -> dict:
    a = {"type": "array", "items": item}
    if n is not None:
        a.update(minItems=n, maxItems=n)
    return a


def enum(vals) -> dict:  # noqa: ANN001
    return {"type": "string", "enum": list(vals)}


S, B = {"type": "string"}, {"type": "boolean"}
