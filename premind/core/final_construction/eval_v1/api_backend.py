"""grpo_v2.Backend over a closed-source API (EVAL ONLY): same items, prompts, two-stage protocol, reward_v5 scoring.

``G.rollout`` works on token ids; here the ids are HANDLES into this backend's segment table, so the protocol code is
reused unchanged:
  * ``encode_prompt`` -> (h,) for the rendered messages (system + user turn with the query image / passive images);
  * ``generate`` returns [h_text] + [PAD] * (n_out - 1) (+ [EOS] when the turn ended naturally), n_out = the
    provider's visible output-token count, so ``n_stage2`` (degenerate gate, >= 512) counts provider tokens;
  * ``encode_env_turn`` -> (h,) for the env user turn; Stage 2 = [system, user prompt, assistant Stage-1 text,
    user env turn] - the same conversation the Qwen models continue, as provider-native turns;
  * ``decode`` concatenates the text handles (Stage-1 / Stage-2 text exactly as returned).
Calls go through ``CallStore``: a write-once JSONL response cache keyed by the canonical request hash (resume: a
re-run never re-pays a finished call) and a cost ledger (one JSON line per PAID call) with a hard USD cap checked
before every call against the worst case (estimated input + max output tokens, + the reasoning allowance for OpenAI
reasoning profiles). Ledger cost = the provider's reported usage (OpenAI: billed output INCLUDING reasoning tokens).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from eval_v1 import api_providers as AP
from trainer_v2.grpo_v2 import Enc, Images
from trainer_v2.hf_backend import template_messages

logger = logging.getLogger(__name__)
EOS, PAD, FIRST = 0, 1, 16
EMPTY_STAGE1 = "(empty response)"     # providers reject empty assistant text; substitution counted in stats


class BudgetExceededError(RuntimeError):
    """The ledger total (or the worst case of the next call) exceeds the per-run USD cap."""


def _merge(parts: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for k, v in parts:
        if k == "text" and not v:
            continue
        if k == "text" and out and out[-1][0] == "text":
            out[-1] = ("text", out[-1][1] + v)
        else:
            out.append((k, v))
    return tuple(out)


def _parts(content: list[dict]) -> list[tuple[str, str]]:
    tm, img = template_messages([{"role": "user", "content": content}])
    paths = iter(img)
    return [("image", next(paths)) if p["type"] == "image" else ("text", p["text"]) for p in tm[0]["content"]]


def estimate_input_tokens(model: str, req: AP.ApiRequest, sizes: dict[str, tuple[int, int]]) -> int:
    rule = AP.profile(model).image_rule
    text = req.system + "".join(v for _, parts in req.turns for k, v in parts if k == "text")
    return AP.text_tokens(text) + sum(AP.image_tokens(rule, *sizes[p]) for p in req.images()) + 8 * len(req.turns)


def request_key(model: str, req: AP.ApiRequest) -> str:
    canon = {"model": model, "params": AP.profile(model).params, "system": req.system, "max_tokens": req.max_tokens,
             "turns": [[role, [[k, AP.file_sha(v) if k == "image" else v] for k, v in parts]]
                       for role, parts in req.turns]}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, default=str).encode()).hexdigest()


def _append(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


class CallStore:
    """Response cache (resume) + cost ledger + budget cap for one provider model."""

    def __init__(self, model: str, cache: Path, ledger: Path, budget_usd: float,
                 call: Callable[[AP.ApiRequest], AP.ApiResponse]) -> None:
        if not budget_usd > 0:
            raise BudgetExceededError("a positive --budget-usd is required for API runs")
        self.model, self.cache, self.ledger, self.budget, self.call = model, cache, ledger, budget_usd, call
        cache.parent.mkdir(parents=True, exist_ok=True)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if cache.is_file():
            for line in cache.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.done.setdefault(rec["key"], rec["response"])
        self.spent = self.ledger_total(ledger)
        if self.spent >= budget_usd:
            raise BudgetExceededError(f"REFUSED: ledger {ledger} already at ${self.spent:.4f} >= cap ${budget_usd}")
        self.n_cached = self.n_paid = self.reasoning_tokens = 0

    @staticmethod
    def ledger_total(ledger: Path) -> float:
        if not ledger.is_file():
            return 0.0
        return sum(json.loads(x)["est_cost_usd"] for x in ledger.read_text().splitlines() if x.strip())

    def __call__(self, req: AP.ApiRequest, sizes: dict[str, tuple[int, int]]) -> AP.ApiResponse:
        key = request_key(self.model, req)
        if key in self.done:
            self.n_cached += 1
            return AP.ApiResponse(**self.done[key])
        worst = AP.cost_usd(self.model, estimate_input_tokens(self.model, req, sizes),
                            req.max_tokens + AP.reasoning_allowance(self.model))
        if self.spent + worst > self.budget:
            raise BudgetExceededError(f"STOP: spent ${self.spent:.4f} + worst case ${worst:.4f} > cap ${self.budget}")
        resp = self.call(req)
        reasoning = int(resp.extra.get("reasoning_tokens", resp.extra.get("thoughts_tokens", 0)) or 0)
        billed_in = int(resp.extra.get("billed_input_tokens", resp.input_tokens))
        billed_out = resp.extra.get("billed_output_tokens")      # OpenAI: usage.output_tokens incl. reasoning
        billed_out = int(billed_out) if billed_out is not None else resp.output_tokens + int(
            resp.extra.get("thoughts_tokens", 0))
        cost = AP.cost_usd(self.model, billed_in, billed_out)
        self.spent += cost
        self.n_paid += 1
        self.reasoning_tokens += reasoning
        _append(self.ledger, {"ts": time.time(), "model": self.model, "key": key, "input_tokens": resp.input_tokens,
                              "output_tokens": resp.output_tokens, "billed_output_tokens": billed_out,
                              "reasoning_tokens": reasoning, "extra": resp.extra, "est_cost_usd": cost})
        rec = {"text": resp.text, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
               "stopped": resp.stopped, "raw_stop": resp.raw_stop, "extra": resp.extra}
        _append(self.cache, {"key": key, "response": rec})
        self.done[key] = rec
        return resp


class ApiBackend:
    """grpo_v2.Backend whose token ids are segment handles (module docstring)."""

    eos_token_id = EOS

    def __init__(self, model: str, call: Callable[[AP.ApiRequest, dict], AP.ApiResponse]) -> None:
        AP.profile(model)
        self.model, self.call = model, call
        self.seg: list[tuple[str, Any]] = [("reserved", None)] * FIRST
        self.sizes: dict[str, tuple[int, int]] = {}
        self.stats = {"calls": 0, "empty_stage1_substituted": 0, "stop": {}}

    def _new(self, kind: str, value: Any) -> int:
        self.seg.append((kind, value))
        return len(self.seg) - 1

    def _note_sizes(self, paths: Sequence[str], images: Images) -> None:
        for p in paths:
            im = images[p]
            self.sizes[p] = im.size if isinstance(im, Image.Image) else tuple(im)

    def encode_prompt(self, messages: list[dict], images: Images) -> Enc:
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if len(system) > 1 or [m["role"] for m in rest] != ["user"]:
            raise ValueError("API path expects [system?, user] prompt messages")
        sys_text = "".join(p["text"] for p in system[0]["content"]) if system else ""
        if system and any(p["type"] != "text" for p in system[0]["content"]):
            raise ValueError("system message must be text only")
        parts = _parts(rest[0]["content"])
        paths = tuple(v for k, v in parts if k == "image")
        self._note_sizes(paths, images)
        return Enc((self._new("prompt", (sys_text, _merge(parts))),), paths)

    def encode_env_turn(self, content: list[dict], images: Images, *, stage1_ended_with_eos: bool) -> Enc:
        parts = _parts(content)
        paths = tuple(v for k, v in parts if k == "image")
        self._note_sizes(paths, images)
        return Enc((self._new("env", _merge(parts)),), paths)

    def encode_text(self, text: str) -> list[int]:
        return [self._new("text", text)]

    def request(self, seq: Enc, max_tokens: int) -> AP.ApiRequest:
        """Provider-neutral conversation of a sequence of handles (prompt, [Stage-1 text, env turn])."""
        kind, (system, parts) = self.seg[seq.ids[0]]
        if kind != "prompt":
            raise ValueError("a sequence must start with a prompt handle")
        turns: list[tuple[str, tuple]] = [("user", parts)]
        asst: list[str] = []
        for i in seq.ids[1:]:
            if i in (EOS, PAD):
                continue
            k, v = self.seg[i]
            if k == "text":
                asst.append(v)
            elif k == "env":
                text = "".join(asst)
                if not text.strip():
                    self.stats["empty_stage1_substituted"] += 1
                    text = EMPTY_STAGE1
                turns += [("assistant", (("text", text),)), ("user", v)]
                asst = []
            else:
                raise ValueError(f"unexpected segment {k!r} inside a sequence")
        if asst:
            raise ValueError("sequence ends inside an assistant turn (no prefill on the API path)")
        return AP.ApiRequest(system, tuple(turns), max_tokens)

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:
        if not greedy or num_generations != 1 or teacher or gen_batch:
            raise ValueError("API path: greedy-equivalent single generation only (--samples 0)")
        outs: list[list[int]] = []
        for seq in seqs:
            resp = self.call(self.request(seq, max_tokens), self.sizes)
            self.stats["calls"] += 1
            self.stats["stop"][resp.raw_stop] = self.stats["stop"].get(resp.raw_stop, 0) + 1
            n = max(0, int(resp.output_tokens))
            body = [self._new("text", resp.text)] + [PAD] * (n - 1) if n else []
            if not n and resp.text:
                body = [self._new("text", resp.text)]
            outs.append(body + ([EOS] if resp.stopped else []))
        return outs, [None] * len(outs)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.seg[i][1] for i in ids if i >= FIRST and self.seg[i][0] == "text")

    def logits_at(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError("API backend is eval only")

    def logits_at_batch(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError("API backend is eval only")
