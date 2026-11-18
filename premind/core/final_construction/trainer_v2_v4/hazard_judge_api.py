"""API backend of the v4wh hazard judge (owner 2026-10-14: gpt-6-luna at medium effort, paid use approved).

Same interface as the local backend (hazard_judge.HazardJudge): ``run(prompts, budget_s) -> ({prompt: reply|None},
stats)`` and ``health_check()``; failures / caps -> None -> the cue fallback (hazard_judge.prefetch).
* Client: the repo's OpenAI Responses client (eval_v1/api_providers.openai_call: store=False, reasoning effort from the
  model profile, usage from the API) with ``with_retries`` (3 attempts, 2 / 4 s backoff) and a per-call timeout
  (OpenAI(timeout=30 s, max_retries=0)). Model pinned as a spec string (default ``gpt-6-luna@medium``); the spec and
  the API-reported model are recorded with every judgment and in the ledger.
* Prompt caching: instructions = hazard_rubric.PREFIX (header + the frozen rubric, identical on every call), input =
  the per-item text, so the provider's prefix cache applies.
* Usage ledger (append-only JSONL, default OUT/usage_hazard_judge.jsonl): its directory is created in __init__,
  BEFORE any paid call. One line per billed call: time, model spec, actual model, input / cached / output / reasoning
  tokens, usd (full input price; cached discount ignored = conservative), prompt sha256, status. Spend already in the
  ledger counts against the caps (resume).
* Run caps: ``usd_cap`` (default $5) and ``token_cap`` (default 10M input+output tokens). A call is only started while
  spent + a per-call reserve stays under both caps; once a cap is hit every further prompt returns None (cue
  fallback) with one warning, and ``stats()["capped"]`` is set.
* Concurrency: a thread pool of ``concurrency`` (default 8) per step; the step budget bounds the wait.
* Reply cache (optional JSONL, sha256(model spec + prompt) -> reply): a resumed run replays the same judgments.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from eval_v1 import api_providers as AP     # read-only use
from trainer_v2_v4 import hazard_rubric as HR

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-6-luna@medium"
USD_CAP, TOKEN_CAP, CONCURRENCY, TIMEOUT_S, ATTEMPTS = 5.0, 10_000_000, 8, 30.0, 3
MAX_VISIBLE = 24                      # visible reply cap (three 0/1 fields); the profile adds the reasoning allowance
RESERVE_TOKENS = 4000                 # per-call reserve checked against the caps before a call starts
CallFn = Callable[[str, "AP.ApiRequest"], "AP.ApiResponse"]


def _no_truncate(text: str, cap: int) -> tuple[str, int, bool]:
    return text, len(text.split()), False


def default_call(model: str, timeout_s: float) -> CallFn:  # pragma: no cover - network (paid)
    from openai import OpenAI
    client = OpenAI(api_key=AP.require_key("openai"), max_retries=0, timeout=timeout_s)
    return lambda m, req: AP.openai_call(m, req, client=client, truncate=_no_truncate)


class ApiHazardJudge:
    def __init__(self, usage_log: Path, *, model: str = DEFAULT_MODEL, usd_cap: float = USD_CAP,
                 token_cap: int = TOKEN_CAP, concurrency: int = CONCURRENCY, timeout_s: float = TIMEOUT_S,
                 attempts: int = ATTEMPTS, cache_path: Optional[Path] = None, call: Optional[CallFn] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        prof = AP.profile(model)                          # validates the spec (and the effort) before anything else
        if prof.provider != "openai":
            raise SystemExit(f"REFUSED: hazard judge API backend supports OpenAI models only (got {model!r})")
        self.model, self.prof = model, prof
        self.usd_cap, self.token_cap = float(usd_cap), int(token_cap)
        self.concurrency, self.timeout_s, self.attempts, self.sleep = concurrency, timeout_s, attempts, sleep
        self.usage_log = Path(usage_log)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)     # BEFORE any paid call (owner: past crash)
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._call = call
        self._lock = threading.Lock()
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("model") == self.model:
                    self.cache[rec["key"]] = rec["reply"]
        self.spent_usd, self.spent_tokens, self.n_calls, self.n_failed = 0.0, 0, 0, 0
        if self.usage_log.is_file():                      # resume: earlier spend counts against the caps
            for line in self.usage_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.spent_usd += float(r.get("usd") or 0.0)
                    self.spent_tokens += int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
        self.capped: Optional[str] = None
        self.last_model: Optional[str] = None
        self.served: dict = {}

    # ------------------------------------------------------------------ budget
    def _reserve_ok(self) -> bool:
        reserve_usd = AP.cost_usd(self.model, RESERVE_TOKENS, RESERVE_TOKENS)
        with self._lock:
            if self.capped:
                return False
            if self.spent_usd + reserve_usd > self.usd_cap:
                self.capped = f"usd_cap {self.usd_cap} (spent {self.spent_usd:.4f})"
            elif self.spent_tokens + 2 * RESERVE_TOKENS > self.token_cap:
                self.capped = f"token_cap {self.token_cap} (spent {self.spent_tokens})"
            if self.capped:
                logger.warning("hazard judge API cap hit: %s -> cue fallback for the rest of the run", self.capped)
                return False
            return True

    def _ledger(self, prompt: str, resp: Optional["AP.ApiResponse"], status: str) -> None:
        ex = (resp.extra if resp is not None else {}) or {}
        usage = ex.get("usage") or {}
        cached = int(((usage.get("input_tokens_details") or {}).get("cached_tokens")) or 0)
        tin = int(ex.get("billed_input_tokens", getattr(resp, "input_tokens", 0) or 0) or 0)
        tout = int(ex.get("billed_output_tokens", getattr(resp, "output_tokens", 0) or 0) or 0)
        usd = AP.cost_usd(self.model, tin, tout)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": "hazard_judge_v4wh", "model": self.model,
               "actual_model": ex.get("actual_model"), "reasoning_effort": AP.reasoning_effort(self.model),
               "input_tokens": tin, "cached_input_tokens": cached, "output_tokens": tout,
               "reasoning_tokens": int(ex.get("reasoning_tokens") or 0), "usd": round(usd, 8), "status": status,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "rubric_sha256": HR.RUBRIC_SHA256}
        with self._lock:
            self.spent_usd += usd
            self.spent_tokens += tin + tout
            if ex.get("actual_model"):
                self.last_model = ex["actual_model"]
            with self.usage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ calls
    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def request(self, prompt: str) -> "AP.ApiRequest":
        instructions, item = HR.split_prompt(prompt)
        return AP.ApiRequest(system=instructions, turns=(("user", (("text", item),)),), max_tokens=MAX_VISIBLE)

    def _one(self, prompt: str) -> Optional[str]:
        if not self._reserve_ok():
            return None
        if self._call is None:                           # pragma: no cover - network
            self._call = default_call(self.model, self.timeout_s)
        try:
            resp = AP.with_retries(lambda: self._call(self.model, self.request(prompt)), attempts=self.attempts,
                                   base=2.0, cap=8.0, sleep=self.sleep)
        except Exception as exc:  # noqa: BLE001 - any failure -> cue fallback (counted, logged)
            logger.warning("hazard judge API call failed: %s: %s", type(exc).__name__, str(exc)[:200])
            with self._lock:
                self.n_failed += 1
            return None
        self._ledger(prompt, resp, resp.raw_stop)
        with self._lock:
            self.n_calls += 1
        return resp.text if resp.stopped or resp.text else None

    def run(self, prompts: list[str], budget_s: float, *, use_cache: bool = True
            ) -> tuple[dict[str, Optional[str]], dict]:
        t0 = time.monotonic()
        out = {p: self.cache.get(self.key(p)) if use_cache else None for p in prompts}
        miss = [p for p in dict.fromkeys(prompts) if out[p] is None]
        if miss and budget_s > 0 and not self.capped:
            ex = cf.ThreadPoolExecutor(max_workers=self.concurrency)
            futs = {ex.submit(self._one, p): p for p in miss}
            done, pending = cf.wait(futs, timeout=budget_s)
            ex.shutdown(wait=False, cancel_futures=True)
            if pending:
                logger.warning("hazard judge API step budget: %d / %d calls unanswered -> cue", len(pending), len(miss))
            new = []
            for f in done:
                p, rep = futs[f], f.result()
                out[p] = rep
                if rep is not None and use_cache:
                    self.cache[self.key(p)] = rep
                    new.append(json.dumps({"key": self.key(p), "model": self.model, "reply": rep}))
            if new and self.cache_path is not None:
                with self.cache_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(new) + "\n")
        return out, {"seconds": round(time.monotonic() - t0, 3), "cache_hits": len(prompts) - len(miss)}

    def stats(self) -> dict:
        return {"usd_spent": round(self.spent_usd, 6), "tokens_spent": self.spent_tokens, "api_calls": self.n_calls,
                "api_failed": self.n_failed, "capped": self.capped is not None}

    def health_check(self, calibration: tuple = HR.CALIBRATION) -> dict:
        """SystemExit unless the 3 rubric calibration prompts get their known answers (paid: 3 calls)."""
        got, _ = self.run([p for p, _ in calibration], budget_s=3 * self.timeout_s * self.attempts,
                          use_cache=False)
        bad = []
        for p, want in calibration:
            parsed = HR.parse_reply(got[p])
            if parsed is None or any(parsed[k] != v for k, v in want.items()):
                bad.append((want, got[p]))
        if bad:
            raise SystemExit(f"REFUSED: hazard judge ({self.model}) calibration failed (want, reply): {bad}")
        self.served = {"id": self.model, "actual_model": self.last_model, "provider": "openai",
                       "rubric": HR.RUBRIC_VERSION, "rubric_sha256": HR.RUBRIC_SHA256, "usd_cap": self.usd_cap,
                       "token_cap": self.token_cap, "usage_log": str(self.usage_log)}
        logger.info("hazard judge (API) healthy: %s", self.served)
        return self.served


__all__ = ["ApiHazardJudge", "DEFAULT_MODEL", "USD_CAP", "TOKEN_CAP", "CONCURRENCY", "TIMEOUT_S"]
