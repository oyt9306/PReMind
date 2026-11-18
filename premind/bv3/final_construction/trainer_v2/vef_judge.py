"""VEF judge client (general_mix_v1/DESIGN.md §2.7): an OpenAI-compatible vLLM server on localhost (text-only
Qwen3.5-9B, served as ``vef-judge``; never started by this module).

* URL: ``--vef-judge-url`` / env VEF_JUDGE_URL (e.g. http://127.0.0.1:8016/v1); a non-localhost URL is refused.
  Model name: env VEF_JUDGE_MODEL (default "vef-judge").
* Requests: one user message, temperature 0, max_tokens 16, ``chat_template_kwargs {"enable_thinking": false}``.
* ``run``: one async batch per call, concurrency 64, per-request timeout 20 s, 3 retries (1 / 2 / 4 s backoff), and a
  wall-clock budget (the caller's remaining per-step deadline); a request without a reply after that is a failure.
* On-disk cache sha256(model + prompt) -> reply (JSONL, append-only) so a resumed run replays the same replies.
* ``health_check``: ``/v1/models`` must list the model, and 3 fixed calibration prompts must get their known answers;
  otherwise SystemExit (the run refuses to start).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from . import vef as VF

logger = logging.getLogger(__name__)
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_MODEL = "vef-judge"
TIMEOUT_S, RETRIES, BACKOFF_S, CONCURRENCY, MAX_TOKENS = 20.0, 3, (1.0, 2.0, 4.0), 64, 16
_N = {"polarity": "neg", "correct": "D", "question": VF.Q_NOBJ,
      "options": {"A": "dog", "B": "horse", "C": "guitar", "D": VF.D_TEXT}}
_H = {"polarity": "pos", "correct": "A", "question": VF.Q_HAZKIND,
      "options": {"A": "a person who has fallen, is falling or is lying on the floor", "B": "a wet or slippery floor",
                  "C": "exposed or sparking electrical wiring or cables", "D": VF.D_TEXT}}
_W = {"polarity": "neg", "correct": "B", "question": VF.Q_NHAZ,
      "options": {"A": "a heater or flame next to flammable material", "B": "a wet or slippery floor",
                  "C": "toxic vapor, smoke or fumes", "D": VF.D_TEXT}}
# answers stated verbatim in the description, so any working judge gets them right
CALIBRATION = ((VF.judge_prompt("In the kitchen, a person who has fallen, is falling or is lying on the floor is "
                                "visible next to a chair.", _H), "A"),
               (VF.judge_prompt("", _N), "D"),
               (VF.judge_prompt("A wet or slippery floor is visible: water is spilled across the bathroom tiles.", _W),
                "B"))


def check_url(url: str) -> str:
    u = urlparse(url)
    if u.scheme != "http" or u.hostname not in LOCAL_HOSTS or not u.port:
        raise SystemExit(f"REFUSED: VEF judge URL must be http://127.0.0.1:<port>/v1 (got {url!r})")
    return url.rstrip("/")


class VefJudge:
    def __init__(self, url: str, model: Optional[str] = None, cache_path: Optional[Path] = None, *,
                 timeout_s: float = TIMEOUT_S, retries: int = RETRIES, backoff_s: tuple = BACKOFF_S,
                 concurrency: int = CONCURRENCY) -> None:
        self.url = check_url(url)
        self.model = model or os.environ.get("VEF_JUDGE_MODEL") or DEFAULT_MODEL
        self.timeout_s, self.retries, self.backoff_s, self.concurrency = timeout_s, retries, backoff_s, concurrency
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("model") == self.model:
                    self.cache[rec["key"]] = rec["reply"]
        self.served: dict = {}

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def body(self, prompt: str) -> dict:
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                "max_tokens": MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}}

    # ------------------------------------------------------------------ batch
    async def _one(self, client, sem, prompt: str) -> Optional[str]:
        for attempt in range(self.retries + 1):
            try:
                async with sem:
                    r = await client.post(f"{self.url}/chat/completions", json=self.body(prompt))
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"]
                    if isinstance(content, str):
                        return content
                logger.warning("VEF judge HTTP %s (attempt %d)", r.status_code, attempt + 1)
            except Exception as exc:  # noqa: BLE001 - network / parse errors are retried, then the call fails
                logger.warning("VEF judge request failed (attempt %d): %s: %s", attempt + 1, type(exc).__name__, exc)
            if attempt < self.retries:
                await asyncio.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
        return None

    async def _batch(self, prompts: list[str], budget_s: float) -> list[Optional[str]]:
        import httpx
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_s, limits=limits, trust_env=False) as client:
            tasks = [asyncio.ensure_future(self._one(client, sem, p)) for p in prompts]
            done, pending = await asyncio.wait(tasks, timeout=budget_s)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                logger.warning("VEF judge step deadline: %d / %d calls unanswered", len(pending), len(prompts))
        return [t.result() if t in done and not t.cancelled() else None for t in tasks]

    def run(self, prompts: list[str], budget_s: float, *, use_cache: bool = True
            ) -> tuple[dict[str, Optional[str]], dict]:
        """prompt -> reply (None = failed after retries or past the budget); cached prompts make no call."""
        t0 = time.monotonic()
        out = {p: self.cache.get(self.key(p)) if use_cache else None for p in prompts}
        miss = [p for p in prompts if out[p] is None]
        if miss and budget_s > 0:
            got = asyncio.run(self._batch(miss, budget_s))
            new = []
            for p, rep in zip(miss, got):
                out[p] = rep
                if rep is not None and use_cache:
                    self.cache[self.key(p)] = rep
                    new.append(json.dumps({"key": self.key(p), "model": self.model, "reply": rep}))
            if new and self.cache_path is not None:
                with self.cache_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(new) + "\n")
        return out, {"seconds": round(time.monotonic() - t0, 3), "cache_hits": len(prompts) - len(miss)}

    # ------------------------------------------------------------------ startup
    def models(self) -> list[dict]:
        import httpx
        r = httpx.get(f"{self.url}/models", timeout=self.timeout_s, trust_env=False)
        r.raise_for_status()
        return r.json().get("data") or []

    def health_check(self) -> dict:
        """SystemExit unless the server lists the model and answers the 3 calibration prompts correctly."""
        try:
            listed = {m.get("id"): m for m in self.models()}
        except Exception as exc:  # noqa: BLE001 - any failure refuses the run
            raise SystemExit(f"REFUSED: VEF judge not reachable at {self.url}: {type(exc).__name__}: {exc}") from exc
        if self.model not in listed:
            raise SystemExit(f"REFUSED: VEF judge {self.url} does not serve {self.model!r} (serves {sorted(listed)})")
        got, _ = self.run([p for p, _ in CALIBRATION], budget_s=3 * self.timeout_s, use_cache=False)
        bad = [(want, got[p]) for p, want in CALIBRATION if got[p] is None or VF.parse_choice(got[p]) != want]
        if bad:
            raise SystemExit(f"REFUSED: VEF judge calibration failed (want, reply): {bad}")
        self.served = {"id": self.model, "root": listed[self.model].get("root"), "url": self.url}
        logger.info("VEF judge healthy: %s", self.served)
        return self.served
