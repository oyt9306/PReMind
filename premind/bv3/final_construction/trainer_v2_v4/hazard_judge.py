"""In-loop open-ended hazard judge (backend "judge" of the v4wh reward). Reuses the VEF judge infrastructure
(trainer_v2/vef_judge.VefJudge, import only): an OpenAI-compatible vLLM server on localhost serving the text-only
Qwen3.5-9B snapshot (default served name ``vef-judge``, e.g. http://127.0.0.1:8016/v1; non-localhost refused), async
batch per rollout round, concurrency 64, 20 s timeout, 3 retries, append-only reply cache (a resumed run replays the
same replies), greedy (T 0), thinking off. Differences: ``max_tokens`` 24 (three 0/1 fields) and its own calibration
(hazard_rubric.CALIBRATION). A failed / unparseable reply -> that episode is scored by the cue path (logged).
Server launch (print-only): ``bash trainer_v2_v4/scripts/hazard_judge_server.sh``.
The API backend (owner 2026-10-14 default: gpt-6-luna@medium) is hazard_judge_api.ApiHazardJudge; ``prefetch`` works
with either.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from trainer_v2 import vef_judge as VJ
from trainer_v2_v4 import hazard_rubric as HR

logger = logging.getLogger(__name__)
MAX_TOKENS = 24
STEP_BUDGET_S = 60.0


class HazardJudge(VJ.VefJudge):
    def body(self, prompt: str) -> dict:
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                "max_tokens": MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}}

    def health_check(self, calibration: tuple = HR.CALIBRATION) -> dict:
        """SystemExit unless the server lists the model and answers the rubric calibration prompts correctly."""
        try:
            listed = {m.get("id"): m for m in self.models()}
        except Exception as exc:  # noqa: BLE001 - any failure refuses the run
            raise SystemExit(f"REFUSED: hazard judge not reachable at {self.url}: {type(exc).__name__}: {exc}") from exc
        if self.model not in listed:
            raise SystemExit(f"REFUSED: hazard judge {self.url} does not serve {self.model!r} (serves {sorted(listed)})")
        got, _ = self.run([p for p, _ in calibration], budget_s=3 * self.timeout_s, use_cache=False)
        bad = []
        for p, want in calibration:
            parsed = HR.parse_reply(got[p])
            if parsed is None or any(parsed[k] != v for k, v in want.items()):
                bad.append((want, got[p]))
        if bad:
            raise SystemExit(f"REFUSED: hazard judge calibration failed (want, reply): {bad}")
        self.served = {"id": self.model, "root": listed[self.model].get("root"), "url": self.url,
                       "rubric": HR.RUBRIC_VERSION, "rubric_sha256": HR.RUBRIC_SHA256}
        logger.info("hazard judge healthy: %s", self.served)
        return self.served


def prefetch(jobs: list[tuple[Any, str]], judge: Any, budget_s: float = STEP_BUDGET_S) -> dict:
    """``jobs`` = [(episode, prompt)]; sets ``ep.hazard_judge`` = parsed reply | None (-> cue fallback) and
    ``ep.hazard_judge_raw``. Returns call stats."""
    t0 = time.monotonic()
    uniq = list(dict.fromkeys(p for _, p in jobs))
    replies, st = judge.run(uniq, budget_s) if uniq else ({}, {})
    failed = 0
    for ep, p in jobs:
        raw = replies.get(p)
        ep.hazard_judge_raw = raw
        ep.hazard_judge = HR.parse_reply(raw)
        if ep.hazard_judge is not None:                  # the pinned model spec travels with every judgment
            ep.hazard_judge["model"] = getattr(judge, "model", None)
        failed += ep.hazard_judge is None
    if failed:
        logger.warning("hazard judge: %d / %d episodes unjudged -> cue fallback", failed, len(jobs))
    return {"calls": len(uniq), "failed": failed, "cache_hits": st.get("cache_hits", 0),
            "seconds": round(time.monotonic() - t0, 3)}


__all__ = ["HazardJudge", "prefetch", "MAX_TOKENS", "STEP_BUDGET_S"]
