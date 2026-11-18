"""Harm-severity cap on contraindicated acute rollouts (``--harm-severity-cap on``; CUR23, 2026-10-15). NEW file; every
pinned / existing module is untouched, ``off`` (default) installs nothing.

For every training rollout on an acute row that the hazard JUDGE (not the cue fallback) marks contraindicated, the
frozen severity v2 judge (v4_build/text_knowledge_probe/severity/severity_v2.py + RUBRIC_SEVERITY_v2.md, sha-pinned;
text only, gpt-6-luna@medium) grades the worst harmful instruction:
    critical  -> episode total capped:  total = min(total, CAP = -0.30)   (component ``harm_severity_cap`` = the delta)
    minor     -> unchanged (the existing contraindicated extra, GAMMA -0.15, stays the only penalty; delta 0.0)
    unjudged  -> unchanged (cap / failure / unparseable; logged, never silently capped)
The cap is the LAST transform of the episode reward (installed on reward_v4hg.score_episode after main_v4haa's
warning credit, i.e. outermost over warning credit, absent_visible_risk, acute_absent_fix, false-alarm pen and acute
memory bonus v1.1), so it bounds the final total. Not applicable (ledger unchanged): non-acute rows, cue / fallback
verdicts, not contraindicated, degenerate episodes (total already -1). Idempotent (an applied component is kept).
Ledger: component ``harm_severity_cap`` {applicable, value = delta, status active:<severity>[:capped]}, key
``harm_severity`` {severity, capped, base_total, cap, harmful_quote, version}, flags.harm_severity_cap = "on".

Severity item (severity_v1.item_text format): hazard_types / situation = the row's RowActions types + visible hazard,
expected_key_actions = their descriptions (LABEL_REMAP_V4 applied), earlier_flag = the hazard judge's reason + quote
(rubric v4; empty under v3), response = the G1-sanitised <answer>.
Plumbing: hazard_judge_v4.PurposeApiJudge (= hazard_judge_api.ApiHazardJudge client / retries / caps / append-only
ledger / cache / concurrency / step budget) with ledger purpose ``harm_severity_v2_train``, its own USD cap
(``--harm-severity-usd-cap``, split over DDP ranks), ledger OUT.judge/usage_harm_severity[.rank<r>].jsonl, cache
OUT.judge/harm_severity_cache[.rank<r>].jsonl, verdict log OUT.judge/harm_severity[.rank<r>].jsonl, a 2-call
calibration (adversarial_v2 C02 critical, M01 minor). Calls run right after the hazard judge batch of the step.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from trainer_v2_v4 import hazard_judge as HJ
from trainer_v2_v4 import hazard_judge_v4 as HJ4

logger = logging.getLogger(__name__)
MODES = ("off", "on")
CAP = -0.30
COMPONENT = "harm_severity_cap"
PURPOSE = "harm_severity_v2_train"
MODEL = "gpt-6-luna@medium"
USD_CAP, TOKEN_CAP = 1.0, 2_000_000
STEP_BUDGET_S = 60.0
SEV_DIR = HJ4.FC / "v4_build" / "text_knowledge_probe" / "severity"
SHA256 = {"severity_v1.py": "31545cadeacbc7b4b7951d81fc9aebf9f41121aaf79f714049abbf8606765534",
          "severity_v2.py": "7fa4676f8617e3d3b53080a074018bac58ff8fac5b22c2ca6fe49065b0fb7cd2",   # FREEZE_v2.json
          "adversarial_v2.jsonl": "f9cd41681bbe7b4fb40d6ac3f9c4901327dd42af639f080a068040210b1cf2c8"}
RUBRIC_SHA256 = "b7f608885900eabeb68b49e249bf5310d049c2f5dc288ff0887dfe7b36caaa5f"            # FREEZE_v2.json
CALIBRATION_IDS = {"C02": "critical", "M01": "minor"}
_S2 = None
_STATE: dict = {}


def _check(name: str) -> bytes:
    raw = (SEV_DIR / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256[name]:
        raise RuntimeError(f"{SEV_DIR / name} changed after its freeze (sha256)")
    return raw


def s2() -> Any:
    """severity_v2 (and its base severity_v1), imported by path, sha256-pinned."""
    global _S2
    if _S2 is None:
        for name in ("severity_v1", "severity_v2"):
            _check(f"{name}.py")
            if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(SEV_DIR / f"{name}.py"):
                continue
            spec = importlib.util.spec_from_file_location(name, SEV_DIR / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod                       # severity_v2 does `import severity_v1`
            spec.loader.exec_module(mod)
        _S2 = sys.modules["severity_v2"]
        if _S2.RUBRIC_SHA256 != RUBRIC_SHA256:
            raise RuntimeError("severity_v2 rubric sha256 is not the frozen v2 rubric")
    return _S2


def sev_item(ra: Any, row: dict, judge: Optional[dict]) -> dict:
    j = judge or {}
    return {"hazard_types": list(ra.types), "situation": ra.what,
            "key_actions": [a["desc"] for a in HJ4.item_of(ra, row)["key_actions"]],
            "earlier_reason": j.get("reason") or "the hazard judge marked this reply contraindicated",
            "earlier_quote": j.get("harmful_quote") or ""}


def item_text(ra: Any, row: dict, judge: Optional[dict], answer: str) -> str:
    from trainer_v2_v4 import reward_v4hg as RHG
    return s2().item_text(sev_item(ra, row, judge), RHG.guard().sanitize(answer))


class SeverityJudge(HJ4.PurposeApiJudge):
    purpose, version, rubric_sha256 = PURPOSE, "tkp_severity_v2", RUBRIC_SHA256

    def __init__(self, usage_log: Path, **kw: Any) -> None:
        kw.setdefault("model", MODEL)
        kw.setdefault("usd_cap", USD_CAP)
        kw.setdefault("token_cap", TOKEN_CAP)
        super().__init__(usage_log, **kw)
        S = s2()
        self.system = S.SYSTEM_HEAD + S.rubric_text()

    def health_check(self) -> dict:
        S = s2()
        cases = {c["case_id"]: c for c in map(json.loads, _check("adversarial_v2.jsonl").decode().splitlines()) if c}
        cal = [(S.item_text(cases[k]["item"], cases[k]["answer"]), want) for k, want in CALIBRATION_IDS.items()]
        got, _ = self.run([p for p, _ in cal], budget_s=3 * self.timeout_s * self.attempts, use_cache=False)
        bad = [(want, got[p]) for p, want in cal if (S.parse_reply(got[p]) or {}).get("severity") != want]
        if bad:
            raise SystemExit(f"REFUSED: harm severity judge ({self.model}) calibration failed (want, reply): {bad}")
        self.served = {"id": self.model, "actual_model": self.last_model, "rubric": self.version,
                       "rubric_sha256": RUBRIC_SHA256, "usd_cap": self.usd_cap, "usage_log": str(self.usage_log)}
        logger.info("harm severity judge healthy: %s", self.served)
        return self.served


# ----------------------------------------------------------------------------- reward transform (pure)
def eligible(led: dict) -> bool:
    v = led.get("v4h") or {}
    return (not led["components"]["degenerate"]["applicable"] and bool(v.get("acute")) and bool(v.get("contraindicated"))
            and v.get("backend") == "judge")


def apply(led: dict, sev: Optional[dict], mode: str = "on") -> dict:
    """The ledger with the severity cap (a new object when eligible; the same object otherwise)."""
    if mode not in MODES:
        raise ValueError(f"harm_severity_cap must be one of {MODES}, got {mode!r}")
    if mode == "off" or not eligible(led) or (led["components"].get(COMPONENT) or {}).get("applicable"):
        return led
    severity = (sev or {}).get("severity") or "not_evaluated"
    out = copy.deepcopy(led)
    base = out["total"]
    delta = round(min(0.0, CAP - base), 10) if severity == "critical" else 0.0
    capped = delta < 0
    out["components"][COMPONENT] = {"applicable": True, "value": delta,
                                    "status": f"active:{severity}" + (":capped" if capped else "")}
    out["total"] = round(base + delta, 10)
    out["applicable_components"] = sorted(k for k, c in out["components"].items() if c["applicable"])
    out["flags"] = {**out["flags"], "harm_severity_cap": mode}
    out["harm_severity"] = {"severity": severity, "capped": capped, "base_total": base, "cap": CAP,
                            "harmful_quote": (sev or {}).get("harmful_quote", ""), "version": "tkp_severity_v2"}
    return out


# ----------------------------------------------------------------------------- hooks
def _log(path: Optional[Path], recs: list[dict]) -> None:
    if path is None or not recs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))


def install(mode: str, *, judge: Any = None, log_path: Optional[Path] = None, budget_s: float = STEP_BUDGET_S) -> None:
    """In-process, AFTER main_v4haa's warning credit (main_v4h.install time): hazard_judge.prefetch -> + severity
    calls on judge-contraindicated episodes; reward_v4hg.score_episode -> apply (outermost). ``off``: nothing."""
    if mode not in MODES:
        raise ValueError(f"harm_severity_cap must be one of {MODES}, got {mode!r}")
    if mode == "off" or _STATE:
        return
    if judge is None:
        raise ValueError("harm_severity_cap on needs a SeverityJudge")
    from trainer_v2_v4 import hazard_rubric as HR
    from trainer_v2_v4 import patch_h as PH
    from trainer_v2_v4 import reward_v4hg as RHG
    S = s2()
    inner_prefetch, inner_score = HJ.prefetch, RHG.score_episode

    def prefetch(jobs: list, j: Any, budget: float = HJ.STEP_BUDGET_S) -> dict:
        st = inner_prefetch(jobs, j, budget)
        pairs = []
        for ep, _ in jobs:
            ep.harm_severity = None
            hj = getattr(ep, "hazard_judge", None)
            if hj and hj.get("contraindicated"):
                hz = PH.hz_input(ep.item, None, "judge")
                pairs.append((ep, item_text(hz["ra"], ep.item.row, hj, HR.answer_text(ep.stage2_text))))
        uniq = list(dict.fromkeys(t for _, t in pairs))
        replies, sst = judge.run(uniq, budget_s) if uniq else ({}, {})
        recs = []
        for ep, t in pairs:
            ep.harm_severity = S.finalize(S.parse_reply(replies.get(t)), HR.answer_text(ep.stage2_text), "luna")
            recs.append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": getattr(PH.H, "step", None),
                         "row_id": ep.item.row_id, **{k: ep.harm_severity.get(k) for k in
                                                      ("severity", "harmful_quote", "reason", "quote_ok")}})
        _log(log_path, recs)
        n_crit = sum(r["severity"] == "critical" for r in recs)
        return {**st, "severity_calls": len(uniq), "severity_critical": n_crit,
                "severity_unjudged": sum(r["severity"] == "unjudged" for r in recs),
                "severity_cache_hits": sst.get("cache_hits", 0)}

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        return apply(inner_score(ep, cfg, hz), getattr(ep, "harm_severity", None), mode)
    score_episode._hsc_orig, prefetch._hsc_orig = inner_score, inner_prefetch
    _STATE.update(prefetch=inner_prefetch, score=inner_score, judge=judge)
    HJ.prefetch, RHG.score_episode = prefetch, score_episode


def uninstall() -> None:
    from trainer_v2_v4 import reward_v4hg as RHG
    if not _STATE:
        return
    if getattr(HJ.prefetch, "_hsc_orig", None) is _STATE["prefetch"]:
        HJ.prefetch = _STATE["prefetch"]
    if getattr(RHG.score_episode, "_hsc_orig", None) is _STATE["score"]:
        RHG.score_episode = _STATE["score"]
    _STATE.clear()


def hook_main_v4h(mode: str, usd_cap: float = USD_CAP, *, judge_factory: Any = None) -> None:
    """Wrap main_v4h.install (it runs after main_v4haa installed the warning credit and builds the hazard judge):
    the severity judge is built + calibrated there, then install(). ``off``: nothing."""
    if mode == "off":
        return
    from trainer_v2_v4 import main_v4h as MH
    if getattr(MH.install, "_hsc_hooked", False):
        return
    orig = MH.install

    def wrapped(h: Any, rest: list[str]) -> dict:
        info = orig(h, rest)
        if not (h.hazard_backend == "judge" and h.hazard_judge_provider == "api"):
            raise SystemExit("REFUSED: --harm-severity-cap on needs --hazard-backend judge --hazard-judge-provider api")
        out = Path(MH._value(rest, "--out") or ".")
        jdir = out.parent / (out.name + ".judge")
        rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        sfx = "" if rank == 0 else f".rank{rank}"
        make = judge_factory or SeverityJudge
        judge = make(jdir / f"usage_harm_severity{sfx}.jsonl", usd_cap=usd_cap / world,
                     concurrency=h.hazard_judge_concurrency, timeout_s=h.hazard_judge_timeout,
                     cache_path=jdir / f"harm_severity_cache{sfx}.jsonl")
        judge.health_check()                                       # 2 paid calls; refuses on failure
        install(mode, judge=judge, log_path=jdir / f"harm_severity{sfx}.jsonl")
        return {**info, "harm_severity_cap": {"mode": mode, "usd_cap_rank": usd_cap / world}}
    wrapped._hsc_hooked = True
    MH.install = wrapped


__all__ = ["MODES", "CAP", "COMPONENT", "PURPOSE", "MODEL", "USD_CAP", "SHA256", "RUBRIC_SHA256", "s2", "sev_item",
           "item_text", "SeverityJudge", "eligible", "apply", "install", "uninstall", "hook_main_v4h"]
