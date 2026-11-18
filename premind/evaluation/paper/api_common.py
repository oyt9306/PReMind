"""Spend ledger + paid-step gates shared by the fix2 API inference and judge launchers. No API client is imported here.

Every paid step must be called with --execute-paid --ledger <SPEND_LEDGER.jsonl> --usd-cap <X>; otherwise it refuses.
The ledger records a reservation (the step cap) before the paid subprocess starts and the actual spend (summed from the
tool's own ledger) when it ends. Committed spend per provider = closed actuals + caps of open reservations; a new step is
refused if committed + cap would exceed the owner's provider cap (OpenAI $30: Luna/Sol inference + Luna judges;
Anthropic $10: Opus). API keys are read only by the frozen tools from the environment and are never printed or logged.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix2_common as F  # noqa: E402

KEY_VAR = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# default per-step caps (USD); sum per provider stays within PROVIDER_CAP_USD
STEP_CAPS = {"luna_infer_conflict": 1.0, "luna_infer_main": 1.0, "luna_judges_conflict": 4.0, "luna_judges_main": 12.0,
             "sol_infer_conflict": 4.0, "sol_infer_main": 6.0, "opus_infer_conflict": 4.0, "opus_infer_main": 6.0}
CHECKPOINT = "OWNER_CHECKPOINT_AFTER_LUNA.json"


class Refused(SystemExit):
    pass


def default_ledger(materialize: bool) -> Path:
    return F.out_root(materialize) / "fix2_spend_v1" / "SPEND_LEDGER.jsonl"


def require_paid(args) -> None:
    if not getattr(args, "execute_paid", False) or not getattr(args, "ledger", None) or getattr(args, "usd_cap", None) is None:
        raise Refused("REFUSED: paid step needs --execute-paid, --ledger <file> and --usd-cap <USD>")
    if args.usd_cap <= 0:
        raise Refused("REFUSED: --usd-cap must be > 0")


def events(ledger: Path) -> list[dict]:
    return F.jl(ledger) if Path(ledger).is_file() else []


def committed(ledger: Path) -> dict[str, float]:
    open_, out = {}, {p: 0.0 for p in F.PROVIDER_CAP_USD}
    for e in events(ledger):
        if e["event"] == "reserve":
            open_[e["step"]] = e
        elif e["event"] == "close" and e["step"] in open_:
            r = open_.pop(e["step"]);out[r["provider"]] += float(e["actual_usd"])
    for r in open_.values():
        out[r["provider"]] += float(r["cap_usd"])
    return out


def _append(ledger: Path, rec: dict) -> None:
    ledger = Path(ledger);ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(rec | {"time": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")


def group_committed(ledger: Path, group: str) -> float:
    open_, tot = {}, 0.0
    for e in events(ledger):
        if e["event"] == "reserve" and e.get("group") == group:
            open_[e["step"]] = e
        elif e["event"] == "close" and e["step"] in open_:
            open_.pop(e["step"]);tot += float(e["actual_usd"])
    return tot + sum(float(r["cap_usd"]) for r in open_.values())


def reserve(ledger: Path, step: str, provider: str, cap: float, group: str | None = None) -> None:
    if group is not None:
        g = group_committed(ledger, group)
        if g + cap > STEP_CAPS[group] + 1e-9:
            raise Refused(f"REFUSED: group {group} committed ${g:.3f} + cap ${cap:.3f} > group cap ${STEP_CAPS[group]:.2f}")
    if provider not in F.PROVIDER_CAP_USD:
        raise Refused(f"REFUSED: unknown provider {provider}")
    if any(e["event"] == "reserve" and e["step"] == step for e in events(ledger)):
        raise Refused(f"REFUSED: step {step} already reserved in {ledger} (re-runs need a new step id and owner approval)")
    c = committed(ledger)[provider]
    if c + cap > F.PROVIDER_CAP_USD[provider] + 1e-9:
        raise Refused(f"REFUSED: {provider} committed ${c:.2f} + cap ${cap:.2f} > owner cap ${F.PROVIDER_CAP_USD[provider]:.2f}")
    _append(ledger, {"event": "reserve", "step": step, "provider": provider, "cap_usd": cap, "group": group})


def close(ledger: Path, step: str, tool_ledgers: list[Path]) -> float:
    usd = sum(tool_ledger_usd(p) for p in tool_ledgers)
    _append(ledger, {"event": "close", "step": step, "actual_usd": round(usd, 6), "tool_ledgers": [str(p) for p in tool_ledgers]})
    return usd


def tool_ledger_usd(p: Path) -> float:
    tot = 0.0
    if not Path(p).is_file():
        return 0.0
    for e in F.jl(p):
        for src in (e, e.get("extra") or {}):
            for k in ("usd", "est_cost_usd", "cost_usd"):
                if isinstance(src.get(k), (int, float)):
                    tot += float(src[k]);break
            else:
                continue
            break
    return tot


def require_key(provider: str) -> None:
    if not os.environ.get(KEY_VAR[provider]):
        raise Refused(f"REFUSED: {KEY_VAR[provider]} not set in the environment (value is never printed)")


def require_owner_checkpoint(ledger: Path) -> None:
    p = Path(ledger).parent / CHECKPOINT
    ok = p.is_file() and json.loads(p.read_text()).get("approved") is True
    if not ok:
        raise Refused(f"REFUSED: Opus/Sol run only after the owner has seen the Luna results: {p} with "
                      '{"approved": true, "by": "owner", "note": ...} (written by the orchestrator after the owner OK)')


def status(ledger: Path) -> str:
    c = committed(ledger)
    return " | ".join(f"{p}: committed ${c[p]:.3f} / cap ${F.PROVIDER_CAP_USD[p]:.2f}" for p in F.PROVIDER_CAP_USD)
