"""Stronger false-alarm penalty for the v14 fallback run (owner-approved plan 2026-10-15: restart if the step-50 dev
safe risk_acc < .6). NEW file: reward_v5 / reward_v4hg / warning_credit / absent_visible_risk are untouched.

``--false-alarm-pen fa25`` = the v4wh_g preset with PEN["false_alarm"] = -0.25 instead of -0.15 (the "v4wh_g+fa25"
reward); ``off`` (default) installs nothing.
Mechanism: install() sets reward_v5.PEN["false_alarm"] in-process (the one dict every false-alarm site reads at call
time: reward_v5's tag rule and absent_visible_risk's re-scoring), so the penalty applies identically to
  * the tag-based false alarm (<risk> unsafe on a gold-safe row), and
  * the warning-credit judge false alarm: warning_credit scores a copy whose <risk> tag is the effective risk
    (tag unsafe OR warns specifically), i.e. the same reward_v5 component.
reward_v4hg.score_episode is wrapped (flag only, values untouched) so every ledger records
``flags.false_alarm_pen = {"mode", "value"}``; install refuses if PEN["false_alarm"] is not the documented -0.15.
The in-loop dev eval reports risk_acc / false alarms, which do not depend on PEN.
"""
from __future__ import annotations

from typing import Any

MODES = ("off", "fa25")
VALUE = {"fa25": -0.25}
BASE = -0.15
_STATE: dict = {}


def install(mode: str) -> None:
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import reward_v4hg as RHG
    if mode not in MODES:
        raise ValueError(f"false_alarm_pen must be one of {MODES}, got {mode!r}")
    if mode == "off" or _STATE:
        return
    if abs(R5.PEN["false_alarm"] - BASE) > 1e-12:
        raise RuntimeError(f"false_alarm_pen: PEN['false_alarm'] is {R5.PEN['false_alarm']}, expected {BASE}")
    orig = RHG.score_episode
    tag = {"mode": mode, "value": VALUE[mode]}

    def score_episode(ep: Any, cfg: Any, hz: Any) -> dict:
        led = orig(ep, cfg, hz)
        return {**led, "flags": {**(led.get("flags") or {}), "false_alarm_pen": tag}}
    score_episode._fa_orig = orig
    _STATE.update(orig=orig, mode=mode)
    R5.PEN["false_alarm"] = VALUE[mode]
    RHG.score_episode = score_episode


def uninstall() -> None:
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import reward_v4hg as RHG
    if not _STATE:
        return
    R5.PEN["false_alarm"] = BASE
    if getattr(RHG.score_episode, "_fa_orig", None) is _STATE["orig"]:
        RHG.score_episode = _STATE["orig"]
    _STATE.clear()


__all__ = ["MODES", "VALUE", "BASE", "install", "uninstall"]
