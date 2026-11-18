"""In-process hooks of the judge-free CUR22B ablations (CUR22_ABLATIONS.md; prepared 2026-10-16). NEW file: every
module it touches is wrapped at run time, none is edited.

  warning credit "cue"   = warning_credit.py (``--warning-credit judge``) with the WarningRewardHook backend "cue"
                           (frozen wd_cue_v2, CPU, deterministic; the judge mode's own fallback) and NO judge client.
                           Installed at the exact place main_v4haa installs ``--warning-credit judge``: right after
                           absent_visible_risk.install, so it is still the OUTERMOST reward_v4hg.score_episode wrapper.
  hazard first "off"     = reward_v5's hazard_first term (R_early: acute +0.08 / safe-row leading alarm -0.05) off,
                           exactly as the old ``hf-off`` preset (V5Flags.hazard_first False), for every reward_v5 score
                           in the process (reward_v5._score is wrapped; the ledger flags show hazard_first false).
  stop at step N         = eval_v1.stop_rule.StopRule whose ``stopped`` turns True after step N (``--steps`` keeps the
                           curriculum horizon); ddp_main then checkpoints (if not yet) and leaves the loop.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
WARNING_MODES = ("off", "cue")
HF_MODES = ("on", "off")
HORIZON_REASON = "ablation_horizon"
_STATE: dict = {}


class NoApiJudge:
    """Stand-in judge handed to warning_credit.install so it never builds the paid WarningJudge; never called."""

    def run(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("judge-free ablation: the warning judge must never be called")

    def health_check(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("judge-free ablation: the warning judge must never be called")


def install_warning_cue_now(release: Path) -> Any:
    """warning_credit hooks with the cue backend (no client); returns warning_credit.W."""
    from trainer_v2_v4 import warning_credit as WC
    from trainer_v2_v4 import warning_reward as WR
    WC.install("judge", release=Path(release), out=Path("."), judge=NoApiJudge())   # injected: no make_judge
    WC.W.hook = WR.WarningRewardHook(WC.W.hook.refs, backend="cue")
    WC.W.judge = None
    logger.info("warning credit installed with the cue backend (no judge): refs=%d", len(WC.W.hook.refs))
    return WC.W


def install_warning_cue(mode: str, release: Optional[str]) -> None:
    """``cue``: install the cue warning credit right after absent_visible_risk.install (main_v4haa's WC slot)."""
    if mode not in WARNING_MODES:
        raise ValueError(f"warning credit must be one of {WARNING_MODES}, got {mode!r}")
    if mode == "off" or "avr_install" in _STATE:
        return
    if not release:
        raise SystemExit("REFUSED: the cue warning credit needs --release")
    from trainer_v2_v4 import absent_visible_risk as AVR
    orig = AVR.install
    _STATE["avr_install"] = orig

    def install(m: str) -> None:
        orig(m)
        install_warning_cue_now(Path(release))
    AVR.install = install


def install_hazard_first(mode: str) -> None:
    """``off``: reward_v5 scores with hazard_first False (= preset hf-off) everywhere in this process."""
    if mode not in HF_MODES:
        raise ValueError(f"hazard first must be one of {HF_MODES}, got {mode!r}")
    if mode == "on" or "score" in _STATE:
        return
    from trainer_v2 import reward_v5 as R5
    orig = R5._score
    _STATE["score"] = orig

    def _score(*a: Any, flags: Any, **kw: Any) -> dict:
        return orig(*a, flags=dataclasses.replace(flags, hazard_first=False), **kw)
    R5._score = _score


def horizon_stop_rule(base: type, stop_at: int) -> type:
    class HorizonStopRule(base):
        """StopRule + a hard stop after step ``stop_at`` (independent of --hard-stop / the over-abstain rule)."""

        def update(self, step: int, monitor_rec: Optional[dict], monitor_stopped: bool) -> dict:
            out = super().update(step, monitor_rec, monitor_stopped)
            if step >= stop_at and not self.stopped:
                self.stopped, self.step, self.horizon = True, step, True
                logger.warning("ablation horizon: stop after step %d (of --steps)", step)
            out["stop/horizon"] = int(bool(getattr(self, "horizon", False)))
            return out

        def summary(self) -> dict:
            s = super().summary()
            if getattr(self, "horizon", False):
                s = {**s, "stop/reason": HORIZON_REASON, "stop/horizon_step": stop_at}
            return s
    return HorizonStopRule


def install_stop_at(stop_at: int) -> None:
    if stop_at <= 0 or "stop_rule" in _STATE:
        return
    from eval_v1 import stop_rule as SR
    _STATE["stop_rule"] = SR.StopRule
    SR.StopRule = horizon_stop_rule(SR.StopRule, stop_at)


def store_args(values: dict) -> None:
    """Non-default values join the checkpoint args (``v4h_<key>``), like main_v4haa._store_args."""
    if not values:
        return
    from trainer_v2_v4 import main_v4 as MV4
    orig = MV4.parse

    def parse(argv=None):
        args, pc = orig(argv)
        for k, v in values.items():
            setattr(args, f"v4h_{k}", v)
        return args, pc
    MV4.parse = parse


def uninstall() -> None:
    from eval_v1 import stop_rule as SR
    from trainer_v2 import reward_v5 as R5
    from trainer_v2_v4 import absent_visible_risk as AVR
    if "avr_install" in _STATE:
        AVR.install = _STATE.pop("avr_install")
    if "score" in _STATE:
        R5._score = _STATE.pop("score")
    if "stop_rule" in _STATE:
        SR.StopRule = _STATE.pop("stop_rule")


__all__ = ["WARNING_MODES", "HF_MODES", "HORIZON_REASON", "NoApiJudge", "install_warning_cue",
           "install_warning_cue_now", "install_hazard_first", "horizon_stop_rule", "install_stop_at", "store_args",
           "uninstall"]
