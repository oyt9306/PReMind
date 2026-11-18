#!/usr/bin/env python3
"""PReMind v4 training-v2 entry point = main_v4 (P3 recipe) or main_v4b (P1 recipe, ``--base v4b``) + the v4h options
(patch_h.py), GPU jobs launched by the orchestrator only (CPU: ``--print-config``).

  --reward-preset v4wh_g | v4wh | v4wh_s   v4wh_g (v2 default) = v4wh + shortcut guards G1-G5 (reward_v4hg.py);
                                  v4wh = v4wf + hazard appropriateness (reward_v4h.py); v4wh_s adds the structure term C
  --hazard-backend cue|judge      deterministic cue path (default) or the open-ended judge (+ cue fallback)
  --hazard-judge-provider api|local   api (default, owner 2026-10-14): OpenAI gpt-6-luna@medium via hazard_judge_api
                                  (no GPU; paid, capped); local: the vLLM judge at --hazard-judge-url
  --hazard-judge-url URL          local only: http://127.0.0.1:<port>/v1
  --hazard-judge-model SPEC       api: pinned model spec (default gpt-6-luna@medium); local: served name (vef-judge)
  --hazard-usd-cap USD / --hazard-token-cap N   api run caps (default $5 / 10M tokens; split evenly over DDP ranks)
  --hazard-judge-concurrency N / --hazard-judge-timeout S   api thread pool (8) / per-call timeout (30 s)
  --hazard-judge-cache PATH       reply cache (default OUT/hazard_judge_cache[.rank<r>].jsonl)
  api ledger: OUT.judge/usage_hazard_judge.jsonl (rank 0; rank r: usage_hazard_judge.rank<r>.jsonl)
  --hazard-key-actions PATH       KEY_ACTIONS jsonl by row id (default: v4_build/train_diversify/TRAIN_KEY_ACTIONS_v9
                                  .jsonl if present, else trainer_v2_v4/out/KEY_ACTIONS_FALLBACK_v9.jsonl)
  --hazard-monitor                hazard/* monitor with any preset (reward unchanged; on by itself under v4wh)
  --log-rollout-texts             OUT/rollout_texts.jsonl (acute rollouts + --rollout-text-frac of the others)
  --acute-absent keep|mask        pilots A/B (v2_headroom REPORT #1): mask drops the acute (image verdict or key-action
                                  row) AND context_condition gt_absent rows from the TRAIN split only (dev unchanged);
                                  keep (default) = unchanged. Recorded in the v4h pin / checkpoint args only when mask.
  --reward-preset v4wh_g_nw       pilot B: v4wh_g with missed_warning x2 (-0.30) on non-acute unsafe rows (reward_v4hg)
Every option is OFF by default; with all of them off this is exactly main_v4 / main_v4b (patch_h is not installed).
The v4h values are added to the checkpoint args (``v4h_*``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import key_actions_v4h as KA  # noqa: E402
from trainer_v2_v4 import reward_v4h as RH  # noqa: E402
from trainer_v2_v4 import reward_v4hg as RHG  # noqa: E402

logger = logging.getLogger(__name__)
H_KEYS = ("base", "hazard_backend", "hazard_judge_provider", "hazard_judge_url", "hazard_judge_model",
          "hazard_judge_cache", "hazard_usd_cap", "hazard_token_cap", "hazard_judge_concurrency",
          "hazard_judge_timeout", "hazard_key_actions", "hazard_monitor", "log_rollout_texts", "rollout_text_frac",
          "prereg_v4h")


def release_pin_presets() -> None:
    """In-process: the release pin (trainer_v2.pins, written in a plain process) lists the base reward presets only;
    the v4 patches register extra presets (v4wf, v4wh_g, ...), which made pins.verify refuse 'reward_presets'. Verify
    exactly the presets the release pin lists (any change to them still refuses); the v4h presets are pinned by the
    v4h run pin (code sha256 + guard + preset name)."""
    from trainer_v2 import pins as PN
    if getattr(PN.verify, "_v4h_presets", False):
        return
    orig_verify, orig_table = PN.verify, PN.presets_table

    def verify(path=PN.PREREG, *, rows=None):
        names = set(json.loads(Path(path).read_text())["reward"]["presets"])
        PN.presets_table = lambda: {k: v for k, v in orig_table().items() if k in names}
        try:
            return orig_verify(path, rows=rows)
        finally:
            PN.presets_table = orig_table
    verify._v4h_presets = True
    PN.verify = verify


def no_abbrev() -> None:
    """In-process: main_v4.v4_parser / main_v4b.b_parser with allow_abbrev=False. With abbreviations, their
    ``--prereg-v4`` / ``--prereg-v4b`` options swallow main_run's ``--prereg <release pin>`` (unique prefix), so the
    release pin never reached main_run and, without --prereg-v4, was verified as a v4 pin (KeyError 'constants')."""
    from trainer_v2_v4 import main_v4 as MV4
    from trainer_v2_v4 import main_v4b as MB
    for mod, name in ((MV4, "v4_parser"), (MB, "b_parser")):
        fn = getattr(mod, name)
        if getattr(fn, "_no_abbrev", False):
            continue

        def wrapped(_fn=fn):
            ap = _fn()
            ap.allow_abbrev = False
            return ap
        wrapped._no_abbrev = True
        setattr(mod, name, wrapped)


def h_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: main_run's "--r 0.1" must never be read as a prefix of --rollout-text-frac (REPORT.md §6)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--base", default="v4", choices=("v4", "v4b"))
    ap.add_argument("--hazard-backend", default="cue", choices=("cue", "judge"))
    ap.add_argument("--hazard-judge-provider", default="api", choices=("api", "local"))
    ap.add_argument("--hazard-judge-url", default=None)
    ap.add_argument("--hazard-judge-model", default=None)
    ap.add_argument("--hazard-usd-cap", type=float, default=5.0)
    ap.add_argument("--hazard-token-cap", type=int, default=10_000_000)
    ap.add_argument("--hazard-judge-concurrency", type=int, default=8)
    ap.add_argument("--hazard-judge-timeout", type=float, default=30.0)
    ap.add_argument("--hazard-judge-cache", default=None)
    ap.add_argument("--hazard-key-actions", default=None)
    ap.add_argument("--hazard-monitor", action="store_true")
    ap.add_argument("--log-rollout-texts", action="store_true")
    ap.add_argument("--rollout-text-frac", type=float, default=0.05)
    ap.add_argument("--acute-absent", default="keep", choices=("keep", "mask"))
    ap.add_argument("--prereg-v4h", default=None, help="v4h run pin (pins_v4h.py write); verified before anything runs")
    return ap


def _value(rest: list[str], flag: str) -> Optional[str]:
    for i, x in enumerate(rest):
        if x == flag and i + 1 < len(rest):
            return rest[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return None


def key_actions_path(h: argparse.Namespace) -> Path:
    if h.hazard_key_actions:
        return Path(h.hazard_key_actions)
    return KA.DIVERSIFY_FILE if KA.DIVERSIFY_FILE.is_file() else KA.FALLBACK_FILE


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str], bool]:
    """(v4h options, the main_v4 / main_v4b argv, whether patch_h is needed). Refuses inconsistent options."""
    RHG.register()
    h, rest = h_parser().parse_known_args(argv)
    preset = _value(rest, "--reward-preset")
    active = RHG.is_hazard_preset(preset or "")
    if h.hazard_backend == "judge" and not active:
        raise SystemExit("REFUSED: --hazard-backend judge needs a v4wh preset")
    local = h.hazard_backend == "judge" and h.hazard_judge_provider == "local"
    if local and not h.hazard_judge_url:
        raise SystemExit("REFUSED: --hazard-judge-provider local needs --hazard-judge-url")
    if h.hazard_judge_url and not local:
        raise SystemExit("REFUSED: --hazard-judge-url is only used with --hazard-backend judge "
                         "--hazard-judge-provider local")
    if h.hazard_backend == "judge" and h.hazard_judge_provider == "api":
        from eval_v1 import api_providers as AP
        h.hazard_judge_model = h.hazard_judge_model or "gpt-6-luna@medium"
        try:
            AP.profile(h.hazard_judge_model)
        except Exception as exc:  # noqa: BLE001 - unknown model / effort -> refuse before anything runs
            raise SystemExit(f"REFUSED: hazard judge model {h.hazard_judge_model!r}: {exc}") from exc
        if not (h.hazard_usd_cap > 0 and h.hazard_token_cap > 0 and h.hazard_judge_concurrency >= 1):
            raise SystemExit("REFUSED: API judge caps and concurrency must be positive")
    if not 0.0 <= h.rollout_text_frac <= 1.0:
        raise SystemExit("REFUSED: --rollout-text-frac must be in [0, 1]")
    if h.acute_absent == "mask" and float(_value(rest, "--monitor-frac") or 0.08) > 0:   # 0.08 = main_v4 default
        raise SystemExit("REFUSED: --acute-absent mask needs --monitor-frac 0 (the monitor slice is the in-loop dev "
                         "set drawn from the train file; the mask must not touch it)")
    need = active or h.hazard_monitor or h.log_rollout_texts
    return h, rest, need


def install(h: argparse.Namespace, rest: list[str]) -> dict:
    from trainer_v2_v4 import patch_h as PH
    kpath = key_actions_path(h)
    if not kpath.is_file():
        raise SystemExit(f"REFUSED: key-action file missing: {kpath} (build: python -I -m trainer_v2_v4.key_actions_v4h "
                         "--release $REL)")
    rel = _value(rest, "--release")
    ka = KA.load(kpath, Path(rel) if rel else None, _value(rest, "--mode") or "gt_retrieved_drop")
    if not ka:
        raise SystemExit(f"REFUSED: no key-action row of {kpath} maps to a row of --release {rel}")
    judge = None
    out = Path(_value(rest, "--out") or ".")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    sfx = "" if rank == 0 else f".rank{rank}"
    if h.hazard_backend == "judge" and h.hazard_judge_provider == "local":
        from trainer_v2_v4 import hazard_judge as HJ
        judge = HJ.HazardJudge(h.hazard_judge_url, h.hazard_judge_model,
                               Path(h.hazard_judge_cache or out / "hazard_judge_cache.jsonl"))
        judge.health_check(RHG.calibration(_value(rest, "--reward-preset") or ""))
    elif h.hazard_backend == "judge":
        from trainer_v2_v4 import hazard_judge_api as HA
        jdir = out.parent / (out.name + ".judge")   # outside OUT: main_run refuses a non-empty fresh OUT
        judge = HA.ApiHazardJudge(jdir / f"usage_hazard_judge{sfx}.jsonl", model=h.hazard_judge_model,
                                  usd_cap=h.hazard_usd_cap / world, token_cap=h.hazard_token_cap // world,
                                  concurrency=h.hazard_judge_concurrency, timeout_s=h.hazard_judge_timeout,
                                  cache_path=Path(h.hazard_judge_cache + sfx) if h.hazard_judge_cache
                                  else jdir / f"hazard_judge_cache{sfx}.jsonl")
        judge.health_check(RHG.calibration(_value(rest, "--reward-preset") or ""))   # 3 paid calls; refuses on failure
    from trainer_v2_v4 import hazard_monitor as HMN
    gold_med = HMN.gold_len_median(Path(rel), _value(rest, "--mode") or "gt_retrieved_drop") if rel else None
    PH.install(key_actions=ka, backend=h.hazard_backend, judge=judge, gold_len_median=gold_med,
               monitor=h.hazard_monitor or RHG.is_hazard_preset(_value(rest, "--reward-preset") or ""),
               log_texts=h.log_rollout_texts, text_frac=h.rollout_text_frac, register_presets=False)
    return {"key_actions": str(kpath), "n_rows": len(ka), "shared_sha256": KA.shared_sha256()}


def h_keys(h: argparse.Namespace) -> tuple[str, ...]:
    """H_KEYS + acute_absent when it is not the default (keep: print-config / checkpoint args exactly as before)."""
    return H_KEYS + (("acute_absent",) if getattr(h, "acute_absent", "keep") != "keep" else ())


def is_acute_absent(item, ka_ids: set) -> bool:
    from eval_v1 import acute_rule_v4 as AR                # read-only
    return item.row.get("condition") == "gt_absent" and (AR.is_acute_row(item.row) or item.row_id in ka_ids)


def install_acute_absent_mask(h: argparse.Namespace, rest: list[str]) -> set:
    """--acute-absent mask: eval_v1.release_data.load_release drops acute AND gt_absent items of split "train" (the
    training pool, hence the absent-share sampler); every other split (the in-loop dev eval) is returned unchanged.
    Acute = image verdict (acute_rule_v4.is_acute_row) or a row of the key-action file. In-process only."""
    from eval_v1 import release_data as D
    rel, mode = _value(rest, "--release"), _value(rest, "--mode") or "gt_retrieved_drop"
    kp = key_actions_path(h)
    ka_ids = set(KA.load(kp, Path(rel) if rel else None, mode)) if kp.is_file() else set()
    orig = D.load_release

    def load_release(release, split, mode, **kw):
        items, man = orig(release, split, mode, **kw)
        if split != "train":
            return items, man
        drop = {it.row_id for it in items if is_acute_absent(it, ka_ids)}
        kept = [it for it in items if it.row_id not in drop]
        logger.info("ACUTE_ABSENT=mask: excluded %d acute & gt_absent train items (%d rows) of %d; %d kept",
                    len(items) - len(kept), len(drop), len(items), len(kept))
        return kept, {**man, "n_items": len(kept), "acute_absent_mask": {"n_items_excluded": len(items) - len(kept),
                                                                         "row_ids": sorted(drop)}}
    D.load_release = load_release
    return ka_ids


def _store_args(h: argparse.Namespace, info: dict) -> None:
    """The v4h values join the checkpoint args (resume then refuses a different v4h configuration)."""
    from trainer_v2_v4 import main_v4 as MV4
    orig = MV4.parse

    def parse(argv=None):
        args, pc = orig(argv)
        for k in h_keys(h):
            setattr(args, f"v4h_{k}", getattr(h, k))
        args.v4h_key_actions_rows = info.get("n_rows")
        return args, pc
    MV4.parse = parse


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--"]:                      # torchrun commands put "--" before the script arguments
        argv = argv[1:]
    no_abbrev()
    release_pin_presets()
    h, rest, need = resolve(argv)
    if h.prereg_v4h:
        from trainer_v2_v4 import pins_v4h as PV
        PV.verify(Path(h.prereg_v4h), h, rest)
        logger.info("v4h pin verified: %s", h.prereg_v4h)
    elif need:
        logger.warning("v4h options active without --prereg-v4h (allowed for smoke / CPU only; the v2 launcher "
                       "requires a pin)")
    if "--print-config" in rest:
        kp = key_actions_path(h)
        info = {"key_actions": str(kp), "key_actions_sha256": hashlib.sha256(kp.read_bytes()).hexdigest()
                if kp.is_file() else None, "shared_sha256": KA.shared_sha256(), "guard_sha256": RHG.GUARD_SHA256,
                "rubric_sha256": RHG.HR.RUBRIC_SHA256}
        print(json.dumps({"v4h": {k: getattr(h, k) for k in h_keys(h)}, "patch_h": need, **info,
                          "reward": {"alpha": RH.ALPHA, "beta": RH.BETA, "omega": RH.OMEGA, "gamma": RH.GAMMA,
                                     "structure_w": RH.STRUCT_W}}, indent=1, default=str))
    else:
        if h.acute_absent == "mask":
            install_acute_absent_mask(h, rest)
        if need or h.acute_absent == "mask":
            _store_args(h, install(h, rest) if need else {})
    if h.base == "v4b":
        from trainer_v2_v4 import main_v4b as MB
        return MB.main(rest)
    from trainer_v2_v4 import main_v4 as MV4
    return MV4.main(rest)


if __name__ == "__main__":
    sys.exit(main())
