"""Pin of a CUR23R recall-phase run (main_v4hcur23r): the full CUR23 block (pins_v4hcur23.block, frozen, imported
read-only: v4h + aa + cur + amb + fa + h23, rebuilt from the same arguments) + ``r23``:
  recall    call-decision mode + weights (frozen v52ws_ag), crowd-out guard mode + rule, r
  model     size / id / revision + memory knobs (gen batch, micro batch, LoRA r, prompts per rank, world)
  ws        the warm-start LoRA (path, adapter sha256, COMPLETE text, source run)
  stage_a   the source Stage-A CUR23 pin (OUT.v4hcur23_pin.json or OUT.v4hcur23_9b_pin.json of the INIT_LORA run):
            path + sha256 + the compared fields; the pin is REFUSED unless the recall run uses the same reward stack
            (aa, amb, fa, h23 blocks), the same curriculum rule / data / strata (cur minus the step-dependent
            schedule), the same v4h judge / guard / key-action pins, the same constants (minus arm / steps) and model
  code      sha256 of the new files
prereg PREREG_TRAINER_V2_V4HCUR23R, base_prereg PREREG_TRAINER_V2_V4HCUR23.
World size = PIN_WORLD, else WORLD_SIZE (torchrun), else 1: export PIN_WORLD=<ranks> for the pin write AND the run
(the CUR23 9B launch once failed because the run-time check read the PIN_WORLD default 1).
  python -I -m trainer_v2_v4.pins_v4hcur23r write --out PIN.json -- <main_v4hcur23r args>   (write once)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import pins_v4hcur21 as P21  # noqa: E402
from trainer_v2_v4 import pins_v4hcur23 as P23  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR23R"
CODE = ("recall_decision.py", "main_v4hcur23r.py", "pins_v4hcur23r.py", "scripts/v4hcur23r_plan.sh")
BLOCKS = P23.BLOCKS + ("r23",)
STAGE_A_PINS = (("v4hcur23_pin.json", "PREREG_TRAINER_V2_V4HCUR23"), ("v4hcur23_9b_pin.json",
                                                                       "PREREG_TRAINER_V2_V4HCUR23_9B"))
SAME_BLOCKS = ("aa", "amb", "fa", "h23")
SAME_CUR = ("code_sha256", "curriculum", "seed", "requeue", "min_gap_steps", "replay", "runtime", "data")
SAME_QUOTAS = ("absent_share", "safe_min_frac", "replay_budget_requested", "counts", "acute_heavy",
               "acute_heavy_frac")
SAME_V4H = ("code_sha256", "guard_sha256", "rubric_sha256", "eval_rubric_sha256", "shared_sha256", "key_actions_sha256",
            "release", "config")
SAME_CONSTANTS = ("reward_preset", "prompts_per_step", "kl_beta", "kl_estimator", "aux_ratio", "aux_methods",
                  "aux_prompts_per_step", "monitor_frac", "monitor_seed")


def world() -> int:
    return int(os.environ.get("PIN_WORLD") or os.environ.get("WORLD_SIZE") or "1")


def _value(argv: list[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    from trainer_v2_v4 import main_v4h as MH
    v = MH._value(argv, flag)
    return default if v is None else v


def model_block(run: list[str]) -> dict:
    from trainer_v2 import paths_v2 as P
    size = _value(run, "--model", "4b")
    mid, rev = P.MODELS[size]
    return {"model": size, "model_id": mid, "revision": rev, "gen_batch": int(_value(run, "--gen-batch", "16")),
            "batch_rollout": "--batch-rollout" in run,
            "train_micro_batch": int(_value(run, "--train-micro-batch", "1")),
            "lora_r": int(_value(run, "--lora-r", "16")), "prompts_per_rank": int(_value(run, "--prompts-per-step")),
            "world": world()}


def stage_a_pin(ws: dict) -> tuple[Path, dict]:
    """The CUR23 pin of the run that wrote the INIT_LORA checkpoint (exactly one must exist)."""
    out = Path(ws["source_out"])
    found = [(out.parent / f"{out.name}.{suf}", pre) for suf, pre in STAGE_A_PINS
             if (out.parent / f"{out.name}.{suf}").is_file()]
    if len(found) != 1:
        raise SystemExit(f"REFUSED: need exactly one Stage-A CUR23 pin next to {out} "
                         f"({', '.join(s for s, _ in STAGE_A_PINS)}); found {[str(p) for p, _ in found]}")
    p, want = found[0]
    pre = json.loads(p.read_text())
    if pre.get("prereg") != want:
        raise SystemExit(f"REFUSED: {p} is not a Stage-A CUR23 pin (prereg={pre.get('prereg')!r})")
    return p, pre


def stage_a_diff(src: dict, now: dict) -> list[str]:
    """Fields where the recall run differs from its Stage-A run (must be empty)."""
    bad = [f"{b}" for b in SAME_BLOCKS if src.get(b) != now.get(b)]
    sc, nc = src.get("cur") or {}, now.get("cur") or {}
    bad += [f"cur.{k}" for k in SAME_CUR if sc.get(k) != nc.get(k)]
    bad += [f"cur.quotas.{k}" for k in SAME_QUOTAS if (sc.get("quotas") or {}).get(k) != (nc.get("quotas") or {}).get(k)]
    bad += [f"v4h.{k}" for k in SAME_V4H if (src.get("v4h") or {}).get(k) != (now.get("v4h") or {}).get(k)]
    bad += [f"constants.{k}" for k in SAME_CONSTANTS
            if (src.get("constants") or {}).get(k) != (now.get("constants") or {}).get(k)]
    src_model = (src.get("m9b") or {}).get("model", "4b")
    if src_model != now["r23"]["model"]["model"]:
        bad.append(f"model({src_model}->{now['r23']['model']['model']})")
    return bad


def r23_block(o, run: list[str], ws: dict, src_path: Path) -> dict:
    from trainer_v2_v4 import recall_decision as RD
    return {"code_sha256": {f: P21._sha(HERE / f) for f in CODE},
            "recall": {"recall_decision": o.recall_decision, "crowdout_guard": o.crowdout_guard,
                       "source_preset": RD.SOURCE_PRESET, "missed_call": RD.MISSED_CALL,
                       "unneeded_call": RD.UNNEEDED_CALL, "r": float(_value(run, "--r", "nan")),
                       "guard_rule": RD.GUARD_RULE, "version": RD.VERSION},
            "model": model_block(run),
            "ws": {k: ws[k] for k in ("path", "adapter_sha256", "source_complete", "step_dir", "source_out")},
            "stage_a": {"pin": str(src_path), "pin_sha256": P21._sha(src_path),
                        "compared": {"blocks": list(SAME_BLOCKS), "cur": list(SAME_CUR), "quotas": list(SAME_QUOTAS),
                                     "v4h": list(SAME_V4H), "constants": list(SAME_CONSTANTS), "model": True}}}


def block(run: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur23 as M23
    from trainer_v2_v4 import main_v4hcur23r as MR
    o, rest, ws = MR.resolve(run)
    o23, o22b, rest23 = M23.resolve(rest)
    if o23.prereg_v4hcur23:
        raise SystemExit("REFUSED: a recall run is pinned by pins_v4hcur23r, not --prereg-v4hcur23")
    out = P23.block(o23, o22b, rest23)
    out["base_prereg"] = P23.PREREG
    out["prereg"] = PREREG
    src_path, src = stage_a_pin(ws)
    out["r23"] = r23_block(o, run, ws, src_path)
    out["r23"]["stage_a"]["schedule_sha256"] = {"stage_a": (src.get("cur") or {}).get("schedule_sha256"),
                                                "recall": (out.get("cur") or {}).get("schedule_sha256")}
    out = json.loads(json.dumps(out, default=str))
    bad = stage_a_diff(src, out)
    if bad:
        raise SystemExit(f"REFUSED: the recall run differs from its Stage-A run {src_path}: {bad}")
    return out


def verify(pin: Path, run: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(run)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur23r pin (prereg={pre.get('prereg')!r})")
    bad = P21.diff(pre, now, BLOCKS)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur23r pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur23r.py write --out PIN.json -- <main_v4hcur23r args>")
    out, run = Path(argv[2]), argv[argv.index("--") + 1:]
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    blk = block(run)
    if (blk.get("cur") or {}).get("curriculum") != "v22":
        raise SystemExit("REFUSED: a v4hcur23r pin needs --curriculum v22")
    if blk["r23"]["recall"]["recall_decision"] == "off":
        raise SystemExit("REFUSED: a v4hcur23r pin needs --recall-decision ws52")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
