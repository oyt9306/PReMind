"""Pin of a judge-free CUR22B ablation run (main_v4hcur22abl; CUR22_ABLATIONS.md) = the CUR22B block rebuilt by the
frozen pins_v4hcur22b.block from the same arguments (v4h / aa / cur / amb / fa / constants / code_sha256) + an ``abl``
block (warning credit backend, hazard_first, stop step, recall terms, judge-free proof, code sha256) + an ``ablation``
record written once: the run id, the declared factor, and the field-level diff vs the A0 pin and vs the CUR22B pin.
  python -I -m trainer_v2_v4.pins_v4hcur22abl write --out PIN.json --id A3 [--a0-pin A0.json] [--cur22b-pin P.json] \
      -- <main_v4hcur22abl args>                       (write once; PIN_WORLD=<ranks>)
The write refuses unless the diff vs A0 is exactly the declared factor of the run (A0: the diff vs CUR22B is only the
judge-backend fields + the new abl block). ``verify`` compares everything except ``ablation``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import pins_v4hcur21 as P21  # noqa: E402

PREREG = "PREREG_TRAINER_V2_V4HCUR22ABL"
CODE = ("abl_hooks.py", "main_v4hcur22abl.py", "pins_v4hcur22abl.py", "recall_decision.py", "warning_reward.py",
        "warning_credit.py")
# fields that may differ between A0 and CUR22B: the judge backends (hazard: judge -> cue; warning: judge -> cue via abl)
JUDGE_FIELDS = ("v4h.config.hazard_backend", "v4h.config.hazard_judge_provider", "v4h.config.hazard_judge_model",
                "v4h.config.hazard_usd_cap", "v4h.config.hazard_token_cap", "v4h.config.hazard_judge_concurrency",
                "v4h.config.hazard_judge_timeout", "aa.config.warning_credit", "aa.warning", "abl")
# run id -> (factor, dotted-path prefixes that must make up the whole diff vs A0)
FACTORS = {
    "A0": ("CUR22B with judge-free rewards (hazard cue, warning credit cue), stop at step 100", ()),
    "A1": ("no curriculum (uniform absent-share sampler, same data and absent share)", ("cur",)),
    "A2": ("replay budget 0", ("cur.quotas", "cur.schedule_sha256", "cur.replay.budget",
                               "cur.runtime.replay_budget_requested")),
    "A3": ("false-alarm penalty fa25 -> off (base -0.15)", ("fa.false_alarm_pen", "fa.value")),
    "A4": ("acute memory bonus off", ("amb.acute_memory_bonus",)),
    "A5": ("hazard-first reward R_early off (= hf-off)", ("abl.hazard_first",)),
    "A6": ("single stage: memory_call arm from the base model + CUR22R recall terms (ws52, credit_gate)",
           ("constants.arm", "abl.recall")),
}
SKIP = ("ablation",)


def abl_block(o) -> dict:
    from trainer_v2_v4 import recall_decision as RD
    from trainer_v2_v4 import warning_reward as WR
    recall = None
    if o.recall_decision != "off" or o.crowdout_guard != "off":
        recall = {"decision": o.recall_decision, "guard": o.crowdout_guard, "missed_call": RD.MISSED_CALL,
                  "unneeded_call": RD.UNNEEDED_CALL, "source_preset": RD.SOURCE_PRESET, "guard_rule": RD.GUARD_RULE,
                  "version": RD.VERSION}
    wd = WR.WD.__file__
    return {"code_sha256": {f: P21._sha(HERE / f) for f in CODE},
            "warning_credit": {"mode": o.abl_warning_credit, "backend": "cue" if o.abl_warning_credit == "cue" else None,
                               "judge": None,
                               "wd_sha256": {f: P21._sha(Path(wd).parent / f) for f in ("warning_detector.py",
                                                                                         "wd_cue.py")}},
            "hazard_first": o.abl_hazard_first, "stop_at_step": o.abl_stop_at_step, "recall": recall,
            "api": {"hazard_judge": None, "warning_judge": None, "usd_cap": 0}}


def block(argv: list[str]) -> dict:
    from trainer_v2_v4 import main_v4hcur22abl as MA
    from trainer_v2_v4 import pins_v4hcur22b as PB
    o, o22b, rest22 = MA.resolve(argv)
    out = PB.block(o22b, rest22)
    out["abl_base_prereg"] = out.get("prereg")
    out["prereg"] = PREREG
    out["abl"] = abl_block(o)
    return json.loads(json.dumps(out, default=str))


def flat(d, pre: str = "") -> dict:
    if not isinstance(d, dict):
        return {pre: d}
    out = {}
    for k, v in d.items():
        p = f"{pre}.{k}" if pre else k
        out.update(flat(v, p) if isinstance(v, dict) and v else {p: v})
    return out


def diff(a: dict, b: dict, skip: tuple = SKIP) -> list[str]:
    """Dotted leaf paths whose values differ (missing = differ); top-level keys in ``skip`` and prereg ids ignored."""
    fa = flat({k: v for k, v in a.items() if k not in skip})
    fb = flat({k: v for k, v in b.items() if k not in skip})
    ign = ("prereg", "abl_base_prereg", "base_prereg")
    return sorted(k for k in set(fa) | set(fb) if k not in ign and fa.get(k, "<absent>") != fb.get(k, "<absent>"))


def covered(paths: list[str], prefixes: tuple) -> bool:
    return all(any(p == x or p.startswith(x + ".") for x in prefixes) for p in paths)


def check(rid: str, blk: dict, a0: Optional[dict], c22b: Optional[dict]) -> dict:
    """The ``ablation`` record; SystemExit unless the diff is exactly the declared factor."""
    factor, fields = FACTORS[rid]
    rec: dict = {"id": rid, "factor": factor, "declared_vs_A0": list(fields)}
    if c22b is not None:
        d = diff(c22b, blk)
        rec["diff_vs_CUR22B"] = d
        if rid == "A0" and not covered(d, JUDGE_FIELDS):
            raise SystemExit(f"REFUSED: A0 differs from CUR22B beyond the judge fields: "
                             f"{[p for p in d if not covered([p], JUDGE_FIELDS)]}")
    if rid != "A0":
        if a0 is None:
            raise SystemExit(f"REFUSED: {rid} needs --a0-pin")
        if (a0.get("ablation") or {}).get("id") != "A0":
            raise SystemExit("REFUSED: --a0-pin is not an A0 ablation pin")
        d = diff(a0, blk)
        rec["diff_vs_A0"] = d
        miss = [f for f in fields if not any(p == f or p.startswith(f + ".") for p in d)]
        if not d or not covered(d, fields) or miss:
            raise SystemExit(f"REFUSED: {rid} must differ from A0 only in {list(fields)}: diff {d}, unchanged {miss}")
    return rec


def verify(pin: Path, argv: list[str]) -> None:
    pre, now = json.loads(Path(pin).read_text()), block(argv)
    if pre.get("prereg") != PREREG:
        raise SystemExit(f"REFUSED: {pin} is not a v4hcur22abl pin (prereg={pre.get('prereg')!r})")
    bad = diff(pre, now)
    if bad:
        raise SystemExit(f"REFUSED: v4hcur22abl pin {pin} differs: {bad}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 5 or argv[0] != "write" or argv[1] != "--out" or "--" not in argv:
        raise SystemExit("usage: pins_v4hcur22abl.py write --out PIN.json --id A# [--a0-pin P] [--cur22b-pin P] -- <args>")
    cut = argv.index("--")
    head, run = argv[:cut], argv[cut + 1:]
    kv = dict(zip(head[1::2], head[2::2]))
    out, rid = Path(kv["--out"]), kv.get("--id")
    if rid not in FACTORS:
        raise SystemExit(f"REFUSED: --id must be one of {sorted(FACTORS)}")
    if out.exists():
        raise SystemExit(f"REFUSED: pin exists (write once): {out}")
    if any(a == "--prereg-v4hcur22abl" or a.startswith("--prereg-v4hcur22abl=") for a in run):
        raise SystemExit("REFUSED: the pinned arguments must not contain --prereg-v4hcur22abl")
    blk = block(run)
    if (blk.get("cur") or {}).get("curriculum") != ("off" if rid == "A1" else "v22"):
        raise SystemExit(f"REFUSED: {rid} needs --curriculum {'off' if rid == 'A1' else 'v22'}")
    load = (lambda k: json.loads(Path(kv[k]).read_text()) if kv.get(k) else None)
    blk["ablation"] = check(rid, blk, load("--a0-pin"), load("--cur22b-pin"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blk, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
