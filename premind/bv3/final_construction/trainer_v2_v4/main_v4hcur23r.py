#!/usr/bin/env python3
"""PReMind v2 entry point of the RECALL phase (Stage B) CUR23R = the frozen CUR23 stack (main_v4hcur23, imported
read-only: curriculum v2.2, bonus v1.1, fa25, AA / AVR / warning credit, hazard rubric v4, severity cap) on the
memory_call arm, warm-started from a finished Stage-A CUR23 checkpoint, + (recall_decision.py):
  --recall-decision off|ws52        ws52: call decision re-valued with the frozen v52ws_ag weights (missed -2r,
                                    unneeded -0.5r, correct +r)
  --crowdout-guard off|credit_gate  credit_gate: on acute rows the +r call credit is paid only if the answer addresses
                                    the hazard (state specific / generic_only, not contraindicated)
  --prereg-v4hcur23r PIN            (pins_v4hcur23r.py write) verified before anything runs; the CUR23 pin flags
                                    (--prereg-v4hcur23, --prereg-v4hcur23-9b) and every older one are refused.
Refuses unless --arm memory_call, --model 4b|9b and a warm start: --init-lora <Stage-A run>/checkpoints/step_N/lora
(COMPLETE marker + adapter files) or, with --resume, OUT/init_lora.json of the warm-started run.
The inner main_v4hcur23.main logs "run without --prereg-v4hcur23": expected (the CUR23R pin covers its blocks).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2_v4 import recall_decision as RD  # noqa: E402

logger = logging.getLogger(__name__)
FLAG = "--prereg-v4hcur23r"
REFUSED_FLAGS = ("--prereg-v4hcur23", "--prereg-v4hcur23-9b")
MODELS = ("4b", "9b")
R_KEYS = ("recall_decision", "crowdout_guard")


def opt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--recall-decision", default="off", choices=RD.DECISION_MODES)
    ap.add_argument("--crowdout-guard", default="off", choices=RD.GUARD_MODES)
    return ap


def _value(argv: list[str], flag: str) -> Optional[str]:
    from trainer_v2_v4 import main_v4h as MH
    return MH._value(argv, flag)


def init_lora_of(rest: list[str]) -> dict:
    """The warm-start LoRA of the run: {path, adapter_sha256, source_complete, step_dir, source_out, via}. Refuses
    (SystemExit) when it is missing: no --init-lora (fresh run), a path without the COMPLETE marker / adapter files,
    or a --resume without OUT/init_lora.json."""
    from trainer_v2 import paths_v2 as P
    lora = _value(rest, "--init-lora")
    via = "--init-lora"
    if lora is None and "--resume" in rest:
        out = _value(rest, "--out")
        meta = Path(out or ".") / "init_lora.json"
        if not meta.is_file():
            raise SystemExit(f"REFUSED: --resume of a recall run needs {meta} (written by the warm-started run)")
        lora, via = json.loads(meta.read_text())["path"], "OUT/init_lora.json"
    if lora is None:
        raise SystemExit("REFUSED: the recall phase needs --init-lora <Stage-A run>/checkpoints/step_N/lora (INIT_LORA)")
    p = Path(lora)
    if not p.is_absolute():
        raise SystemExit(f"REFUSED: INIT_LORA must be absolute: {lora}")
    step = p.parent
    need = [step / "COMPLETE", p / "adapter_model.safetensors", p / "adapter_config.json"]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        raise SystemExit(f"REFUSED: INIT_LORA missing / incomplete ({lora}): no {', '.join(miss)}")
    return {"path": str(p.resolve()), "adapter_sha256": P.sha256_file(p / "adapter_model.safetensors"),
            "source_complete": (step / "COMPLETE").read_text().strip(), "step_dir": step.name,
            "source_out": str(step.parent.parent.resolve()), "via": via}


def split(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """(CUR23R pin path or None, the remaining argv)."""
    if argv[:1] == ["--"]:
        argv = argv[1:]
    pin, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == FLAG and i + 1 < len(argv):
            pin, i = argv[i + 1], i + 2
            continue
        if a.startswith(FLAG + "="):
            pin, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    return pin, rest


def resolve(argv: list[str]) -> tuple[argparse.Namespace, list[str], dict]:
    """(recall options, the main_v4hcur23 argv, init LoRA meta). ``argv`` must not contain the CUR23R pin flag."""
    from trainer_v2_v4 import main_v4hcur23 as M23
    if argv[:1] == ["--"]:
        argv = argv[1:]
    for flag in (FLAG,) + REFUSED_FLAGS:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(f"REFUSED: {flag} is not accepted here (main_v4hcur23r is pinned by {FLAG} only)")
    o, rest = opt_parser().parse_known_args(argv)
    if _value(rest, "--arm") != "memory_call":
        raise SystemExit("REFUSED: the recall phase needs --arm memory_call")
    if _value(rest, "--model") not in MODELS:
        raise SystemExit(f"REFUSED: main_v4hcur23r needs --model {' | '.join(MODELS)}")
    if _value(rest, "--reward-preset") != "v4wh_g":
        raise SystemExit("REFUSED: the recall phase is defined over --reward-preset v4wh_g (the CUR23 stack)")
    ws = init_lora_of(rest)
    M23.resolve(rest)                                   # CUR23 refusals (older pin flags, judge requirements)
    return o, rest, ws


def _store_args(o: argparse.Namespace) -> None:
    """Non-default values join the checkpoint args (v4h_<key>)."""
    on = {k: getattr(o, k) for k in R_KEYS if getattr(o, k) != "off"}
    if not on:
        return
    from trainer_v2_v4 import main_v4 as MV4
    orig = MV4.parse

    def parse(argv=None):
        args, pc = orig(argv)
        for k, v in on.items():
            setattr(args, f"v4h_{k}", v)
        return args, pc
    MV4.parse = parse


def main(argv: Optional[list[str]] = None) -> int:
    from trainer_v2_v4 import main_v4hcur23 as M23
    logging.basicConfig(level=logging.INFO)
    pin, run = split(list(sys.argv[1:] if argv is None else argv))
    o, rest, ws = resolve(run)
    if pin:
        from trainer_v2_v4 import pins_v4hcur23r as PR
        PR.verify(Path(pin), run)
        logger.info("v4hcur23r pin verified (CUR23 blocks + r23): %s", pin)
    else:
        logger.warning("v4hcur23r run without %s (CPU only; the launcher requires a pin)", FLAG)
    if "--print-config" in rest:
        print(json.dumps({"v4hcur23r": {"recall_decision": o.recall_decision, "crowdout_guard": o.crowdout_guard,
                                        "missed_call": RD.MISSED_CALL, "unneeded_call": RD.UNNEEDED_CALL,
                                        "guard_rule": RD.GUARD_RULE, "init_lora": ws}}, indent=1))
    else:
        RD.install(o.recall_decision, o.crowdout_guard)    # innermost: before main_v4hcur23 installs its hooks
        _store_args(o)
        logger.info("cur23r: recall decision %s (missed %sr, unneeded %sr), crowd-out guard %s, init LoRA %s",
                    o.recall_decision, RD.MISSED_CALL, RD.UNNEEDED_CALL, o.crowdout_guard, ws["path"])
    return M23.main(rest)


if __name__ == "__main__":
    sys.exit(main())
