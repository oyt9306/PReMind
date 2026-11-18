#!/usr/bin/env python3
"""Training PREREG pins (WP-T7): trainer_v2/PREREG_TRAINER_V2.json.

Pinned: sha256 of the trainer_v2 code that defines rewards, data, hints and the loss; the model id + revision for
4B and 9B; the reward-preset table (every preset's flags) and the default preset; the OPSD hint templates; the data
file sha256 + render_version; seed, r and lambda as recorded POC values. ``verify`` refuses a run (SystemExit) when any
pinned sha differs or the run's data file is not the pinned one. Regenerate only on purpose:
``python -m trainer_v2.pins write [--rows ...]`` (the orchestrator records the reason in COLLABORATION.md).
VEF reward arms (general_mix_v1/DESIGN.md §2.7) use a separate release pin with a ``vef`` block (vef.py /
vef_judge.py sha256, QA + judge template sha, W table, judge served name + snapshot path), written once with
``python -m trainer_v2.pins write-vef --base <release PREREG> --rows <train.jsonl> --judge-root <snapshot> --out ...``;
``verify`` checks the block when present and ``verify_vef`` checks the live judge against it.
VEF v2 (general_mix_v1/DESIGN_v2.md): ``write-vef2`` adds a ``vef2`` block (vef2.py sha, W_scene / W_match, length cap,
v2 template sha incl. the CONFIDANT_SCHEMA=vef2 output format); ``verify`` checks it when present and ``verify_vef``
requires it for the vef2 preset.
Warm start (owner 2026-10-10, Stage B: memory_call DAPO from the always-memory LoRA): ``write-ws`` adds a ``ws`` block
(init LoRA dir + adapter sha256 + source step) to a copy of a release pin; ``verify`` re-hashes the pinned adapter and
``verify_ws`` requires the run's --init-lora to be the pinned one (and a ws pin to be used only with --init-lora).
Warm start A+C (owner 2026-10-10, arm DAPO_ws_ag, preset v52ws_ag): ``write-ws-ag`` = ``write-ws`` + an ``ag`` block
(preset flags, prompt schema vef2, withhold_ag.py / vef2.py / vef.py sha256, vef2 template sha incl. the output format);
``verify`` checks the block when present and ``verify_ag`` requires it for a withhold_match preset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from trainer_v2 import opsd as OP  # noqa: E402
from trainer_v2 import paths_v2 as P  # noqa: E402
from trainer_v2 import reward_v5 as R5  # noqa: E402
from trainer_v2 import vef as VF  # noqa: E402
from trainer_v2 import vef2 as VF2  # noqa: E402

PREREG = HERE / "PREREG_TRAINER_V2.json"
CODE = ("reward_v5.py", "withhold_v9.py", "vcm_data.py", "opsd.py", "grpo_v2.py", "hf_backend.py", "metrics_v9.py",
        "run_v2.py", "paths_v2.py", "teacher_table.py", "zs_eval.py", "policy_loss.py", "batch_rollout.py",
        "train_mb.py", "algo_v2.py", "records_v2.py", "cli_v2.py", "fast_conv.py", "ddp_v2.py", "personal_match.py", "hazard_match.py")
MODELS = {"4b": ("Qwen/Qwen3.5-4B", P.MODEL_REVISION), "9b": ("Qwen/Qwen3.5-9B", P.MODEL_REVISION_9B)}
VEF_CODE = ("vef.py", "vef_judge.py")
VEF2_CODE = ("vef2.py",)
AG_CODE = ("withhold_ag.py", "vef2.py", "vef.py")
AG_PRESET = "v52ws_ag"


def code_shas() -> dict[str, str]:
    return {f: P.sha256_file(HERE / f) for f in CODE}


def hint_template_sha() -> str:
    return P.sha256_text(json.dumps([OP.HEADER, OP.H1_PRINCIPLES, OP.H2_TEXT, OP.HINT_SETS], sort_keys=True))


def presets_table() -> dict:
    return {k: R5.flags_dict(v) for k, v in R5.PRESETS.items()}


def vef_block(judge_root: str, served_name: str = "vef-judge") -> dict:
    return {"W": dict(VF.PRESETS), "code_sha256": {f: P.sha256_file(HERE / f) for f in VEF_CODE},
            "template_sha256": VF.template_sha(), "judge": {"served_model_name": served_name, "root": judge_root}}


def vef2_block() -> dict:
    return {"W": dict(VF2.PRESETS), "max_words": VF2.MAX_WORDS, "schema": VF2.SCHEMA,
            "code_sha256": {f: P.sha256_file(HERE / f) for f in VEF2_CODE}, "template_sha256": VF2.template_sha()}


def _vef_bad(pre: dict) -> list[str]:
    if "vef" not in pre:
        return []
    now = vef_block(pre["vef"]["judge"]["root"], pre["vef"]["judge"]["served_model_name"])
    bad = [f"vef:{k}" for k in ("W", "code_sha256", "template_sha256") if pre["vef"][k] != now[k]]
    if "vef2" in pre:
        bad += [f"vef2:{k}" for k, v in vef2_block().items() if pre["vef2"].get(k) != v]
    return bad


WS_ADAPTER = "adapter_model.safetensors"


def ws_block(init_lora: Path) -> dict:
    """``init_lora`` = a checkpoint's ``lora`` dir (step dir = parent, which must carry the COMPLETE marker)."""
    lora = Path(init_lora).resolve()
    if not (lora.parent / "COMPLETE").is_file():
        raise SystemExit(f"REFUSED: {lora.parent} has no COMPLETE marker")
    return {"init_lora": str(lora), "adapter_sha256": P.sha256_file(lora / WS_ADAPTER),
            "source_complete": (lora.parent / "COMPLETE").read_text().strip(),
            "arms": {"DAPO_ws": "main", "DAPO_ws_miss": "v52miss"},
            "protocol": "fresh run (step 0, fresh optimizer / scheduler, RNG from SEED) from this LoRA; memory_call arm"}


def ag_block() -> dict:
    return {"preset": AG_PRESET, "flags": R5.flags_dict(R5.PRESETS[AG_PRESET]), "schema": VF2.SCHEMA,
            "code_sha256": {f: P.sha256_file(HERE / f) for f in AG_CODE}, "vef2_template_sha256": VF2.template_sha()}


def _ag_bad(pre: dict) -> list[str]:
    if "ag" not in pre:
        return []
    return [f"ag:{k}" for k, v in ag_block().items() if pre["ag"].get(k) != v]


def verify_ag(path: Path, preset: str) -> dict:
    """A withhold_match preset (v52ws_ag) needs a pin with the ``ag`` block (checked by ``verify``)."""
    pre = json.loads(Path(path).read_text())
    if R5.PRESETS[preset].withhold_match and "ag" not in pre:
        raise SystemExit(f"REFUSED: {path} has no ag block (the {preset} arm needs the WSAG release pin)")
    return pre


def _ws_bad(pre: dict) -> list[str]:
    if "ws" not in pre:
        return []
    f = Path(pre["ws"]["init_lora"]) / WS_ADAPTER
    return [] if f.is_file() and P.sha256_file(f) == pre["ws"]["adapter_sha256"] else ["ws:adapter_sha256"]


def verify_ws(path: Path, init_lora: Optional[str], sha: Optional[str]) -> dict:
    """A warm-start run needs a pin with the matching ``ws`` block; a ``ws`` pin needs a warm-start run."""
    pre = json.loads(Path(path).read_text())
    if "ws" not in pre and init_lora is None:
        return pre
    if "ws" not in pre:
        raise SystemExit(f"REFUSED: {path} has no ws block (a warm-start run needs the WS release pin)")
    if init_lora is None:
        raise SystemExit(f"REFUSED: {path} is a warm-start pin; the run has no --init-lora")
    if (str(Path(init_lora).resolve()), sha) != (pre["ws"]["init_lora"], pre["ws"]["adapter_sha256"]):
        raise SystemExit(f"REFUSED: init LoRA {init_lora} ({sha}) differs from the pinned {pre['ws']}")
    return pre


def build(rows: Path) -> dict:
    data = json.loads(Path(rows).read_text())
    rv = sorted({r.get("render_version") or "s4_canary" for r in data["rows"]})
    return {"prereg": "PREREG_TRAINER_V2", "status": "POC (overnight 2026-10-03; diagnostic, not confirmatory)",
            "code_sha256": code_shas(), "models": {k: {"id": a, "revision": b} for k, (a, b) in MODELS.items()},
            "reward": {"version": R5.REWARD_VERSION, "default_preset": "main", "presets": presets_table(),
                       "r": {"value": 0.1, "status": "plumbing value; R24/R25 calibration (WP-T4) pending"}},
            "opsd": {"lambda": OP.LAMBDA, "topk": OP.TOPK, "default_hint_set": "main",
                     "hint_template_sha256": hint_template_sha(), "table_schema": OP.TABLE_SCHEMA},
            "seed": 0, "data": {"rows": str(Path(rows).resolve()), "sha256": P.sha256_file(Path(rows)),
                                "render_version": rv}}


def verify(path: Path = PREREG, *, rows: Optional[Path] = None) -> dict:
    """Refuse (SystemExit) when code, model pins, preset table, hint templates or the data file differ."""
    pre = json.loads(Path(path).read_text())
    bad = [f"code:{f}" for f, s in code_shas().items() if pre["code_sha256"].get(f) != s]
    bad += [f"model:{k}" for k, (a, b) in MODELS.items() if pre["models"][k] != {"id": a, "revision": b}]
    bad += ["reward_presets"] if pre["reward"]["presets"] != presets_table() else []
    bad += ["hint_templates"] if pre["opsd"]["hint_template_sha256"] != hint_template_sha() else []
    bad += _vef_bad(pre)
    bad += _ws_bad(pre)
    bad += _ag_bad(pre)
    if rows is not None and (str(Path(rows).resolve()) != pre["data"]["rows"]
                             or P.sha256_file(Path(rows)) != pre["data"]["sha256"]):
        bad.append("data")
    if bad:
        raise SystemExit(f"REFUSED: PREREG pins differ: {bad} (regenerate deliberately with trainer_v2.pins write)")
    return pre


def verify_vef(path: Path, served: dict, preset: Optional[str] = None) -> dict:
    """The live judge (VefJudge.health_check) must be the pinned served name + snapshot path; a vef2 preset needs the
    pin's ``vef2`` block."""
    pre = json.loads(Path(path).read_text())
    if "vef" not in pre:
        raise SystemExit(f"REFUSED: {path} has no vef block (a VEF arm needs the VEF release pin)")
    if preset in VF2.PRESETS and "vef2" not in pre:
        raise SystemExit(f"REFUSED: {path} has no vef2 block (the {preset} arm needs the VEF2 release pin)")
    want = pre["vef"]["judge"]
    if (served.get("id"), served.get("root")) != (want["served_model_name"], want["root"]):
        raise SystemExit(f"REFUSED: VEF judge {served} differs from the pinned {want}")
    return pre


def build_vef(base: Path, rows: Path, judge_root: str) -> dict:
    """A release pin for the VEF arms: ``base`` (e.g. PREREG_RELEASE_MAIN_v7h_*) with the current code / preset pins
    and the ``vef`` block; the data pin must be the base's."""
    pre = json.loads(Path(base).read_text())
    if str(Path(rows).resolve()) != pre["data"]["rows"] or P.sha256_file(Path(rows)) != pre["data"]["sha256"]:
        raise SystemExit("REFUSED: rows differ from the base pin's data")
    pre.update(prereg="PREREG_EVAL_V1_RELEASE_VEF", code_sha256=code_shas(), vef=vef_block(judge_root),
               status=f"draft VEF release pin from {Path(base).name}; orchestrator approval required")
    pre["reward"] = {**pre["reward"], "presets": presets_table(),
                     "version": {"main": R5.REWARD_VERSION, "vef": R5.VEF_REWARD_VERSION}}
    return pre


def build_vef2(base: Path, rows: Path, judge_root: str) -> dict:
    """``build_vef`` + the ``vef2`` block (the VEF v2 arm's release pin)."""
    pre = build_vef(base, rows, judge_root)
    pre.update(prereg="PREREG_EVAL_V1_RELEASE_VEF2", vef2=vef2_block(),
               status=f"draft VEF2 release pin from {Path(base).name}; orchestrator approval required")
    pre["reward"]["version"]["vef2"] = R5.VEF2_REWARD_VERSION
    return pre


def build_ws(base: Path, rows: Path, init_lora: Path) -> dict:
    """A release pin for the warm-start arms: ``base`` (PREREG_RELEASE_MAIN_v7h_*) with the current code / preset pins
    and the ``ws`` block; the data pin must be the base's."""
    pre = json.loads(Path(base).read_text())
    if str(Path(rows).resolve()) != pre["data"]["rows"] or P.sha256_file(Path(rows)) != pre["data"]["sha256"]:
        raise SystemExit("REFUSED: rows differ from the base pin's data")
    pre.update(prereg="PREREG_EVAL_V1_RELEASE_WS", code_sha256=code_shas(), ws=ws_block(init_lora),
               status=f"draft warm-start release pin from {Path(base).name}; orchestrator approval required")
    pre["reward"] = {**pre["reward"], "presets": presets_table()}
    return pre


def build_ws_ag(base: Path, rows: Path, init_lora: Path) -> dict:
    """``build_ws`` (same init LoRA block) for the DAPO_ws_ag arm + the ``ag`` block."""
    pre = build_ws(base, rows, init_lora)
    pre["ws"]["arms"] = {"DAPO_ws_ag": AG_PRESET}
    pre.update(prereg="PREREG_EVAL_V1_RELEASE_WSAG", ag=ag_block(),
               status=f"draft warm-start A+C ({AG_PRESET}) release pin from {Path(base).name}; orchestrator approval "
                      "required")
    return pre


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["write", "verify", "write-vef", "write-vef2", "write-ws", "write-ws-ag"])
    ap.add_argument("--rows", default=str(P.CANARY_RENDERED))
    ap.add_argument("--out", default=str(PREREG))
    ap.add_argument("--base", help="write-vef: the release PREREG to extend")
    ap.add_argument("--judge-root", help="write-vef: judge snapshot path as passed to vllm serve")
    ap.add_argument("--init-lora", help="write-ws: the warm-start LoRA dir (<run>/checkpoints/step_N/lora)")
    args = ap.parse_args(argv)
    if args.cmd in ("write-ws", "write-ws-ag"):
        if Path(args.out).exists() or Path(args.out).resolve() == PREREG.resolve():
            raise SystemExit(f"REFUSED: {args.out} exists (write-once)")
        build_pin = build_ws_ag if args.cmd == "write-ws-ag" else build_ws
        Path(args.out).write_text(json.dumps(build_pin(Path(args.base), Path(args.rows), Path(args.init_lora)),
                                             indent=2, sort_keys=True) + "\n")
        verify(Path(args.out), rows=Path(args.rows))
        print(f"wrote {args.out}")
        return 0
    if args.cmd in ("write-vef", "write-vef2"):
        if Path(args.out).exists() or Path(args.out).resolve() == PREREG.resolve():
            raise SystemExit(f"REFUSED: {args.out} exists (write-once)")
        build_pin = build_vef2 if args.cmd == "write-vef2" else build_vef
        Path(args.out).write_text(json.dumps(build_pin(Path(args.base), Path(args.rows), args.judge_root),
                                             indent=2, sort_keys=True) + "\n")
        verify(Path(args.out), rows=Path(args.rows))
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "write":
        Path(args.out).write_text(json.dumps(build(Path(args.rows)), indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    else:
        verify(Path(args.out), rows=Path(args.rows))
        print("PINS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
