"""Judges for one system's conflict-subset run: acute v4 -> answer-bound packets -> severity v2 (contraindicated only) + MIE v1.

Frozen operators, unchanged: acute = v4i_freeze_codex/acute_v4_bridge V (acute_adequacy_v4.main, rubric v4, Luna) with
the canonical bridge g0_redesign_v1/rebuild_v20/acute_bridge_ae.build_item; severity / MIE = final_eval_assembly/secondary.
The only rebinding is the reference data:
  * acute key actions: acute_bridge_ae.configure() pins KEY_ACTIONS_CANONICAL_v4q.jsonl (sha in rebuild_v20/BINDING.json)
    and binds it to the 3-mode v4q Catalog; it cannot take another file. Here acute_bridge_ae.REFS is filled from
    KEY_ACTIONS_conflict_v1.jsonl (sha pinned by the conflict MANIFEST, each row bound to its release row by canonical
    id + source_image_sha256, as configure() does) and V.KEY_ACTIONS points at it, so verdicts record its sha.
  * MIE inventory: INVENTORY_conflict_v1.jsonl (MANIFEST-pinned) via secondary's --inventory.
Packets are built as the assembler builds EPISODES rows (canonical_id, image = source_image_sha256, response_sha256 =
sha256(stage2), answer = answer_text(stage2), item = bridge item, acute_verdict joined by episode key).

  python judge_conflict.py --system <sid> --task acute|severity|mie [--cap X]          # PRELAUNCH line only (CPU)
  python judge_conflict.py approve --line-file <file>                                   # owner (OWNER_API_APPROVED=1)
  env TEST_FREEZE_APPROVED=1 OWNER_API_APPROVED=1 G0_QA_KEYS_REVIEWED=1 OPENAI_API_KEY=... \
    python judge_conflict.py --system <sid> --task <t> --execute [--cap X]             # paid, Luna
  python judge_conflict.py --system <sid> --task <t> --backend export|cue|table [--table T] --out-root <dir>   # CPU tests
Default caps (per system): acute 0.12, severity 0.05, MIE 0.04. Outputs judges/<sid>_<task>.jsonl, write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conflict_common as C  # noqa: E402

FC = C.FC
AE = FC / "v4_build/g0_redesign_v1/rebuild_v20"
SW = FC / "v4_build/ablation_ckpt_eval_ag_v4q"
CAP = {"acute": 0.12, "severity": 0.05, "mie": 0.04}
SETTING = "memory_call,retrieved,greedy"


def _paths():
    sys.dont_write_bytecode = True
    for p in (str(FC), str(FC / "v4_build/final_eval_assembly"), str(AE), str(SW)):
        if p not in sys.path:
            sys.path.insert(0, p)


def bridge():
    """acute_bridge_ae with conflict key actions (see docstring); returns (BR, V)."""
    _paths()
    import acute_bridge_ae as BR
    m = C.manifest()
    rows = {r["evaluator"]["row_id"]: r for r in C.jl(C.HOLDOUT)}
    refs = {}
    for k in C.jl(C.KEY_ACTIONS):
        r = rows.get(k["row_id"])
        if r is None or r["evaluator"]["source_image_sha256"] != k["source_image_sha256"] or k["row_id"] in refs:
            raise ValueError("unbound/mismatched/duplicate conflict key action " + k["row_id"])
        refs[k["row_id"]] = k
    if set(refs) != set(rows):
        raise ValueError("key actions do not cover the release exactly")
    BR.CAT = "conflict_v1 (build_item is called without a mode)";BR.REFS = refs
    BR.V.KEY_ACTIONS = (C.KEY_ACTIONS, m["key_actions_sha256"]);BR.V.rubric_text("v4");BR.V.build_item = BR.build_item
    return BR, BR.V


def out_path(root: Path, sid: str, task: str) -> Path:
    return root / f"{sid}_{task}.jsonl"


def packets_path(root: Path, sid: str) -> Path:
    return root / "packets" / f"{sid}_episodes.jsonl"


def build_packets(sid: str, run: Path, root: Path) -> Path:
    """EPISODES-like rows (acute only, all 81) with the acute verdict attached; write-once."""
    BR, V = bridge()
    from eval_v1 import hazard_match as HM
    acute = {(v["row_id"], v["arm"], v["mode"], v["kind"], v.get("sample_idx", 0)): v for v in C.jl(out_path(root, sid, "acute"))}
    rows = {r["sample_id"]: r for r in C.jl(C.HOLDOUT)}
    out = []
    for t in C.jl(run / "texts.jsonl"):
        k = (t["row_id"], t["arm"], t["mode"], t["kind"], t.get("sample_idx", 0))
        if (t["arm"], t["mode"], t["kind"]) != ("memory_call", "retrieved", "greedy"):
            continue
        row = rows[t["row_id"]];s2 = t.get("stage2") or ""
        out.append({"model": sid, "canonical_id": row["evaluator"]["row_id"], "mode": "retrieved", "arm": "memory_call",
                    "kind": "greedy", "sample_idx": t.get("sample_idx", 0), "local_id": t["row_id"],
                    "response_sha256": hashlib.sha256(s2.encode()).hexdigest(), "image": row["evaluator"]["source_image_sha256"],
                    "run": str(run), "acute": True, "acute_verdict": acute.get(k), "acute_state": (acute.get(k) or {}).get("state"),
                    "answer": HM.answer_text(s2), "item": BR.build_item(row)})
    if len(out) != C.N_ROWS or any(e["acute_verdict"] is None for e in out):
        raise RuntimeError(f"{sid}: packets need {C.N_ROWS} rows with acute verdicts")
    p = packets_path(root, sid);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        f.writelines(json.dumps(e) + "\n" for e in out)
    return p


def descriptor(sid: str, task: str, cap: float, run: Path, packet: Path | None) -> str:
    m = C.manifest()
    d = {"work": "CONFLICT_V10", "system": sid, "task": task, "backend": "luna", "cap_usd": cap,
         "model": C.API.get(sid) or f"local:{C.LOCAL[sid][0]}:{C.LOCAL[sid][1]}", "release_sha256": m["holdout_sha256"],
         "reference_sha256": m["key_actions_sha256"] if task != "mie" else m["mie_inventory_sha256"],
         "run_report_sha256": C.sha(run / "eval_report.json") if (run / "eval_report.json").is_file() else "PENDING_GENERATION",
         "texts_sha256": C.sha(run / "texts.jsonl") if (run / "texts.jsonl").is_file() else "PENDING_GENERATION",
         "packet_sha256": None if task == "acute" else (C.sha(packet) if packet and packet.is_file() else "PENDING_PACKETS")}
    return "PRELAUNCH " + json.dumps(d, sort_keys=True, separators=(",", ":"))


def approve(line: str, path: Path = C.APPROVAL) -> None:
    if os.environ.get("OWNER_API_APPROVED") != "1":
        raise RuntimeError("OWNER_API_APPROVED=1 required to record an approval")
    line = line.strip()
    if not line.startswith("PRELAUNCH {") or '"work":"CONFLICT_V10"' not in line:
        raise ValueError("not a conflict PRELAUNCH line")
    old = C.jl(path) if path.exists() else []
    if any(r.get("line") == line for r in old):
        print("already approved");return
    with path.open("a") as f:
        f.write(json.dumps({"owner_approved": True, "approved_by": "owner (judge_conflict.py approve, OWNER_API_APPROVED=1)",
                            "line": line, "line_sha256": hashlib.sha256(line.encode()).hexdigest()}) + "\n")
    print("recorded in", path)


def execute(sid: str, task: str, backend: str, cap: float, root: Path, run: Path, table: Path | None = None) -> Path:
    out = out_path(root, sid, task)
    if out.exists():
        raise FileExistsError(f"write-once verdict output exists: {out}")
    root.mkdir(parents=True, exist_ok=True)
    if task == "acute":
        BR, V = bridge()
        args = ["--run", str(run), "--setting", SETTING, "--backend", backend, "--rubric", "v4", "--out", str(out)]
        if backend == "luna":
            args += ["--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))]
        if backend == "table":
            args += ["--table", str(table)]
        V.main(args)
        return out
    packet = packets_path(root, sid)
    if not packet.is_file():
        raise RuntimeError("packets missing: run the PRELAUNCH print for severity/mie first (it builds them after acute)")
    _paths()
    import secondary
    secondary.HERE = root.resolve()
    args = ["--task", task, "--episodes", str(packet), "--backend", backend, "--out", str(out)]
    if backend == "luna":
        args += ["--execute-paid", "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))]
    if backend == "table":
        args += ["--table", str(table)]
    if task == "mie":
        args += ["--inventory", str(C.INVENTORY)]
    secondary.main(args)
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["approve"]:
        if argv[1:2] != ["--line-file"] or len(argv) != 3:
            raise SystemExit(__doc__)
        lines = [x for x in Path(argv[2]).read_text().splitlines() if x.startswith("PRELAUNCH ")]
        if len(lines) != 1:
            raise SystemExit(f"expected exactly one PRELAUNCH line, found {len(lines)}")
        return approve(lines[0])
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=C.SYSTEMS, required=True);p.add_argument("--task", choices=list(CAP), required=True)
    p.add_argument("--cap", type=float);p.add_argument("--execute", action="store_true")
    p.add_argument("--backend", choices=["luna", "export", "cue", "table"], default="luna");p.add_argument("--table", type=Path)
    p.add_argument("--run", type=Path);p.add_argument("--out-root", type=Path, default=C.JUDGES);p.add_argument("--approval", type=Path, default=C.APPROVAL)
    a = p.parse_args(argv)
    cap = a.cap or CAP[a.task];run = a.run or C.run_dir(a.system);root = a.out_root
    if a.backend != "luna":   # CPU paths (tests): no gates, never paid
        if a.task != "acute" and not packets_path(root, a.system).is_file():
            build_packets(a.system, run, root)
        print(execute(a.system, a.task, a.backend, cap, root, run, a.table));return
    if (run / "eval_report.json").is_file():
        C.check_run(a.system, run)
    packet = None
    if a.task != "acute":
        packet = packets_path(root, a.system)
        if not packet.is_file() and out_path(root, a.system, "acute").is_file():
            packet = build_packets(a.system, run, root);print(f"# wrote packets {packet}")
    line = descriptor(a.system, a.task, cap, run, packet)
    if not a.execute:
        print(line);print("# owner must approve this exact line (judge_conflict.py approve) before --execute");return
    if any(os.environ.get(k) != "1" for k in ("TEST_FREEZE_APPROVED", "OWNER_API_APPROVED", "G0_QA_KEYS_REVIEWED")):
        raise RuntimeError("owner gates required")
    _paths()
    from judges_sweep_v2 import approval
    approval(a.approval, line)
    C.check_run(a.system, run)
    print(execute(a.system, a.task, "luna", cap, root, run))


if __name__ == "__main__":
    main()
