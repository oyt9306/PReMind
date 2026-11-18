"""fix2 judge plan + launchers: acute v4 / severity v2 / MIE v1 (Luna, frozen operators) on CHANGED rows only. Imports no API client.

Conflict (package <root>/conflict_eval_fix2_v1 = judge_conflict.py copy, work tag CONFLICT_FIX2):
  conflict-plan                         print, per system, PRELAUNCH -> approve -> execute commands (free)
  conflict-run --system S --task T      PAID: reserve (group luna_judges_conflict) -> judge_conflict.py --execute --cap X -> close
  conflict-rejudge --system S           PAID: MIE re-judge of judge-visible-only rows on the OLD answer packets
                                        (conflict_eval_v{1,v2add}/judges/packets) with INVENTORY_conflict_fix2_rejudge.jsonl
  conflict-merge                        CPU: judges_rejudge_merged/<set>/ = GT-fix-1 merged judges with those MIE verdicts replaced
                                        (template conflict_gtfix_v1/rejudge_mie.py --merge)
Main test (runs from infer_main_fix2.py / run_api.py; release RELEASE_TEST_v4q_fix2; canonical key actions unchanged, sha-pinned):
  main-acute --system S [--backend luna|cue] PAID with luna: acute_bridge_ae configured on the fix2 release + frozen acute v4
                                        (as judge_conflict acute) per delta job -> judges/<sid>__<mode>__<arm>_acute.jsonl
  main-episodes [--tag T]               CPU: DELTA_CONFIG_<T>.json + DELTA_EPISODES_<T>.jsonl (frozen assembler collect over the delta runs
                                        with the acute verdicts; full fields, the packets source for severity / MIE)
  main-secondary --system S --task severity|mie   PAID: final_eval_assembly/secondary.py on DELTA_EPISODES (fix2 inventory)
  main-rejudge-mie --system S           PAID: MIE re-judge of NOT re-inferred episodes whose inventory row changed (old answers,
                                        base assembly EPISODES) -> recompute_table2.py --rejudge-mie
Every paid call: --execute-paid --ledger L --usd-cap X, owner PRELAUNCH approval, OPENAI_API_KEY in env (never printed).
Merging with existing verdicts for unchanged rows happens in cr_fix2.py (conflict) and recompute_table2.py (main), both
reproduction-tested on CPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import api_common as A  # noqa: E402
import fix2_common as F  # noqa: E402

CONFLICT_SYSTEMS = F.LOCAL_SYSTEMS + F.API_ORDER
CONFLICT_CAP = {"acute": 0.12, "severity": 0.05, "mie": 0.04}       # judge_conflict defaults (<= 110 rows)
MAIN_CAP_PER_1K = {"acute": 1.0, "severity": 0.5, "mie": 0.3}       # USD per 1000 episodes (COST.md unit costs x ~1.2)
# conflict system -> (main assembly id, base assembly config) for the main-test delta
ASM = {"ours_4b_p2e": "ours_4b", "ours_4b_p2e_s1": "ours_4b"}
CFG_OF = {s: "abl" if s.startswith("abl_") else ("s1" if s == "ours_4b_p2e_s1" else "main") for s in CONFLICT_SYSTEMS}
GW = "TEST_FREEZE_APPROVED=1 OWNER_API_APPROVED=1 G0_QA_KEYS_REVIEWED=1 CUDA_VISIBLE_DEVICES="


def croot(materialize: bool) -> Path:
    return F.out_root(materialize) / F.CONFLICT_EVAL


def mroot(materialize: bool) -> Path:
    return F.out_root(materialize) / F.MAIN_EVAL


def paid_gate(a, step: str, group: str, cap: float) -> None:
    A.require_paid(a)
    if not a.materialize:
        raise A.Refused("REFUSED: paid steps write under v4_build: --materialize required")
    for k in ("TEST_FREEZE_APPROVED", "OWNER_API_APPROVED", "G0_QA_KEYS_REVIEWED"):
        if os.environ.get(k) != "1":
            raise A.Refused(f"REFUSED: owner gate {k}=1 required")
    A.require_key("openai")
    A.reserve(a.ledger, step, "openai", cap, group=group)


# ----------------------------------------------------------------------------------------------- conflict
def conflict_plan(a) -> None:
    CE = croot(a.materialize)
    print(f"# conflict judges (Luna); package {CE}; caps per system: {CONFLICT_CAP}; group cap ${A.STEP_CAPS['luna_judges_conflict']}")
    for s in CONFLICT_SYSTEMS:
        for t in ("acute", "severity", "mie"):
            print(f"{F.PY} -B {CE}/judge_conflict.py --system {s} --task {t} > {CE}/judges/PL_{s}_{t}.txt   # PRELAUNCH -> EXPERIMENT_LOG")
            print(f"OWNER_API_APPROVED=1 {F.PY} -B {CE}/judge_conflict.py approve --line-file {CE}/judges/PL_{s}_{t}.txt")
            print(f"env {GW} OPENAI_API_KEY=... {F.PY} -B {WORK}/judge_fix2.py conflict-run --system {s} --task {t} --materialize "
                  f"--execute-paid --ledger {A.default_ledger(True)} --usd-cap {CONFLICT_CAP[t]}")


def conflict_run(a) -> None:
    CE = croot(a.materialize)
    step = f"judge_conflict:{a.system}:{a.task}"
    paid_gate(a, step, "luna_judges_conflict", a.usd_cap)
    rc = subprocess.run([F.PY, "-B", str(CE / "judge_conflict.py"), "--system", a.system, "--task", a.task, "--execute", "--cap", str(a.usd_cap)],
                        cwd=F.FC).returncode
    print(step, "actual USD", A.close(a.ledger, step, [CE / "judges" / f"{a.system}_{a.task}.ledger.jsonl"]), "|", A.status(a.ledger))
    if rc:
        raise SystemExit(rc)


def _rejudge_sets(materialize: bool) -> tuple[set[str], Path]:
    b = F.out_root(materialize) / F.CONFLICT_BUILD
    ch = json.loads((b / "CHANGED_ROWS_conflict_fix2.json").read_text())
    return set(ch["rejudge_mie"]["rows"]), b / "INVENTORY_conflict_fix2_rejudge.jsonl"


def conflict_rejudge(a) -> None:
    ids, inv = _rejudge_sets(a.materialize)
    if not ids:
        print("no judge-visible-only conflict rows: nothing to re-judge");return
    out_dir = croot(a.materialize) / "judges_rejudge_new"
    work = []
    for s, d in F.CONFLICT_SETS.items():
        pk = d["eval"] / "judges/packets" / f"{a.system}_episodes.jsonl"
        if not pk.is_file():
            raise SystemExit(f"no old packets for {a.system} in {s}")
        eps = [e for e in F.jl(pk) if e["canonical_id"] in ids]
        if eps:
            work.append((d["tag"], eps))
    n = sum(len(e) for _, e in work)
    print(a.system, "packets to re-judge:", n)
    if not a.execute_paid:
        return
    step = f"judge_conflict_rejudge:{a.system}"
    paid_gate(a, step, "luna_judges_conflict", a.usd_cap)
    sys.path[:0] = [str(F.FC), str(F.V4 / "final_eval_assembly")]
    import secondary
    led = []
    for tag, eps in work:
        d = out_dir / tag;d.mkdir(parents=True, exist_ok=True)
        ep = d / f"{a.system}_episodes_rejudge.jsonl";out = d / f"{a.system}_mie.jsonl"
        if out.exists():
            raise FileExistsError(out)
        ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
        secondary.HERE = d.resolve()
        secondary.main(["--task", "mie", "--episodes", str(ep), "--inventory", str(inv), "--backend", "luna", "--execute-paid",
                        "--usd-cap", str(a.usd_cap / max(1, len(work))), "--ledger", str(out.with_suffix(".ledger.jsonl")), "--out", str(out)])
        if len(F.jl(out)) != len(eps):
            raise RuntimeError(f"{tag} {a.system}: {len(F.jl(out))} verdicts for {len(eps)} packets")
        led.append(out.with_suffix(".ledger.jsonl"))
    print(step, "actual USD", A.close(a.ledger, step, led), "|", A.status(a.ledger))


def conflict_merge(a) -> None:
    """judges_rejudge_merged/<tag>/ = GT-fix-1 merged judges (symlinked) with re-judged MIE verdicts replaced."""
    ids, _ = _rejudge_sets(a.materialize)
    key = lambda v: (v["canonical_id"], v["response_sha256"])  # noqa: E731
    for s, d in F.CONFLICT_SETS.items():
        src, dst = d["judges_merged"], croot(a.materialize) / "judges_rejudge_merged" / d["tag"]
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if not p.name.endswith("_mie.jsonl") and not (dst / p.name).exists():
                (dst / p.name).symlink_to(p.resolve())
        for p in src.glob("*_mie.jsonl"):
            sid = p.name[: -len("_mie.jsonl")]
            if sid not in a.systems.split(","):
                continue
            old = F.jl(p);hit = [v for v in old if v["canonical_id"] in ids]
            if (dst / p.name).exists():
                continue
            if not hit:
                (dst / p.name).symlink_to(p.resolve());continue
            newf = croot(a.materialize) / "judges_rejudge_new" / d["tag"] / f"{sid}_mie.jsonl"
            if not newf.is_file():
                raise SystemExit(f"missing re-judge verdicts {newf}")
            new = {key(v): v for v in F.jl(newf)}
            (dst / p.name).write_text("".join(json.dumps(new[key(v)] if v["canonical_id"] in ids else v) + "\n" for v in old))
            print(d["tag"], sid, "replaced", len(hit))


# ----------------------------------------------------------------------------------------------- main
def main_runs(sid: str, materialize: bool) -> list[tuple[str, str, Path]]:
    base = mroot(materialize) / "runs" / sid
    out = []
    for d in sorted(base.glob("*__*")) if base.is_dir() else []:
        mode, arm = d.name.split("__")
        run = d / "out"
        if (run / "eval_report.json").is_file():
            out.append((mode, arm, run))
    return out


def main_rel(materialize: bool) -> Path:
    return F.out_root(materialize) / F.MAIN_BUILD / "RELEASE_TEST_v4q_fix2"


def descriptor(sid: str, task: str, mode: str, arm: str, cap: float, run: Path, materialize: bool, packet: Path | None = None) -> str:
    rel = main_rel(materialize)
    name = "holdout.jsonl" if mode == "retrieved" else f"{mode}/holdout.jsonl"
    d = {"work": "MAIN_FIX2", "system": sid, "task": task, "mode": mode, "arm": arm, "backend": "luna", "cap_usd": cap,
         "release_sha256": F.sha(rel / name), "run_report_sha256": F.sha(run / "eval_report.json"), "texts_sha256": F.sha(run / "texts.jsonl"),
         "reference_sha256": F.sha(F.KEY_ACTIONS_CANON) if task != "mie" else F.sha(F.out_root(materialize) / F.MAIN_BUILD / "INVENTORY_TEST_v4q_APPROVED_fix2.jsonl"),
         "packet_sha256": F.sha(packet) if packet else None}
    return "PRELAUNCH " + json.dumps(d, sort_keys=True, separators=(",", ":"))


def approval_ok(line: str, materialize: bool) -> None:
    p = mroot(materialize) / "OWNER_APPROVAL.jsonl"
    if not p.is_file() or not any(json.loads(x).get("line") == line for x in p.read_text().splitlines() if x.strip()):
        raise A.Refused(f"REFUSED: PRELAUNCH line not approved in {p} (judge_fix2.py main-approve --line '<line>')")


def main_approve(a) -> None:
    if os.environ.get("OWNER_API_APPROVED") != "1":
        raise A.Refused("OWNER_API_APPROVED=1 required to record an approval")
    line = a.line.strip()
    if not line.startswith("PRELAUNCH {") or '"work":"MAIN_FIX2"' not in line:
        raise SystemExit("not a MAIN_FIX2 PRELAUNCH line")
    p = mroot(a.materialize) / "OWNER_APPROVAL.jsonl";p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"owner_approved": True, "line": line, "line_sha256": hashlib.sha256(line.encode()).hexdigest()}) + "\n")
    print("recorded in", p)


def bridge(materialize: bool):
    sys.path[:0] = [str(F.FC), str(F.V4 / "final_eval_assembly"), str(F.V4 / "g0_redesign_v1/rebuild_v20"), str(F.V4 / "ablation_ckpt_eval_ag_v4q")]
    import acute_bridge_ae as BR
    BR.configure(str(main_rel(materialize)))       # canonical key actions (sha-checked) bound to the fix2 release
    BR.V.build_item = BR.build_item
    return BR, BR.V


def main_acute(a) -> None:
    runs = main_runs(a.system, a.materialize)
    if not runs:
        raise SystemExit(f"no completed delta runs for {a.system}")
    jd = mroot(a.materialize) / "judges";jd.mkdir(parents=True, exist_ok=True)
    for mode, arm, run in runs:
        out = jd / f"{a.system}__{mode}__{arm}_acute.jsonl"
        n = len(F.jl(run / "texts.jsonl"))
        cap = a.usd_cap if a.usd_cap is not None else round(max(0.02, n * MAIN_CAP_PER_1K["acute"] / 1000), 3)
        line = descriptor(a.system, "acute", mode, arm, cap, run, a.materialize)
        if a.backend != "luna":           # CPU test path (rule 'cue' / 'export'): never paid, no gates
            BR, V = bridge(a.materialize)
            V.main(["--run", str(run), "--setting", f"{arm},{mode},greedy", "--backend", a.backend, "--rubric", "v4", "--out", str(out)])
            print("wrote", out);continue
        if not a.execute_paid:
            print(line);continue
        approval_ok(line, a.materialize)
        step = f"judge_main:{a.system}:{mode}:{arm}:acute"
        paid_gate(a, step, "luna_judges_main", cap)
        BR, V = bridge(a.materialize)
        if out.exists():
            raise FileExistsError(out)
        V.main(["--run", str(run), "--setting", f"{arm},{mode},greedy", "--backend", "luna", "--rubric", "v4", "--out", str(out),
                "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))])
        print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def delta_config(materialize: bool, systems: list[str]) -> dict:
    import recompute_table2 as R
    cfgs = {"main": json.loads(R.BASE_CFG.read_text()), "abl": json.loads(R.ABL_CFG.read_text()),
            "s1": json.loads((F.V4 / "final_eval_assembly/ag_v4q/configs/models_v4q_final_api_claude_cap1024_excl2_ivl8sev_medfix1_p2e_p2f9_s1.json").read_text())}
    out = copy.deepcopy(cfgs["main"])
    out |= {"release": str(main_rel(materialize)), "mie_inventory": str(F.out_root(materialize) / F.MAIN_BUILD / "INVENTORY_TEST_v4q_APPROVED_fix2.jsonl"),
            "note_fix2": "delta config (fix2): changed rows only; merged into the paper assemblies by recompute_table2.py", "models": []}
    jd = mroot(materialize) / "judges"
    for s in systems:
        mid = ASM.get(s, s)
        base = next((m for m in cfgs[CFG_OF.get(s, "main")]["models"] if m["id"] == mid), None)
        if base is None:
            print("# no base assembly entry for", s, "(not merged into Table 2)");continue
        runs = main_runs(s, materialize)
        if not runs:
            print("# no delta runs yet for", s);continue
        m = copy.deepcopy(base)
        m["runs"] = {"retrieved": [str(r) for md, _, r in runs if md == "retrieved"], "gt": [str(r) for md, _, r in runs if md != "retrieved"]}
        m["runs"]["acute"] = list(m["runs"]["retrieved"]);m["runs"]["nonacute"] = list(m["runs"]["retrieved"])
        m["acute_verdicts"] = [str(p) for p in sorted(jd.glob(f"{s}__*_acute.jsonl"))]
        m["warning_verdicts"] = [];m["severity_verdicts"] = [];m["mie_verdicts"] = []
        m["fix2_system"] = s
        out["models"].append(m)
    return out


def main_episodes(a) -> None:
    import recompute_table2 as R
    dcfg = delta_config(a.materialize, a.systems.split(","))
    d = mroot(a.materialize);d.mkdir(parents=True, exist_ok=True)
    (d / f"DELTA_CONFIG_{a.tag}.json").write_text(json.dumps(dcfg, indent=1) + "\n")
    if not dcfg["models"]:
        print("no delta runs: DELTA_CONFIG written without models");return
    E, collect = R.load_stack()
    R.DROP_FIELDS = ()                         # keep answer / item / acute_verdict: packets need them
    eps = R.assemble_delta(E, collect, dcfg)
    if len({m["id"] for m in dcfg["models"]}) != len(dcfg["models"]):
        raise SystemExit("two systems share an assembly id in one delta config (e.g. ours_4b_p2e and ours_4b_p2e_s1): use separate --tag runs")
    with (d / f"DELTA_EPISODES_{a.tag}.jsonl").open("w") as f:
        for mid, rs in eps.items():
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", d / f"DELTA_EPISODES_{a.tag}.jsonl", {k: len(v) for k, v in eps.items()})


def main_secondary(a) -> None:
    d = mroot(a.materialize)
    mid = ASM.get(a.system, a.system)
    eps = [e for e in F.jl(d / f"DELTA_EPISODES_{a.tag}.jsonl") if e["model"] == mid]
    ep = d / "judges" / f"{a.system}_delta_episodes_{a.task}.jsonl";ep.parent.mkdir(parents=True, exist_ok=True)
    out = d / "judges" / f"{a.system}_delta_{a.task}.jsonl"
    inv = F.out_root(a.materialize) / F.MAIN_BUILD / "INVENTORY_TEST_v4q_APPROVED_fix2.jsonl"
    n = sum(1 for e in eps if e.get("acute"))
    cap = a.usd_cap if a.usd_cap is not None else round(max(0.02, n * MAIN_CAP_PER_1K[a.task] / 1000), 3)
    if not a.execute_paid:
        print(f"# {a.system} {a.task}: {n} acute delta episodes; cap {cap}; then recompute_table2.py --extra-verdicts {mid}:{a.task}_verdicts={out}");return
    step = f"judge_main:{a.system}:{a.task}"
    paid_gate(a, step, "luna_judges_main", cap)
    if ep.exists() or out.exists():
        raise FileExistsError(out)
    ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
    sys.path[:0] = [str(F.FC), str(F.V4 / "final_eval_assembly")]
    import secondary
    secondary.HERE = (d / "judges").resolve()
    args = ["--task", a.task, "--episodes", str(ep), "--backend", "luna", "--execute-paid", "--usd-cap", str(cap),
            "--ledger", str(out.with_suffix(".ledger.jsonl")), "--out", str(out)] + (["--inventory", str(inv)] if a.task == "mie" else [])
    secondary.main(args)
    print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def main_rejudge_mie(a) -> None:
    import recompute_table2 as R
    b = F.out_root(a.materialize) / F.MAIN_BUILD
    ch = json.loads((b / "CHANGED_ROWS_main_fix2.json").read_text())
    keys = {tuple(x) for x in ch["rejudge_mie"]["episodes_mode_sample"]}
    mid = ASM.get(a.system, a.system)
    asm = R.ABL_ASM if a.system.startswith("abl_") else R.BASE_ASM
    eps = []
    with (asm / "EPISODES.jsonl").open() as f:
        for line in f:
            if f'"model": "{mid}"' not in line:
                continue
            e = json.loads(line)
            if e["model"] == mid and e["cohort_role"] == "acute" and (e["mode"], e["local_id"]) in keys:
                eps.append(e)
    d = mroot(a.materialize) / "judges";d.mkdir(parents=True, exist_ok=True)
    out = d / f"{a.system}_rejudge_mie.jsonl"
    cap = a.usd_cap if a.usd_cap is not None else round(max(0.02, len(eps) * MAIN_CAP_PER_1K["mie"] / 1000), 3)
    print(f"# {a.system}: {len(eps)} old-answer episodes for MIE re-judge; cap {cap}; then recompute_table2.py --rejudge-mie {mid}={out}")
    if not a.execute_paid or not eps:
        return
    step = f"judge_main:{a.system}:rejudge_mie"
    paid_gate(a, step, "luna_judges_main", cap)
    ep = d / f"{a.system}_rejudge_episodes.jsonl"
    if ep.exists() or out.exists():
        raise FileExistsError(out)
    ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
    sys.path[:0] = [str(F.FC), str(F.V4 / "final_eval_assembly")]
    import secondary
    secondary.HERE = d.resolve()
    secondary.main(["--task", "mie", "--episodes", str(ep), "--inventory", str(b / "INVENTORY_TEST_v4q_APPROVED_fix2.jsonl"), "--backend", "luna",
                    "--execute-paid", "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl")), "--out", str(out)])
    print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["conflict-plan", "conflict-run", "conflict-rejudge", "conflict-merge", "main-acute", "main-approve",
                                   "main-episodes", "main-secondary", "main-rejudge-mie"])
    p.add_argument("--system");p.add_argument("--systems", default=",".join(CONFLICT_SYSTEMS));p.add_argument("--task", choices=["acute", "severity", "mie"])
    p.add_argument("--backend", default="luna", choices=["luna", "cue", "export"]);p.add_argument("--line")
    p.add_argument("--execute-paid", action="store_true");p.add_argument("--ledger", type=Path);p.add_argument("--usd-cap", type=float)
    p.add_argument("--materialize", action="store_true");p.add_argument("--tag", default="main", help="delta config/episodes tag (s1 needs its own)")
    a = p.parse_args(argv)
    {"conflict-plan": conflict_plan, "conflict-run": conflict_run, "conflict-rejudge": conflict_rejudge, "conflict-merge": conflict_merge,
     "main-acute": main_acute, "main-approve": main_approve, "main-episodes": main_episodes, "main-secondary": main_secondary,
     "main-rejudge-mie": main_rejudge_mie}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
