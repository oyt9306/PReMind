"""fix2b main-test Luna judges (acute v4 / severity v2 / MIE v1, frozen operators). Versioned companion of judge_fix2.py
(imported unchanged) with every reference re-bound to the fix2b build. Imports no API client.

Why a new file: judge_fix2.py main-* is bound to fix2 (RELEASE_TEST_v4q_fix2, INVENTORY_..._fix2, CHANGED_ROWS_main_fix2,
runs under main_eval_fix2_v1/runs) and its acute bridge uses the OLD canonical key actions (KEY_ACTIONS_CANONICAL_v4q,
acute_bridge_ae.configure() cannot take another file). Here:
  release     release_test_v4q_fix2b/RELEASE_TEST_v4q_fix2b
  key actions release_test_v4q_fix2b/KEY_ACTIONS_CANONICAL_v4q_fix2b.jsonl (sha-pinned by FIX2B_BINDING.json; bound to the
              release by canonical id + source_image_sha256 exactly as acute_bridge_ae.configure() binds the v4q file)
  inventory   release_test_v4q_fix2b/INVENTORY_TEST_v4q_APPROVED_fix2b.jsonl
  changed     release_test_v4q_fix2b/CHANGED_ROWS_main_fix2_fix2b_union.json
  runs        main_eval_fix2b_v1/merged/<sid>/<mode>__<arm>/out  (overlay_fix2b.py output: fix2 delta/full + fix2b rows)
  outputs     main_eval_fix2b_v1/{judges/, DELTA_CONFIG_<tag>.json, DELTA_EPISODES_<tag>.jsonl, OWNER_APPROVAL.jsonl}
Acute is judged on the retrieved runs only (both arms), as every paper acute file (final_ours_al/*/judges, PLAN
narrow_test_judges); --modes all also judges gt modes.
Systems: delta systems (changed rows) and FULL systems ours_4b_p2e_nomie / sft_only_4b (all rows; no base assembly; their
assembly entry is the main-config ours_4b entry with id / label / lora replaced).

  status                                   CPU: per system, merged runs present vs queue jobs
  main-acute --system S [--backend luna|cue|export] [--modes retrieved|all]   PRELAUNCH line per run (free) / paid run
  main-approve --line '<PRELAUNCH line>'   owner (OWNER_API_APPROVED=1)
  main-episodes --systems a,b --tag T      CPU: DELTA_CONFIG_T.json + DELTA_EPISODES_T.jsonl (frozen collect, fix2b bridge)
  main-secondary --system S --task severity|mie [--tag T] [--backend luna|export]   free: count + cap; paid with luna
  main-rejudge-mie --system S              free: count + cap; paid: MIE of NOT re-inferred episodes whose inventory row changed
  main-rejudge-acute --system S [--task acute|severity]   free: count + cap (+PRELAUNCH); paid: acute v4 on the OLD answers of
                                           not-re-inferred acute rows whose judge-visible key actions changed (20 rows/arm);
                                           then severity v2 on those that became contraindicated. Other rejudge_acute rows
                                           (gold / general-knowledge rule / hazard_part only) have an unchanged acute item
                                           and an unchanged answer: their old verdicts are reused.
Test overrides (CPU only; paid steps refuse them): --root <dir> (outputs), --runs <dir> (merged runs root).
Every paid call: --execute-paid --ledger L --usd-cap X, owner gates, OPENAI_API_KEY in env (never printed).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import api_common as A  # noqa: E402
import fix2_common as F  # noqa: E402
import judge_fix2 as J  # noqa: E402

B2 = F.V4 / "release_test_v4q_fix2b"
REL2 = B2 / "RELEASE_TEST_v4q_fix2b"
INV2 = B2 / "INVENTORY_TEST_v4q_APPROVED_fix2b.jsonl"
KA2 = B2 / "KEY_ACTIONS_CANONICAL_v4q_fix2b.jsonl"
CH2 = B2 / "CHANGED_ROWS_main_fix2_fix2b_union.json"
MEB = F.V4 / "main_eval_fix2b_v1"
QUEUE_B = MEB / "QUEUE_main_fix2b.json"
FULL = {"ours_4b_p2e_nomie": F.V4 / "test_p2e_4b_nomie_fix2", "sft_only_4b": F.V4 / "test_sft_only_4b_fix2"}
LABEL = {"ours_4b_p2e_nomie": "M3 control (no MIE reward)", "sft_only_4b": "SFT-only"}
S1_CFG = F.V4 / "final_eval_assembly/ag_v4q/configs/models_v4q_final_api_claude_cap1024_excl2_ivl8sev_medfix1_p2e_p2f9_s1.json"
S1_ASM = F.V4 / "final_eval_assembly/ag_v4q/assembled_final_api_claude_cap1024_excl2_ivl8sev_medfix1_p2e_p2f9_s1"
CAP = J.MAIN_CAP_PER_1K
WORK_TAG = "MAIN_FIX2B"


def ka2_sha() -> str:
    """sha of the fix2b key actions; refuses unless the file, the inventory and the release equal the FIX2B_BINDING pins."""
    pin = json.loads((B2 / "FIX2B_BINDING.json").read_text())
    h = F.sha(KA2)
    if h != pin["key_actions_sha256"] or F.sha(INV2) != pin["mie_inventory_sha256"]:
        raise ValueError("fix2b key actions / inventory differ from FIX2B_BINDING.json")
    for m, s in pin["release_sha256"].items():
        if F.sha(REL2 / ("holdout.jsonl" if m == "retrieved" else f"{m}/holdout.jsonl")) != s:
            raise ValueError("fix2b release differs from FIX2B_BINDING.json: " + m)
    return h


# ----------------------------------------------------------------------------------------------- bridge (fix2b key actions)
def patch_bridge():
    """acute_bridge_ae with configure() re-bound to the fix2b key actions (also for the assembler's E.A4.configure)."""
    for p in (str(F.FC), str(F.V4 / "final_eval_assembly"), str(F.V4 / "g0_redesign_v1/rebuild_v20"), str(F.V4 / "ablation_ckpt_eval_ag_v4q")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import acute_bridge_ae as BR
    if getattr(BR, "_fix2b", False):
        return BR
    orig, h = BR.configure, ka2_sha()

    def configure(release=None, key_actions=None):
        if key_actions is not None:
            raise ValueError("fix2b bridge: key actions are fixed to " + str(KA2))
        cat = orig(release)                       # v4q freeze check + catalog (raises on drift)
        old = BR.REFS
        rr = F.jl(KA2)
        refs = {r["row_id"]: r for r in rr}
        if len(refs) != len(rr) or set(refs) != set(old):
            raise ValueError("fix2b key actions: duplicate ids or id set differs from the canonical v4q set")
        for cid, r in refs.items():
            matches = [rs[cid] for rs in BR.CAT.canon.values() if cid in rs]
            if not matches or any(x["evaluator"]["source_image_sha256"] != r["source_image_sha256"] for x in matches):
                raise ValueError("unbound/mismatched fix2b reference " + cid)
        BR.REFS = refs
        BR.KEY_ACTIONS = (KA2, h)
        BR.V.KEY_ACTIONS = (KA2, h)
        BR.V.rubric_text("v4")
        return cat
    BR.configure = configure
    BR.V.build_item = BR.build_item
    BR._fix2b = True
    return BR


def bridge(release: Path = REL2):
    BR = patch_bridge()
    BR.configure(str(release))
    return BR, BR.V


# ----------------------------------------------------------------------------------------------- paths
def roots(a) -> tuple[Path, Path]:
    root = Path(a.root) if a.root else MEB
    runs = Path(a.runs) if a.runs else MEB / "merged"
    return root, runs


def paid_ok_paths(a) -> None:
    if a.root or a.runs:
        raise A.Refused("REFUSED: --root / --runs are CPU-test overrides; paid steps use the default fix2b paths")


def runs_of(sid: str, runs_root: Path) -> list[tuple[str, str, Path]]:
    base = runs_root / sid
    out = []
    for d in sorted(base.glob("*__*")) if base.is_dir() else []:
        parts = d.name.split("__")
        if len(parts) != 2 or parts[0] not in F.MAIN_MODES or parts[1] not in ("memory_call", "passive"):
            continue                                   # e.g. '<mode>__<arm>.failed_*' dirs are never used
        if (d / "out" / "eval_report.json").is_file():
            out.append((parts[0], parts[1], d / "out"))
    return out


def queue_jobs() -> list[dict]:
    return json.loads(QUEUE_B.read_text())["jobs"]


def descriptor(sid: str, task: str, mode: str, arm: str, cap: float, run: Path, n: int) -> str:
    name = "holdout.jsonl" if mode == "retrieved" else f"{mode}/holdout.jsonl"
    d = {"work": WORK_TAG, "system": sid, "task": task, "mode": mode, "arm": arm, "backend": "luna", "cap_usd": cap, "n_items": n,
         "release_sha256": F.sha(REL2 / name), "run_report_sha256": F.sha(run / "eval_report.json"), "texts_sha256": F.sha(run / "texts.jsonl"),
         "reference_sha256": F.sha(KA2) if task != "mie" else F.sha(INV2)}
    return "PRELAUNCH " + json.dumps(d, sort_keys=True, separators=(",", ":"))


def approval_ok(root: Path, line: str) -> None:
    p = root / "OWNER_APPROVAL.jsonl"
    if not p.is_file() or not any(json.loads(x).get("line") == line for x in p.read_text().splitlines() if x.strip()):
        raise A.Refused(f"REFUSED: PRELAUNCH line not approved in {p} (judge_fix2b.py main-approve --line '<line>')")


def paid_gate(a, step: str, cap: float) -> None:
    paid_ok_paths(a)
    A.require_paid(a)
    for k in ("TEST_FREEZE_APPROVED", "OWNER_API_APPROVED", "G0_QA_KEYS_REVIEWED"):
        if os.environ.get(k) != "1":
            raise A.Refused(f"REFUSED: owner gate {k}=1 required")
    A.require_key("openai")
    A.reserve(a.ledger, step, "openai", cap, group="luna_judges_main")


def cap_for(task: str, n: int, given: float | None) -> float:
    return given if given is not None else round(max(0.02, n * CAP[task] / 1000), 3)


# ----------------------------------------------------------------------------------------------- commands
def cmd_status(a) -> None:
    _, runs_root = roots(a)
    by = {}
    for j in queue_jobs():
        by.setdefault(j["system"], []).append(f"{j['mode']}__{j['arm']}")
    for sid, want in by.items():
        have = {f"{m}__{r}" for m, r, _ in runs_of(sid, runs_root)}
        print(f"{sid:22s} merged {len(have & set(want))}/{len(want)}" + ("" if have >= set(want) else f"  missing {sorted(set(want) - have)}"))


def cmd_approve(a) -> None:
    if os.environ.get("OWNER_API_APPROVED") != "1":
        raise A.Refused("OWNER_API_APPROVED=1 required to record an approval")
    line = a.line.strip()
    if not line.startswith("PRELAUNCH {") or f'"work":"{WORK_TAG}"' not in line:
        raise SystemExit(f"not a {WORK_TAG} PRELAUNCH line")
    root, _ = roots(a)
    p = root / "OWNER_APPROVAL.jsonl";p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"owner_approved": True, "line": line, "line_sha256": F.sha_text(line)}) + "\n")
    print("recorded in", p)


def cmd_acute(a) -> None:
    root, runs_root = roots(a)
    runs = [r for r in runs_of(a.system, runs_root) if a.modes == "all" or r[0] == "retrieved"]
    if not runs:
        raise SystemExit(f"no merged runs for {a.system} under {runs_root}")
    BR, V = bridge()
    jd = root / "judges";jd.mkdir(parents=True, exist_ok=True)
    for mode, arm, run in runs:
        out = jd / f"{a.system}__{mode}__{arm}_acute.jsonl"
        n = len(V.run_items(run, (arm, mode, "greedy")))
        cap = cap_for("acute", n, a.usd_cap)
        line = descriptor(a.system, "acute", mode, arm, cap, run, n)
        if a.backend != "luna":                         # CPU test path (cue / export): never paid, no gates
            V.main(["--run", str(run), "--setting", f"{arm},{mode},greedy", "--backend", a.backend, "--rubric", "v4", "--out", str(out)])
            continue
        if not a.execute_paid:
            print(line);continue
        approval_ok(root, line)
        step = f"judge_main_fix2b:{a.system}:{mode}:{arm}:acute"
        if out.exists():
            raise FileExistsError(out)
        paid_gate(a, step, cap)
        V.main(["--run", str(run), "--setting", f"{arm},{mode},greedy", "--backend", "luna", "--rubric", "v4", "--out", str(out),
                "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))])
        print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def base_entry(sid: str) -> dict | None:
    import recompute_table2 as R
    cfgs = {"main": R.BASE_CFG, "abl": R.ABL_CFG, "s1": S1_CFG}
    if sid in FULL:
        m = copy.deepcopy(next(x for x in json.loads(R.BASE_CFG.read_text())["models"] if x["id"] == "ours_4b"))
        plan = json.loads((FULL[sid] / "PLAN.json").read_text())
        for k in ("table3_role", "checkpoint_steps", "selection_status", "checkpoint_selection"):
            m.pop(k, None)
        m |= {"id": sid, "label": LABEL[sid], "lora": plan["checkpoint"], "recipe": "fix2b full run", "fix2b_full": True}
        return m
    mid = J.ASM.get(sid, sid)
    cfg = json.loads(cfgs[J.CFG_OF.get(sid, "main")].read_text())
    return next((x for x in cfg["models"] if x["id"] == mid), None)


def delta_config(systems: list[str], root: Path, runs_root: Path) -> dict:
    import recompute_table2 as R
    out = copy.deepcopy(json.loads(R.BASE_CFG.read_text()))
    out |= {"release": str(REL2), "mie_inventory": str(INV2), "models": [],
            "note_fix2b": "delta config (fix2b): merged runs (overlay_fix2b.py); fix2b key actions via judge_fix2b.patch_bridge"}
    jd = root / "judges"
    for s in systems:
        base = base_entry(s)
        if base is None:
            print("# no base assembly entry for", s);continue
        runs = runs_of(s, runs_root)
        if not runs:
            print("# no merged runs yet for", s);continue
        m = copy.deepcopy(base)
        m["runs"] = {"retrieved": [str(r) for md, _, r in runs if md == "retrieved"], "gt": [str(r) for md, _, r in runs if md != "retrieved"]}
        m["runs"]["acute"] = list(m["runs"]["retrieved"]);m["runs"]["nonacute"] = list(m["runs"]["retrieved"])
        m["acute_verdicts"] = [str(p) for p in sorted(jd.glob(f"{s}__*_acute.jsonl"))]
        m["warning_verdicts"] = [];m["severity_verdicts"] = [];m["mie_verdicts"] = []
        m["fix2_system"] = s
        out["models"].append(m)
    return out


def cmd_episodes(a) -> None:
    import recompute_table2 as R
    root, runs_root = roots(a)
    patch_bridge()
    dcfg = delta_config(a.systems.split(","), root, runs_root)
    if len({m["id"] for m in dcfg["models"]}) != len(dcfg["models"]):
        raise SystemExit("two systems share an assembly id in one delta config: use separate --tag runs")
    root.mkdir(parents=True, exist_ok=True)
    for p in (root / f"DELTA_CONFIG_{a.tag}.json", root / f"DELTA_EPISODES_{a.tag}.jsonl"):
        if p.exists() and not a.root:
            raise FileExistsError(p)
    (root / f"DELTA_CONFIG_{a.tag}.json").write_text(json.dumps(dcfg, indent=1) + "\n")
    if not dcfg["models"]:
        print("no runs: DELTA_CONFIG written without models");return
    E, collect = R.load_stack()
    R.DROP_FIELDS = ()                                 # keep answer / item / acute_verdict: packets need them
    eps = R.assemble_delta(E, collect, dcfg)
    with (root / f"DELTA_EPISODES_{a.tag}.jsonl").open("w") as f:
        for rs in eps.values():
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stat = {k: {"episodes": len(v), "acute_judged": sum(1 for r in v if r.get("acute") and r.get("acute_state") is not None),
                "acute_unjudged": sum(1 for r in v if r.get("acute") and r.get("acute_state") is None and r["mode"] == "retrieved")} for k, v in eps.items()}
    print("wrote", root / f"DELTA_EPISODES_{a.tag}.jsonl", json.dumps(stat))


def _secondary(task: str, ep: Path, out: Path, cap: float, backend: str, inv: Path | None, ledger_cap: bool) -> None:
    sys.path[:0] = [str(F.FC), str(F.V4 / "final_eval_assembly")]
    import secondary
    secondary.HERE = out.parent.resolve()
    args = ["--task", task, "--episodes", str(ep), "--backend", backend, "--out", str(out)] + (["--inventory", str(inv)] if task == "mie" else [])
    if backend == "luna":
        args += ["--execute-paid", "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))]
    secondary.main(args)


def cmd_secondary(a) -> None:
    root, _ = roots(a)
    mid = J.ASM.get(a.system, a.system)
    eps, seen = [], set()
    for e in F.jl(root / f"DELTA_EPISODES_{a.tag}.jsonl"):     # one per answer (secondary.packets dedups the same way)
        k = (e["canonical_id"], e["mode"], e["arm"], e["kind"], e.get("sample_idx"), e["response_sha256"])
        if e["model"] == mid and e.get("acute") and e["mode"] == "retrieved" and k not in seen:
            seen.add(k);eps.append(e)
    if a.task == "severity":
        n = sum(1 for e in eps if (e.get("acute_verdict") or {}).get("state") == "contraindicated")
        unj = sum(1 for e in eps if e.get("acute_state") is None)
    else:
        inv = {x["evaluator_row_id"]: x for x in F.jl(INV2)}
        n = sum(1 for e in eps if e["canonical_id"] in inv and e["mode"] in inv[e["canonical_id"]]["modes"])
        unj = 0
    cap = cap_for(a.task, n, a.usd_cap)
    jd = root / "judges";jd.mkdir(parents=True, exist_ok=True)
    ep = jd / f"{a.system}_delta_episodes_{a.task}.jsonl";out = jd / f"{a.system}_delta_{a.task}.jsonl"
    print(f"# {a.system} {a.task}: {n} packets (acute retrieved episodes {len(eps)}, unjudged acute {unj}); cap {cap}")
    if a.task == "severity" and unj:
        print(f"# WARNING {a.system}: {unj} acute episodes have no acute verdict yet: run main-acute + main-episodes first")
    if a.backend == "export":
        ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
        _secondary(a.task, ep, out, cap, "export", INV2, False);return
    if not a.execute_paid:
        return
    if unj:
        raise SystemExit("acute verdicts incomplete: refusing a paid severity / MIE run on a partial episode set")
    step = f"judge_main_fix2b:{a.system}:{a.task}"
    if ep.exists() or out.exists():
        raise FileExistsError(out)
    paid_gate(a, step, cap)
    ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
    _secondary(a.task, ep, out, cap, "luna", INV2, True)
    print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def base_asm_of(sid: str) -> Path | None:
    import recompute_table2 as R
    if sid in FULL:
        return None
    return R.ABL_ASM if sid.startswith("abl_") else (S1_ASM if sid == "ours_4b_p2e_s1" else R.BASE_ASM)


def base_episodes(sid: str, keys: set, role_acute: bool = True) -> list[dict]:
    asm = base_asm_of(sid)
    if asm is None:
        return []
    mid = J.ASM.get(sid, sid)
    out = []
    with (asm / "EPISODES.jsonl").open() as f:
        for line in f:
            if f'"model": "{mid}"' not in line:
                continue
            e = json.loads(line)
            if e["model"] == mid and (not role_acute or e["cohort_role"] == "acute") and (e["mode"], e["local_id"]) in keys:
                out.append(e)
    return out


def cmd_rejudge_mie(a) -> None:
    root, _ = roots(a)
    ch = json.loads(CH2.read_text())
    keys = {tuple(x) for x in ch["rejudge_mie"]["episodes_mode_sample"]}
    eps = base_episodes(a.system, keys)
    inv = {x["evaluator_row_id"]: x for x in F.jl(INV2)}
    n = sum(1 for e in eps if e["canonical_id"] in inv and e["mode"] in inv[e["canonical_id"]]["modes"])
    cap = cap_for("mie", n, a.usd_cap)
    jd = root / "judges";jd.mkdir(parents=True, exist_ok=True)
    out = jd / f"{a.system}_rejudge_mie.jsonl";ep = jd / f"{a.system}_rejudge_episodes.jsonl"
    print(f"# {a.system}: {len(eps)} old-answer acute episodes in rejudge_mie ({n} in the fix2b inventory); cap {cap}")
    if a.backend == "export" and eps:
        ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
        _secondary("mie", ep, out, cap, "export", INV2, False);return
    if not a.execute_paid or not n:
        return
    step = f"judge_main_fix2b:{a.system}:rejudge_mie"
    if ep.exists() or out.exists():
        raise FileExistsError(out)
    paid_gate(a, step, cap)
    ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
    _secondary("mie", ep, out, cap, "luna", INV2, True)
    print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def visible_changed_ids() -> set[str]:
    """Canonical ids whose judge-visible acute item part (types, visible_hazard, actions after LABEL_REMAP, flags) changed."""
    BR, V = bridge()
    old = {r["row_id"]: r for r in F.jl(F.KEY_ACTIONS_CANON)}
    vis = lambda k: (k["types"], k["visible_hazard"], [V._act(x) for x in k["actions"]], k["flags"])  # noqa: E731
    return {cid for cid, k in BR.REFS.items() if vis(k) != vis(old[cid])}


def rejudge_acute_targets(sid: str) -> dict[tuple[str, str], list[tuple[Path, list[str]]]]:
    """(mode, arm) -> [(base run, row ids)] of not-re-inferred acute rows with a judge-visible key-action change."""
    import recompute_table2 as R
    ch = json.loads(CH2.read_text())
    keys = {tuple(x) for x in ch["rejudge_acute"]["episodes_mode_sample"] if x[0] == "retrieved"}
    vis = visible_changed_ids()
    eps = [e for e in base_episodes(sid, keys) if e["canonical_id"] in vis]
    by: dict = {}
    for e in eps:
        by.setdefault((e["mode"], e["arm"], e["run"]), set()).add(e["local_id"])
    del R
    out: dict = {}
    for (mode, arm, run), ids in sorted(by.items()):
        out.setdefault((mode, arm), []).append((Path(run), sorted(ids)))
    return out


def make_view(run: Path, ids: list[str], dst: Path) -> Path:
    """Row-restricted read-only view of an existing run (eval_report.json copied; records/texts filtered); same answers."""
    dst.mkdir(parents=True, exist_ok=True)
    keep = set(ids)
    for fn in ("records.jsonl", "texts.jsonl"):
        p = dst / fn
        if not p.exists():
            p.write_text("".join(json.dumps(r) + "\n" for r in F.jl(run / fn) if r["row_id"] in keep))
    if not (dst / "eval_report.json").exists():
        rep = json.loads((run / "eval_report.json").read_text())
        rep["meta"]["fix2b_rejudge_view"] = {"source_run": str(run), "row_ids": sorted(keep), "texts_sha256": F.sha(run / "texts.jsonl")}
        (dst / "eval_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    return dst


def cmd_rejudge_acute(a) -> None:
    root, _ = roots(a)
    jd = root / "judges";jd.mkdir(parents=True, exist_ok=True)
    tg = rejudge_acute_targets(a.system)
    if not tg:
        print(f"# {a.system}: no base episodes to re-judge (full system or none in its base assembly)");return
    BR, V = bridge()
    if a.task == "acute":
        for (mode, arm), lst in sorted(tg.items()):
            for i, (run, ids) in enumerate(lst):
                view = make_view(run, ids, root / "rejudge_views" / a.system / f"{mode}__{arm}__{i}")
                n = len(V.run_items(view, (arm, mode, "greedy")))
                out = jd / f"{a.system}_rejudge_acute__{mode}__{arm}__{i}.jsonl"
                cap = cap_for("acute", n, a.usd_cap)
                line = descriptor(a.system, "rejudge_acute", mode, arm, cap, view, n)
                if a.backend != "luna":
                    V.main(["--run", str(view), "--setting", f"{arm},{mode},greedy", "--backend", a.backend, "--rubric", "v4", "--out", str(out)])
                    continue
                print(f"# {a.system} {mode}/{arm}: {n} old answers (base run {run.name}); cap {cap}")
                if not a.execute_paid:
                    print(line);continue
                approval_ok(root, line)
                step = f"judge_main_fix2b:{a.system}:{mode}:{arm}:rejudge_acute:{i}"
                if out.exists():
                    raise FileExistsError(out)
                paid_gate(a, step, cap)
                V.main(["--run", str(view), "--setting", f"{arm},{mode},greedy", "--backend", "luna", "--rubric", "v4", "--out", str(out),
                        "--usd-cap", str(cap), "--ledger", str(out.with_suffix(".ledger.jsonl"))])
                print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))
        return
    # severity on re-judged episodes that are now contraindicated (packets = base episode + new verdict + fix2b item)
    vs = {}
    for f in sorted(jd.glob(f"{a.system}_rejudge_acute__*.jsonl")):
        if f.name.endswith(".ledger.jsonl"):
            continue
        for v in F.jl(f):
            vs[(v["mode"], v["row_id"], v["arm"], v["kind"], int(v.get("sample_idx", 0)))] = v
    ch = json.loads(CH2.read_text())
    keys = {tuple(x) for x in ch["rejudge_acute"]["episodes_mode_sample"] if x[0] == "retrieved"}
    eps = []
    for e in base_episodes(a.system, keys, role_acute=True):
        v = vs.get((e["mode"], e["local_id"], e["arm"], e["kind"], int(e.get("sample_idx", 0))))
        if v is None or v.get("state") != "contraindicated":
            continue
        row = BR.CAT.row(e["mode"], e["local_id"])
        eps.append(dict(e, acute_verdict=v, acute_state=v["state"], item=BR.build_item(row, e["mode"])))
    cap = cap_for("severity", len(eps), a.usd_cap)
    ep = jd / f"{a.system}_rejudge_acute_episodes_severity.jsonl";out = jd / f"{a.system}_rejudge_acute_severity.jsonl"
    print(f"# {a.system}: {len(eps)} re-judged episodes now contraindicated -> severity; cap {cap} (re-judge verdict files: {len(vs)} verdicts)")
    if a.backend == "export":
        ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
        _secondary("severity", ep, out, cap, "export", None, False);return
    if not a.execute_paid or not eps:
        return
    step = f"judge_main_fix2b:{a.system}:rejudge_acute_severity"
    if ep.exists() or out.exists():
        raise FileExistsError(out)
    paid_gate(a, step, cap)
    ep.write_text("".join(json.dumps(e) + "\n" for e in eps))
    _secondary("severity", ep, out, cap, "luna", None, True)
    print(step, "actual USD", A.close(a.ledger, step, [out.with_suffix(".ledger.jsonl")]), "|", A.status(a.ledger))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["status", "main-acute", "main-approve", "main-episodes", "main-secondary", "main-rejudge-mie", "main-rejudge-acute"])
    p.add_argument("--system");p.add_argument("--systems", default="");p.add_argument("--task", choices=["acute", "severity", "mie"])
    p.add_argument("--backend", default="luna", choices=["luna", "cue", "export"]);p.add_argument("--line")
    p.add_argument("--modes", default="retrieved", choices=["retrieved", "all"])
    p.add_argument("--execute-paid", action="store_true");p.add_argument("--ledger", type=Path);p.add_argument("--usd-cap", type=float)
    p.add_argument("--tag", default="main");p.add_argument("--root");p.add_argument("--runs")
    a = p.parse_args(argv)
    if a.cmd == "main-rejudge-acute" and a.task is None:
        a.task = "acute"
    if a.execute_paid and a.backend != "luna":
        raise A.Refused("REFUSED: --execute-paid only with --backend luna")
    {"status": cmd_status, "main-acute": cmd_acute, "main-approve": cmd_approve, "main-episodes": cmd_episodes,
     "main-secondary": cmd_secondary, "main-rejudge-mie": cmd_rejudge_mie, "main-rejudge-acute": cmd_rejudge_acute}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
