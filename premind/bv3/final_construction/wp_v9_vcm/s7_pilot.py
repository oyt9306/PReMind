"""P3 production pilot (`--profile prod --pilot N`): measured cost, stop rule and the pilot report.

check   after author (+ precheck + delta retry): mean measured author-stage tokens per pilot family + the cost_v9
        estimate of every later stage = projected tokens per family; > PROD_TOKENS_PER_FAMILY_MAX writes
        s1/PILOT_STOP.json and gpt_client.guard then refuses every further paid call of this profile.
override  `--resume-after-stop REASON --owner-instruction TEXT` writes s1/PILOT_STOP_OVERRIDE.json (write-once, bound
        to the sha of this PILOT_STOP.json); guard then allows paid calls again. The stop rule itself is unchanged.
report  tokens per stage per family (ledger), author-delta / review-revise rates, acceptance, safety coverage.
Ledger tags: <run_id>::<stage>::<family_id>[@rN][#aK|#retry1].
"""
from __future__ import annotations

from collections import defaultdict

from final_construction.wp_v9_vcm import common as cm

AUTHOR_STAGES = ("author", "author_delta")


def ledger_by_family(cfg: cm.Cfg) -> dict[str, dict[str, int]]:
    import json
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if not cfg.usage_log.exists():
        return {}
    for line in cfg.usage_log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        parts = r["tag"].split("::")
        if len(parts) < 3 or parts[0] != cfg.run_id:
            continue
        fid = parts[2].split("@")[0].split("#")[0]
        out[fid][parts[1]] += int(r["total_tokens"] or 0)
    return {k: dict(v) for k, v in out.items()}


def check(cfg: cm.Cfg) -> dict:
    """Projected mean paid tokens per family after the author stage; writes the stop file when over budget."""
    from final_construction.wp_v9_vcm import cost_v9
    fams = [f["family_id"] for f in cm.plan_raw(cfg)["families"]]
    led = ledger_by_family(cfg)
    measured = [sum(led.get(f, {}).get(s, 0) for s in AUTHOR_STAGES) for f in fams]
    author_mean = sum(measured) / max(1, len(measured))
    rest = cost_v9.production_rest_per_family()
    projected = author_mean + rest
    rep = {"pilot_families": len(fams), "author_stage_mean_tokens": round(author_mean), "rest_estimate": round(rest),
           "projected_tokens_per_family": round(projected), "limit": cm.PROD_TOKENS_PER_FAMILY_MAX,
           "stop": projected > cm.PROD_TOKENS_PER_FAMILY_MAX}
    out = cfg.p("s1", "PILOT_COST_after_author.json")
    if not out.exists():
        cm.write_once(out, rep)
    if rep["stop"] and not cfg.p("s1", "PILOT_STOP.json").exists():
        cm.write_once(cfg.p("s1", "PILOT_STOP.json"), rep)
    return rep


def write_override(cfg: cm.Cfg, reason: str, instruction: str) -> dict:
    """Deliberate owner override of every fired stop file (PILOT_STOP, PROD_STOP_chunk<KK>) that has none yet
    (write-once <stem>_OVERRIDE.json bound to the stop file's sha); returns the latest override."""
    import datetime
    from final_construction.wp_v9_vcm import gpt_client
    d = cfg.p("s1")
    stops = sorted(p for p in d.glob("*STOP*.json") if not p.stem.endswith("_OVERRIDE")) if d.exists() else []
    if not stops:
        raise RuntimeError("--resume-after-stop: no stop file - nothing to override")
    if not reason.strip() or not instruction.strip():
        raise RuntimeError("--resume-after-stop needs a non-empty reason and --owner-instruction")
    for stop in gpt_client.open_stops(cfg):
        path = stop.with_name(stop.stem + "_OVERRIDE.json")
        if not path.exists():
            cm.write_once(path, {"reason": reason, "owner_instruction": instruction, "stop_sha256": cm.sha_file(stop),
                                 "stop_file": stop.name, "stop_report": cm.read(stop), "stop_rule_kept": True,
                                 "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    stop = stops[-1]
    doc = cm.read(stop.with_name(stop.stem + "_OVERRIDE.json"))
    if doc["stop_sha256"] != cm.sha_file(stop):
        raise RuntimeError(f"{stop.stem}_OVERRIDE.json is bound to a different {stop.name}")
    return doc


def report(cfg: cm.Cfg) -> dict:
    from final_construction.wp_v9_vcm import s1_prod
    fams = [f["family_id"] for f in cm.plan_raw(cfg)["families"]]
    led = ledger_by_family(cfg)
    stages = sorted({s for f in fams for s in led.get(f, {})})
    per_stage = {s: round(sum(led.get(f, {}).get(s, 0) for f in fams) / max(1, len(fams))) for s in stages}
    gate_p = cfg.p("s5_final", "gate.json")
    gate = cm.read(gate_p) if gate_p.exists() else {}
    adir = cfg.p("s2", "author_rev2")
    rep = {"pilot_families": len(fams), "tokens_per_family_by_stage": per_stage,
           "tokens_per_family_total": sum(per_stage.values()), "limit": cm.PROD_TOKENS_PER_FAMILY_MAX,
           "author_delta_rate": round(sum("author_delta" in led.get(f, {}) for f in fams) / max(1, len(fams)), 4),
           "review_revise_rate": round(len([p for p in adir.glob("*.json") if p.name != "CARRY_FORWARD.json"])
                                       / max(1, len(fams)), 4) if adir.exists() else 0.0,
           "gate": {k: gate.get(k) for k in ("rates", "counts", "gate_pass", "yield", "safety_coverage")},
           "risk_targets": s1_prod.TARGET_UNSAFE, "stopped": cfg.p("s1", "PILOT_STOP.json").exists(),
           "stop_overridden": cfg.p("s1", "PILOT_STOP_OVERRIDE.json").exists()}
    out = cfg.p("s7", "PILOT_REPORT.json")
    if not out.exists():
        cm.write_once(out, rep)
    return rep
