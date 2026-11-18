"""Production retry round (owner 2026-10-03): units, retry spend, cost model, start gate and the retry check (CPU).

Units = the FINAL-rule trees: regen_pilot, regen_chunk01..03 ("@g2") and chunk_04..chunk_<last> ("g1"); chunk_01..03
and the pilot root are superseded and never retried. A unit's retry is round 2 (`--round 2 --retry`), paid tags
"<family>@r2" (chunks) / "<family>@g2@r2" (regen), so no consumed tag is ever reused.
Projection (concurrency-aware): projected = every paid token in the ledger + for each unfinished retry unit
  * not started (no s2/author_rev2/CARRY_FORWARD.json): expected revise share x its families x per-family retry cost;
  * started, unfinished (in flight; its spend is already in the ledger): max(estimate - its retry spend,
    INFLIGHT_MARGIN x estimate), with its actual revise count from CARRY_FORWARD.json.
Per-family retry cost = the measured mean (retry spend / revised families) over retry units checked under the current
PREREG; before any, the ledger-measured per-call sizes of the earlier review-driven rounds (@r2, g1):
revise_review (attempt 1) + REVIEWED_SHARE x (memtext incl. its retry rate + review_primary + adjudicate rate x
adjudicate). Revise share = measured over started units, else REVISE_SHARE_DEFAULT.
start_gate: the unit's first paid step (revise_review, before its CARRY_FORWARD.json start marker) is a NEW run: it is
refused while any stop file is open, and when the start projection exceeds cm.PROD_BUDGET_STOP it writes the
write-once s1/PROD_STOP_retry_start.json and refuses (owner override: --resume-after-stop). In-flight units are never
blocked by a stop (gpt_client.IN_FLIGHT_STAGES). retry_check (`prod_check --retry`, after finalize --retry) writes
s1/PROD_CHECK_retry_<unit>.json and, over budget or below coverage, s1/PROD_STOP_retry_<unit>.json.
Policy (owner 2026-10-03, option B): `retry_select` writes the write-once s1/RETRY_FAMILIES_v1.json = policy zero_or_dev
over the retry-eligible families of every unit: ALL zero-row families + ALL dev families with a fixable rejection, no
other train family. Every unit reads it (s2_revise_review.select_families); the PREREG and the prelaunch line bind its
sha. Without the file the full policy (every eligible family) applies. The projection then uses its exact counts.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

from final_construction.wp_v9_vcm import common as cm

logger = logging.getLogger(__name__)
INFLIGHT_MARGIN = 0.20
REVISE_SHARE_DEFAULT = 0.77        # dry-run selection over the 8 final units, 2026-10-03: 267 of 346 families
REVIEWED_SHARE = 0.72              # attempt-1 precheck-clean share of review-driven revisions (g1 round 2: 39/54)
START_STOP = "PROD_STOP_retry_start.json"
SELECTION_FILE = "RETRY_FAMILIES_v1.json"
POLICY = "zero_or_dev"


def _selection(cfg: cm.Cfg) -> dict | None:
    p = cfg.p("s1", SELECTION_FILE)
    return cm.read(p) if p.exists() else None


def selected(cfg: cm.Cfg) -> set[str] | None:
    """Families the retry may revise (union over units), or None = full policy (no selection file)."""
    doc = _selection(cfg)
    return None if doc is None else {f for u in doc["units"].values() for f in u["families"]}


def policy(cfg: cm.Cfg) -> str:
    doc = _selection(cfg)
    return "full" if doc is None else f"{doc['policy']} (s1/{SELECTION_FILE} sha256 {cm.sha_file(cfg.p('s1', SELECTION_FILE))})"


def write_selection(cfg: cm.Cfg) -> dict:
    """`retry_select --profile prod` (CPU, write-once): the zero_or_dev family list of every retry unit."""
    from final_construction.wp_v9_vcm import s2_revise_review
    out = cfg.p("s1", SELECTION_FILE)
    if out.exists():
        return {"status": "exists", **{k: v for k, v in cm.read(out).items() if k != "units"}}
    plan = {f["family_id"]: f for f in cm.read(cfg.p("s1", "plan.json"))["families"]}
    res, tot = {}, defaultdict(int)
    for u, uc in units(cfg).items():
        if not uc.p("s5_final", "gate.json").exists():
            continue
        sel = s2_revise_review.select_families(uc, policy=False)
        elig = sorted({f["family_id"] for f, *_ in sel["todo"]} | set(sel["memtext_only"]))
        zero = [f for f in elig if f in sel["zero_row"]]
        dev = [f for f in elig if plan[f]["split"] == "dev"]
        fams = sorted(set(zero) | set(dev))
        res[u] = {"families": fams, "zero_row": zero, "dev_fixable": dev, "eligible": len(elig),
                  "zero_row_all": len(sel["zero_row"]), "zero_row_unfixable": sel["unfixable"]}
        for k in ("families", "zero_row", "dev_fixable"):
            tot[k] += len(res[u][k])
        tot["eligible"] += len(elig)
        tot["zero_row_all"] += len(sel["zero_row"])
        tot["zero_row_unfixable"] += len(sel["unfixable"])
    doc = {"policy": POLICY, "rule": "ALL zero-row families + ALL dev families with a fixable rejection; no other "
                                     "train family (fixable = not general_rule_closed, not partial retrieval)",
           "plan_sha256": cm.sha_file(cfg.p("s1", "plan.json")), "totals": dict(tot), "units": res}
    cm.write_once(out, doc)
    return {k: v for k, v in doc.items() if k != "units"}


def check_cli(a: object) -> None:
    """`--retry` = the production round 2 of a FINAL unit: --chunk-regen T or --chunk K >= RETRY_FIRST_CHUNK."""
    if not getattr(a, "retry", False):
        return
    chunk, regen = getattr(a, "chunk", None), getattr(a, "chunk_regen", None)
    if (a.profile != "prod" or a.round not in (2, *cm.FIX_ROUNDS) or a.no_review_round or a.pilot is not None
            or (regen is None) == (chunk is None or chunk < cm.RETRY_FIRST_CHUNK)):
        raise SystemExit(f"--retry: --profile prod --round 2 (retry) or 3 (repair) with --chunk-regen pilot|1|2|3 or --chunk K >= "
                         f"{cm.RETRY_FIRST_CHUNK} (chunks 1-3 and the pilot root are superseded); no --no-review-round")


def prereg_section(cfg: cm.Cfg) -> dict:
    from final_construction.wp_v9_vcm import s2_revise_review, s5_retry
    return {"units": "regen_pilot, regen_chunk01-03, chunk_04..last", "round": 2, "tags": "@r2 / @g2@r2",
            "revise": "families with a rejected author-fixable row (not general_rule_closed, not "
                      f"{list(s2_revise_review.NOT_AUTHOR_FIXABLE)}) + zero-row families (memtext-only copy when no "
                      "author failure)", "attempts": s2_revise_review.REVIEW_REVISE_ATTEMPTS_PROD,
            "memtext": "changed concepts only; skipped for revisions still failing precheck",
            "review_reuse": "unchanged rows reuse accepting verdicts", "selection_rule": s5_retry.RULE,
            "outputs": "s5_final_r2/, rows_<main>_final_r2.json, s6_r2/, release_r2/",
            "start_gate": f"open stop or start projection > {cm.PROD_BUDGET_STOP} refuses a new unit",
            "inflight_margin": INFLIGHT_MARGIN, "revise_share_default": REVISE_SHARE_DEFAULT,
            "reviewed_share": REVIEWED_SHARE, "policy": policy(cfg),
            "policy_rule": "zero_or_dev: ALL zero-row + ALL dev families with a fixable rejection, no other train",
            "repair_round": {"round": cm.FIX_ROUND, "tags": "@r3 / @g2@r3", "checks": "S1-S4 (s4_acute)",
                             "scope": "rows of s5_final_r2 failing S1-S4; items-only delta (1 attempt, no facts)",
                             "selection": "per row: repaired if accepted, else dropped; others from s5_final_r2",
                             "outputs": "s5_final_r3/, rows_<main>_final_r3.json, s6_r3/, release_r3/",
                             "review_cost_model": "s2_fix (delta + review fixed/per-row + adjudication rate)"},
            "removed_row_types": {t: cm.DROP_REASON for t in cm.DROPPED_ROW_TYPES}}


def retry_cfg(c: cm.Cfg) -> cm.Cfg:
    return cm.Cfg(**{**c.__dict__, "round": 2, "retry": True})


def units(cfg: cm.Cfg) -> dict[str, cm.Cfg]:
    """unit name (= tree name) -> its retry cfg."""
    from final_construction.wp_v9_vcm import s7_prod, s7_regen
    cfg = cm.Cfg(**{**cfg.__dict__, "s1_dir": cfg.p("s1")})        # every unit reads the shared prod_v9/s1
    n = len(cm.read(cfg.p("s1", "plan.json"))["families"])
    last = -(-(n - cm.PROD_PILOT_N) // cm.PROD_CHUNK)
    out = {f"regen_{t}": retry_cfg(s7_regen.regen_cfg(cfg, t)) for t in cm.REGEN_TARGETS}
    out.update({f"chunk_{k:02d}": retry_cfg(s7_prod.chunk_cfg(cfg, k)) for k in range(cm.RETRY_FIRST_CHUNK, last + 1)})
    return out


def _ledger(cfg: cm.Cfg) -> list[tuple[str, str, int]]:
    """(stage, family tag part, tokens) of this run id."""
    out = []
    if cfg.usage_log.exists():
        for line in cfg.usage_log.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            parts = r["tag"].split("::") if r else []
            if len(parts) >= 3 and parts[0] == cfg.run_id:
                out.append((parts[1], parts[2], int(r["total_tokens"] or 0)))
    return out


def unit_spend(cfg: cm.Cfg, uc: cm.Cfg, led: list[tuple[str, str, int]]) -> int:
    """Retry ("@r2") tokens of a unit's families in the unit's own generation."""
    fams = {f["family_id"] for f in cm.plan_raw(uc)["families"]}
    regen = bool(uc.regen)
    return sum(t for _, x, t in led if "@r2" in x and (f"@{cm.GEN_REGEN}" in x) == regen
               and x.split("@")[0].split("#")[0] in fams)


def per_family_estimate(led: list[tuple[str, str, int]]) -> dict:
    """Per revised family, from the ledger's per-call sizes of the earlier review-driven rounds (@r2, g1)."""
    calls: dict[str, list[int]] = defaultdict(list)
    for stage, x, t in led:
        if "@r2" in x and f"@{cm.GEN_REGEN}" not in x:
            calls[stage + ("#retry" if "#retry" in x else "#a2" if x.endswith("#a2") else "")].append(t)
    mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in calls.items()}
    n = {k: len(v) for k, v in calls.items()}
    rr, mt, rp, adj = (mean.get(s, 0.0) for s in ("revise_review", "memtext_family", "review_primary", "adjudicate"))
    mt_retry = n.get("memtext_family#retry", 0) / max(1, n.get("memtext_family", 0)) * mean.get("memtext_family#retry", 0)
    adj_rate = n.get("adjudicate", 0) / max(1, n.get("review_primary", 0))
    per = rr + REVIEWED_SHARE * (mt + mt_retry + rp + adj_rate * adj)
    return {"per_family": round(per), "per_call_mean": {k: round(v) for k, v in mean.items()}, "calls": n,
            "adjudicate_rate": round(adj_rate, 4), "reviewed_share": REVIEWED_SHARE,
            "parts": {"revise_review": round(rr), "memtext": round(REVIEWED_SHARE * (mt + mt_retry)),
                      "review_primary": round(REVIEWED_SHARE * rp), "adjudicate": round(REVIEWED_SHARE * adj_rate * adj)}}


def _carry(uc: cm.Cfg) -> dict | None:
    p = uc.p("s2", "author_rev2", "CARRY_FORWARD.json")
    return cm.read(p) if p.exists() else None


def _n_revise(doc: dict) -> int:
    return len(doc.get("revise", [])) + len(doc.get("memtext_only", []))


def projection(cfg: cm.Cfg, starting: tuple[str, int] | None = None) -> dict:
    """Concurrency-aware projected total; `starting` = (unit, revise count) of a unit about to start."""
    led = _ledger(cfg)
    spent = sum(t for *_, t in led)
    us = units(cfg)
    prereg = cm.sha_file(cfg.prereg) if cfg.prereg.exists() else None
    checks = [cm.read(p) for p in sorted(cfg.p("s1").glob("PROD_CHECK_retry_*.json"))]
    done = [c for c in checks if c.get("prereg_sha256") == prereg and c.get("unit_revised")]
    est = per_family_estimate(led)
    per = (sum(c["unit_retry_spend"] for c in done) / sum(c["unit_revised"] for c in done)) if done else est["per_family"]
    started = {u: d for u, uc in us.items() if (d := _carry(uc))}
    fams = {u: len(cm.plan_raw(uc)["families"]) for u, uc in us.items()}
    share = (sum(_n_revise(d) for d in started.values()) / max(1, sum(fams[u] for u in started))
             if started else REVISE_SHARE_DEFAULT)
    remaining, detail = 0.0, {}
    for u, uc in us.items():
        if uc.p("s5_final_r2", "gate.json").exists():
            continue
        if starting and u == starting[0] and u not in started:
            n, rem, state = starting[1], starting[1] * per, "starting"
        elif u in started:
            n = _n_revise(started[u])
            rem, state = max(n * per - unit_spend(cfg, uc, led), INFLIGHT_MARGIN * n * per), "in_flight"
        else:
            sel = _selection(cfg)
            n = len(sel["units"].get(u, {}).get("families", [])) if sel else share * fams[u]
            rem, state = n * per, "not_started"
        remaining += rem
        detail[u] = {"state": state, "revise_families": round(n, 1), "remaining_tokens": round(rem)}
    projected = spent + remaining
    return {"spent_total": spent, "per_family_retry_tokens": round(per),
            "per_family_basis": "measured (finished retry units, current PREREG)" if done else "ledger per-call sizes",
            "estimate": est, "revise_share": round(share, 4), "units": detail, "inflight_margin": INFLIGHT_MARGIN,
            "projected_total_tokens": round(projected), "projected_usd": round(projected / 1e6 * cm.PRICE_USD_PER_M, 2),
            "budget_stop": cm.PROD_BUDGET_STOP, "cap": cm.PROD_CAP, "prereg_sha256": prereg}


def start_gate(cfg: cm.Cfg, n_revise: int) -> dict:
    """Refuse a NEW retry unit while a stop is open or when the start projection is over the budget stop."""
    from final_construction.wp_v9_vcm import gpt_client as gc
    stops = gc.open_stops(cfg)
    if stops:
        raise RuntimeError(f"stop rule fired (s1/{stops[0].name}): no new retry unit may start "
                           "(owner override: run_v9.py --resume-after-stop REASON --owner-instruction TEXT)")
    rep = projection(cfg, (cfg.out.name, n_revise))
    flag = cfg.p("s1", START_STOP)
    if rep["projected_total_tokens"] > cm.PROD_BUDGET_STOP and not flag.exists():
        cm.write_once(flag, {**rep, "unit": cfg.out.name, "stop": True})
        raise RuntimeError(f"retry start projection {rep['projected_total_tokens']} > budget stop "
                           f"{cm.PROD_BUDGET_STOP}: s1/{START_STOP} written; no retry unit starts until the owner "
                           "overrides it (--resume-after-stop REASON --owner-instruction TEXT)")
    logger.info("retry start %s: %d families, projected %d", cfg.out.name, n_revise, rep["projected_total_tokens"])
    return rep


def selected_accepted(cfg: cm.Cfg) -> list[dict]:
    """Accepted rows of the current final selection, one tree per family (retry r2 when finalized, else s5_final)."""
    out = []
    for uc in units(cfg).values():
        for tag in ("_r8", "_r7", "_r6", "_r5", "_r4", "_r3", "_r2", ""):
            g, r = uc.p(f"s5_final{tag}", "gate.json"), uc.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final{tag}.json")
            if g.exists() and r.exists():
                acc = {o["row_id"] for o in cm.read(g)["outcomes"] if o["accepted"]}
                out += [x for x in cm.read(r)["rows"] if x["row_id"] in acc]
                break
    return out


def retry_check(cfg: cm.Cfg) -> dict:
    """`prod_check --round 2 --retry` after `finalize --round 2 --retry` of one unit (round 3: s5_fix.fix_check)."""
    from final_construction.wp_v9_vcm import s5_final
    if cfg.round in cm.FIX_ROUNDS:
        from final_construction.wp_v9_vcm import s5_fix
        return s5_fix.fix_check(cfg)
    tag = f"retry_{cfg.out.name}"
    out = cfg.p("s1", f"PROD_CHECK_{tag}.json")
    if out.exists():
        return cm.read(out)
    if not cfg.p("s5_final_r2", "gate.json").exists():
        raise RuntimeError(f"{tag} is not finalized ({cfg.p('s5_final_r2', 'gate.json')})")
    carry = _carry(cfg) or {}
    spend = unit_spend(cfg, cfg, _ledger(cfg))
    rep = projection(cfg)
    cov = s5_final.safety_coverage(selected_accepted(cfg))["unsafe_share"]
    low = {t: v for t, v in cov.items() if (v or 0.0) < cm.PROD_COVERAGE_STOP}
    rep.update(check=tag, unit=cfg.out.name, unit_revised=_n_revise(carry), unit_retry_spend=spend,
               unit_tokens_per_revised_family=round(spend / max(1, _n_revise(carry))),
               unit_yield=cm.read(cfg.p("s5_final_r2", "gate.json"))["yield"], unsafe_share_accepted=cov,
               coverage_min=cm.PROD_COVERAGE_STOP, coverage_below=low,
               stop=rep["projected_total_tokens"] > cm.PROD_BUDGET_STOP or bool(low))
    cm.write_once(out, rep)
    if rep["stop"]:
        cm.write_once(cfg.p("s1", f"PROD_STOP_{tag}.json"), rep)
    return rep
