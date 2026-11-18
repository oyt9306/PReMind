"""Production-time stop rule (`prod_check --chunk K`, CPU; run after `finalize --chunk K`).

Budget rule (2026-10-03, owner accepted ~$57): projected total = every paid token in the shared ledger so far (pilot +
production) + remaining production families x the running mean paid tokens per family over the chunks finished under
the CURRENT PREREG (each check records the PREREG sha; chunks of an earlier PREREG do not count). Projected total >
cm.PROD_BUDGET_STOP (12.65M ~ $62) -> s1/PROD_STOP_chunk<KK>.json (gpt_client.guard then refuses every paid call until
the owner overrides it with `--resume-after-stop`); > cm.PROD_BUDGET_RECOMMEND (11.6M ~ $57) -> the write-once flag
s1/RECOMMEND_NO_REVIEW_ROUND.json, which prod_chunk.sh reads to run the following chunks with --no-review-round.
Coverage rule (kept): any row type's unsafe share among the FINAL accepted rows of chunks 1..K (GT-absent rows pooled,
as in finalize; a type with no accepted row counts as 0) < cm.PROD_COVERAGE_STOP -> stop. The check report
s1/PROD_CHECK_chunk<KK>.json is write-once and required before any paid call of chunk K+1.
Regeneration (2026-10-03): `prod_check --chunk-regen T` runs the same rule after a regen run (PROD_CHECK_regen_<T>,
PROD_STOP_regen_<T>). The mean uses original ("g1") tokens of chunk families and "@g2" tokens of regen families, each
only from checks under the current PREREG; remaining = unfinished production families + planned, unfinished regen
families (planned once s1/regen_pilot_overlay.json exists). Coverage counts ONE generation per family (a finalized
regen replaces its original). PROD_BUDGET_STOP is 13.5M (owner, regeneration included).
"""
from __future__ import annotations

import json

from final_construction.wp_v9_vcm import common as cm


INFLIGHT_MARGIN = 0.20


def chunk_cfg(cfg: cm.Cfg, k: int) -> cm.Cfg:
    return cm.Cfg(**{**cfg.__dict__, "chunk": k, "regen": None, "out": cfg.p("s1").parent / f"chunk_{k:02d}"})


def gen_tokens(cfg: cm.Cfg) -> tuple[int, dict[str, dict[str, int]]]:
    """(every paid token of this run id, family -> {"g1": original tokens, "g2": regeneration tokens})."""
    spent, out = 0, {}
    if cfg.usage_log.exists():
        for line in cfg.usage_log.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            parts = r["tag"].split("::") if r else []
            if len(parts) < 3 or parts[0] != cfg.run_id:
                continue
            spent += int(r["total_tokens"] or 0)
            g = cm.GEN_REGEN if f"@{cm.GEN_REGEN}" in parts[2] else cm.GEN_ORIGINAL
            fam = out.setdefault(parts[2].split("@")[0].split("#")[0], {cm.GEN_ORIGINAL: 0, cm.GEN_REGEN: 0})
            fam[g] += int(r["total_tokens"] or 0)
    return spent, out


def _accepted(c: cm.Cfg) -> list[dict]:
    acc = {o["row_id"] for o in cm.read(c.p("s5_final", "gate.json"))["outcomes"] if o["accepted"]}
    return [r for r in cm.read(c.p("s4", f"rows_{cm.MAIN_RETRIEVER}_final.json"))["rows"] if r["row_id"] in acc]


def prod_check(cfg: cm.Cfg) -> dict:
    """After a production chunk (`--chunk K`) or a regeneration run (`--chunk-regen T`); `--retry`: s7_retry."""
    from final_construction.wp_v9_vcm import s5_final, s7_regen
    if cm.is_extra(cfg):
        from final_construction.wp_v9_vcm import s5_extra
        return s5_extra.extra_check(cfg)
    if cfg.retry:
        from final_construction.wp_v9_vcm import s7_retry
        return s7_retry.retry_check(cfg)
    k, tag = cfg.chunk, (f"chunk{cfg.chunk:02d}" if cfg.chunk else f"regen_{cfg.regen}" if cfg.regen else None)
    if not tag:
        raise RuntimeError("prod_check needs --chunk K or --chunk-regen T")
    out = cfg.p("s1", f"PROD_CHECK_{tag}.json")
    if out.exists():
        return cm.read(out)
    if not cfg.p("s5_final", "gate.json").exists():
        raise RuntimeError(f"{tag} is not finalized ({cfg.p('s5_final', 'gate.json')})")
    prereg = cm.sha_file(cfg.prereg) if cfg.prereg.exists() else None
    plan = cm.read(cfg.p("s1", "plan.json"))["families"]
    chunks = [j for j in range(1, (len(plan) - cm.PROD_PILOT_N) // cm.PROD_CHUNK + 2)
              if chunk_cfg(cfg, j).p("s5_final", "gate.json").exists()]
    if k and chunks[:k] != list(range(1, k + 1)):
        raise RuntimeError(f"chunks 1..{k} must all be finalized (finalized: {chunks})")
    regens = {t: s7_regen.regen_cfg(cfg, t) for t in cm.REGEN_TARGETS}
    regen_done = [t for t, rc in regens.items() if rc.p("s5_final", "gate.json").exists()]
    planned = cfg.p("s1", s7_regen.OVERLAY).exists() or bool(cfg.regen)

    def current(name: str) -> bool:
        p = cfg.p("s1", f"PROD_CHECK_{name}.json")
        return name == tag or (p.exists() and cm.read(p).get("prereg_sha256") == prereg)
    fam_of = {f"chunk{j:02d}": [f["family_id"] for f in cm.plan_raw(chunk_cfg(cfg, j))["families"]] for j in chunks}
    fam_of.update({f"regen_{t}": [f["family_id"] for f in cm.plan_raw(regens[t])["families"]] for t in regen_done})
    spent, led = gen_tokens(cfg)
    src = [(f, cm.GEN_REGEN if n.startswith("regen") else cm.GEN_ORIGINAL) for n, fs in fam_of.items() if current(n)
           for f in fs]
    mean = sum(led.get(f, {}).get(g, 0) for f, g in src) / max(1, len(src))
    # Fix 2026-10-03 (concurrent runs): a planned, unfinished family whose generation already has ledger spend is IN
    # FLIGHT - its spend is in `spent`, so it is projected as complete plus a margin (INFLIGHT_MARGIN x mean); only
    # families with no spend yet count a full mean (the old rule counted in-flight runs twice).
    open_fams = [(f["family_id"], cm.GEN_ORIGINAL) for j in range(1, (len(plan) - cm.PROD_PILOT_N) // cm.PROD_CHUNK + 2)
                 if j not in chunks for f in plan[cm.PROD_PILOT_N + cm.PROD_CHUNK * (j - 1):][:cm.PROD_CHUNK]]
    open_fams += [(f["family_id"], cm.GEN_REGEN) for t, (a, b) in cm.REGEN_TARGETS.items()
                  if planned and t not in regen_done for f in plan[a:b]]
    in_flight = sum(led.get(f, {}).get(g, 0) > 0 for f, g in open_fams)
    remaining = len(open_fams) - in_flight
    pending_regen = sum(g == cm.GEN_REGEN for _, g in open_fams)
    projected = spent + remaining * mean + in_flight * INFLIGHT_MARGIN * mean
    rows = [r for j in chunks if f"chunk{j:02d}" not in regen_done for r in _accepted(chunk_cfg(cfg, j))]
    rows += [r for t in regen_done for r in _accepted(regens[t])]          # one generation per family
    cov = s5_final.safety_coverage(rows)["unsafe_share"]
    low = {t: v for t, v in cov.items() if (v or 0.0) < cm.PROD_COVERAGE_STOP}
    rep = {"check": tag, "chunk": k, "regen": cfg.regen, "prereg_sha256": prereg,
           "sources_in_mean": sorted(n for n in fam_of if current(n)), "families_in_mean": len(src),
           "running_mean_tokens_per_family": round(mean), "spent_total_incl_pilot": spent,
           "remaining_families": remaining, "in_flight_families": in_flight, "inflight_margin": INFLIGHT_MARGIN,
           "pending_regen_families": pending_regen,
           "projected_total_tokens": round(projected), "projected_usd": round(projected / 1e6 * cm.PRICE_USD_PER_M, 2),
           "budget_stop": cm.PROD_BUDGET_STOP, "budget_recommend_no_review_round": cm.PROD_BUDGET_RECOMMEND,
           "unsafe_share_accepted": cov, "coverage_min": cm.PROD_COVERAGE_STOP, "coverage_below": low,
           "recommend_no_review_round": projected > cm.PROD_BUDGET_RECOMMEND,
           "stop": projected > cm.PROD_BUDGET_STOP or bool(low), "cap": cfg.cap}
    cm.write_once(out, rep)
    flag = cfg.p("s1", "RECOMMEND_NO_REVIEW_ROUND.json")
    if rep["recommend_no_review_round"] and not flag.exists():
        cm.write_once(flag, {k2: rep[k2] for k2 in ("check", "projected_total_tokens", "projected_usd",
                                                     "budget_recommend_no_review_round", "running_mean_tokens_per_family")})
    if rep["stop"]:
        cm.write_once(cfg.p("s1", f"PROD_STOP_{tag}.json"), rep)
    return rep
