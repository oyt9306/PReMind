"""Per-stage cost model for the HYBRID split (paid GPT tokens + local Qwen GPU minutes), canary and full scale.

author is calibrated on the real canary ledger (10 calls: mean input 5,900, output 2,100 tokens per family). The other
paid stages are estimated with the same image accounting (high detail: 1536x1024 query = 1,105 tokens, 448^2 entry =
255) plus low-effort reasoning (medium for adjudication). The ledger + cap enforce the real spend.
"""
from __future__ import annotations

from final_construction.wp_v9_vcm import common as cm

PRICE_USD_PER_M = 4.9            # blended $/1M tokens (COLLABORATION 2026-10-02 budget note)
QUERY_IMG, ENTRY_IMG = 1105, 255
AUTHOR_IN, AUTHOR_OUT = 5900, 2100   # measured: canary author ledger, 10 families
FULL_SCALE_FAMILIES = 848        # S2 v8 plan size (same images); upper bound since unmapped images are dropped
TWO_CONCEPT_SHARE = 49 / 143     # train+dev mapped images with two governed concepts
ROWS_PER_FAMILY = 6              # G1, G1G2, G0, general-rule + 2 GT-absent (A4-gated, upper bound)
ADJ_FAMILY_SHARE, ADJ_ITEM_SHARE = 0.45, 0.4    # canary: 4 adjudication calls / 9 reviewed families
REVISE_SHARE = 0.8               # families with >= 1 precheck failure (canary round 1: 9/10)
REVIEW_REVISE_SHARE = 0.9        # families with >= 1 gate rejection -> revise_review + round-2 re-run (canary: 9/10)
MEMTEXT_RETRY_SHARE = 0.25       # memtext_family validation retries
QWEN_SEC = {"describe": 0.6, "salvage_rewrite": 1.5, "selfcheck": 1.2, "review_second": 4.0, "author": 25.0,
            "revise": 30.0, "memtext_family": 20.0, "review_primary": 20.0, "adjudicate": 15.0, "revise_review": 30.0}
QWEN_LOAD_MIN = 2.5


def concepts(n_targets: int) -> tuple[int, int]:
    """(target-group concepts incl. replacements, all store concepts incl. replacements)."""
    tg = n_targets * (1 + cm.N_NEG[n_targets] + 1)
    return tg, cm.STORE_SIZE[n_targets] + n_targets


# measured on the canary ledger (2026-10-02, 61 calls): mean total tokens per CALL
PER_CALL = {"revise": 9750, "revise_review": 11120, "memtext_family": 2120, "review_primary": 8500, "adjudicate": 6570}
ATTEMPT2_SHARE = 0.5             # R7 loop: share of revisions that need the 2nd attempt
MEMTEXT_CALLS = 1.05             # memtext_family calls per family incl. validation retries (canary: 20 / 19)


def gpt_tokens_per_family(n_targets: int, revise_share: float = REVISE_SHARE) -> dict[str, dict[str, float]]:
    """Expected paid tokens per family and stage (in/out split kept for the report; totals from the ledger)."""
    rr, loop = REVIEW_REVISE_SHARE, 1 + ATTEMPT2_SHARE

    def tot(x: float) -> dict[str, float]:
        return {"in": 0.75 * x, "out": 0.25 * x}
    return {"author": {"in": AUTHOR_IN, "out": AUTHOR_OUT},
            "revise": tot(revise_share * loop * PER_CALL["revise"]),
            "memtext_family": tot((1 + rr) * MEMTEXT_CALLS * PER_CALL["memtext_family"]),
            "review_primary": tot((1 + rr) * PER_CALL["review_primary"]),
            "adjudicate": tot((1 + rr) * ADJ_FAMILY_SHARE * PER_CALL["adjudicate"]),
            "revise_review": tot(rr * loop * PER_CALL["revise_review"])}


def qwen_requests_per_family(n_targets: int) -> dict[str, float]:
    return {"salvage_rewrite": 1, "selfcheck": 4, "review_second": ROWS_PER_FAMILY, "author": 1,
            "revise": REVISE_SHARE, "memtext_family": 1, "review_primary": 1, "adjudicate": ADJ_FAMILY_SHARE,
            "revise_review": REVIEW_REVISE_SHARE}


def per_family(backends: dict | None = None, two_share: float = TWO_CONCEPT_SHARE,
               revise_share: float = REVISE_SHARE) -> dict[str, float]:
    backends = backends or cm.BACKENDS
    out = {}
    for stage in ("author", "revise", "memtext_family", "review_primary", "adjudicate", "revise_review"):
        if backends.get(stage) != "gpt":
            continue
        out[stage] = round(sum(share * sum(gpt_tokens_per_family(nt, revise_share)[stage].values())
                               for nt, share in ((1, 1 - two_share), (2, two_share))))
    return out


def estimate(n_families: int, backends: dict | None = None, two_share: float = TWO_CONCEPT_SHARE,
             n_identities: int = 260, stages: tuple[str, ...] | None = None,
             revise_share: float = REVISE_SHARE) -> dict:
    backends = backends or cm.BACKENDS
    pf = per_family(backends, two_share, revise_share)
    per_stage: dict[str, dict] = {}
    for stage in sorted(set(backends) - {"describe"}):
        if stages is not None and stage not in stages:
            continue
        if backends[stage] == "gpt":
            total = n_families * pf.get(stage, 0)
            per_stage[stage] = {"backend": "gpt", "tokens": round(total), "usd": round(total / 1e6 * PRICE_USD_PER_M, 2)}
        else:
            req = sum(share * qwen_requests_per_family(nt).get(stage, 0) for nt, share in ((1, 1 - two_share), (2, two_share)))
            per_stage[stage] = {"backend": "qwen", "gpu_minutes": round(n_families * req * QWEN_SEC[stage] / 60, 1)}
    if stages is None:
        per_stage["describe"] = {"backend": backends["describe"],
                                 "gpu_minutes": round(n_identities * QWEN_SEC["describe"] / 60 + QWEN_LOAD_MIN, 1)}
    gpt_total = sum(v.get("tokens", 0) for v in per_stage.values())
    return {"families": n_families, "per_stage": per_stage, "gpt_tokens_total": gpt_total,
            "gpt_tokens_x1_3": round(1.3 * gpt_total), "usd_total": round(gpt_total / 1e6 * PRICE_USD_PER_M, 2),
            "usd_x1_3": round(1.3 * gpt_total / 1e6 * PRICE_USD_PER_M, 2),
            "qwen_gpu_minutes_total": round(sum(v.get("gpu_minutes", 0) for v in per_stage.values()), 1)}


def canary_rerun(n_revise: int, n_review: int, backends: dict | None = None) -> dict:
    """Paid tokens still to spend on a review-driven round: revise_review (verified loop) + memtext for the revised
    families, review_primary + adjudication for every family reviewed in that round (revised + carried forward)."""
    loop = 1 + ATTEMPT2_SHARE
    stages = {"revise_review": n_revise * loop * PER_CALL["revise_review"],
              "memtext_family": n_revise * MEMTEXT_CALLS * PER_CALL["memtext_family"],
              "review_primary": n_review * PER_CALL["review_primary"],
              "adjudicate": n_review * ADJ_FAMILY_SHARE * PER_CALL["adjudicate"]}
    total = round(sum(stages.values()))
    return {"per_stage": {k: round(v) for k, v in stages.items()}, "tokens": total, "tokens_x1_3": round(1.3 * total),
            "usd_x1_3": round(1.3 * total / 1e6 * PRICE_USD_PER_M, 2), "revised": n_revise, "reviewed": n_review}


# P2/P4 production model: verified first-pass author + delta retries + delta review-revise (one verified round)
PROD = {"author": 8000, "author_delta_share": 0.6, "author_delta_call": 3500, "review_revise_share": 0.5,
        "revise_review_delta_call": 4000, "attempts": 1.0 + ATTEMPT2_SHARE}
PROD_TRAIN_DEV_FAMILIES = 346    # production plan (train + dev) on the real pool


def production_per_family() -> dict[str, int]:
    rr = PROD["review_revise_share"]
    memtext = MEMTEXT_CALLS * PER_CALL["memtext_family"]
    review, adj = PER_CALL["review_primary"], ADJ_FAMILY_SHARE * PER_CALL["adjudicate"]
    return {k: round(v) for k, v in {
        "author": PROD["author"], "author_delta": PROD["author_delta_share"] * PROD["author_delta_call"],
        "memtext_family": memtext, "review_primary": review, "adjudicate": adj,
        "revise_review": rr * PROD["attempts"] * PROD["revise_review_delta_call"],
        "round2_memtext_review_adjudicate": rr * (memtext + review + adj)}.items()}


def production_rest_per_family() -> float:
    pf = production_per_family()
    return float(sum(v for k, v in pf.items() if k not in ("author", "author_delta")))


def production_estimate(n_families: int) -> dict:
    pf = production_per_family()
    per = sum(pf.values())
    return {"families": n_families, "per_family": pf, "tokens_per_family": per, "tokens": per * n_families,
            "usd": round(per * n_families / 1e6 * PRICE_USD_PER_M, 2),
            "usd_x1_3": round(1.3 * per * n_families / 1e6 * PRICE_USD_PER_M, 2),
            "within_target": per <= cm.PROD_TOKENS_PER_FAMILY_MAX}


# Pilot ledger (2026-10-03, 30 families): mean total tokens per CALL by stage, and the cost-cut deltas measured offline
# on the 19 first-pass failures (s2_delta.touched_min: 12.5 -> 3.3 returned facts, 1.8 -> 1.2 items, 21% fewer images).
PILOT_CALL = {"author": 9387, "author_delta": 7217, "revise": 5546, "memtext_family": 2502, "review_primary": 9749,
              "revise_review": 6029, "adjudicate": PER_CALL["adjudicate"]}
CUT = {"author_prompt": 790, "delta_payload": 1400, "delta_effort": 200}
PILOT_SHARES = {"delta": 19 / 30, "delta_after_prompt_fix": 16 / 30, "residual_revise": 0.23, "memtext_calls": 1.2,
                "attempts_review_revise": 1.3}


def production_projection(review_revise_share: float, cuts: bool = True) -> dict:
    """Per-family paid tokens for the production run from pilot-measured call sizes (review-driven rework share rr is
    the open parameter: measured by the pilot's round-2 gate)."""
    c, sh = PILOT_CALL, PILOT_SHARES
    delta = c["author_delta"] - (CUT["delta_payload"] + CUT["delta_effort"] if cuts else 0)
    adj = ADJ_FAMILY_SHARE * c["adjudicate"]
    stages = {"author": c["author"] - (CUT["author_prompt"] if cuts else 0),
              "author_delta": (sh["delta_after_prompt_fix"] if cuts else sh["delta"]) * delta,
              "revise": sh["residual_revise"] * (c["revise"] - (CUT["delta_payload"] if cuts else 0)),
              "memtext_family": sh["memtext_calls"] * c["memtext_family"], "review_primary": c["review_primary"],
              "adjudicate": adj,
              "review_driven_round": review_revise_share * (
                  sh["attempts_review_revise"] * (c["revise_review"] - (CUT["delta_effort"] if cuts else 0))
                  + c["memtext_family"] + c["review_primary"] + adj)}
    return {"per_stage": {k: round(v) for k, v in stages.items()}, "tokens_per_family": round(sum(stages.values()))}


def report(backends: dict | None = None) -> dict:
    return {"production_projection_pilot_measured": {f"rr={rr}": {"cuts": production_projection(rr)["tokens_per_family"],
                                                                 "no_cuts": production_projection(rr, False)[
                                                                     "tokens_per_family"]} for rr in (0.2, 0.3, 0.5)},
            "production": {"train_dev": production_estimate(PROD_TRAIN_DEV_FAMILIES),
                           "full_scale": production_estimate(FULL_SCALE_FAMILIES),
                           "pilot": production_estimate(cm.PROD_PILOT_N), "assumptions": PROD},
            "per_family_tokens": per_family(backends), "canary": estimate(cm.CANARY_N, backends),
            "full_scale": estimate(FULL_SCALE_FAMILIES, backends),
            "assumptions": {"price_usd_per_m": PRICE_USD_PER_M, "rows_per_family": ROWS_PER_FAMILY,
                            "two_concept_share": round(TWO_CONCEPT_SHARE, 3), "full_scale_families": FULL_SCALE_FAMILIES,
                            "k": cm.K, "adjudicated_family_share": ADJ_FAMILY_SHARE, "revise_share": REVISE_SHARE,
                            "memtext_retry_share": MEMTEXT_RETRY_SHARE, "author_measured": [AUTHOR_IN, AUTHOR_OUT],
                            "review_revise_share": REVIEW_REVISE_SHARE, "per_call_measured": PER_CALL,
                            "attempt2_share": ATTEMPT2_SHARE}}
