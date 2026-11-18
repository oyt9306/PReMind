"""Phase II-quadruple-prime copy (2026-10-24, PREREG_PHASE3_MIE.md Amendment 9): train_p2d_mie_4b/stillpen.py +
mode "p2e" = p2d with (i) the p2e release (RELEASE_TRAIN_DEV_v20_hca1_p2e = p2b + 87 move-required conflict_train_v2
rows on the hca1 heat/cold images; key actions / MIE inventory = p2e_release side files), (ii) STREAMS_P2E: conflict
stream at the NATURAL conflict share (no upsampling; tuned by the CPU sim) and the G0 replay stream at 1/3 of p2d
(0.06 -> 0.02, owner 2026-10-24). The penalty (P1-P4 0.2, P5 0.1) and the hazard_first gate are the ORIGINAL
Phase II ones (owner 2026-10-24: the unsupported stay-still term stays -0.2; a stay-still answer on a move-required
row costs P1 + P4 = 0.4). Every p2d / p2c / p2b / ws125 sampler path is unchanged.

Phase II-triple-prime copy (2026-10-23): train_p2c_mie_4b/stillpen.py + mode "p2d" = p2c with a MASTERY-AWARE
conflict stream mirroring the Phase I replay rule (trainer_v2_v4/curriculum_v2.py: MASTERY 0.75, REPLAY min/max delay
10/30, acute visit cap 3, priority = 1 - mastery). Every p2c / p2b / ws125 code path is unchanged (no "mastery" key).
  stream positions (non-hca1 positions; u2 < STREAMS_P2D["conflict"] -> conflict, < conflict + g0 -> G0 replay): the
    curriculum row x of that position is drawn (and consumed, as the balance mode's type substitution) and REPLACED by
    the stream row only if x is a PRESENT memory-required row (condition != gt_absent and target memory_required) for
    the conflict stream, or a non-withhold row for the G0 replay stream; otherwise x is kept (displace_skip_*, counted).
    So stream draws never take absent / withhold positions (owner retention review 2026-10-23) and conflict draws never
    take G0 positions; the curriculum's own withhold positions stay exactly where the restored curriculum puts them.
  _mastery_pick: with probability p_replay (seeded u3 = sha256(MASTERY_SEED|step|pos)) the replay queue first, else the
    fresh pool first; the other side if the first is empty; both empty -> x is kept (stream_fallbacks["conflict"]). Replay: due <= step <= deadline, visits < 3, 10-step gap, best =
    (-priority, due, row_id). Fresh: conflict rows never drawn by the stream in this run, image round-robin, 10-step gap.
  after the step (main_v4hcur21 calls requeue(filtered) then observe(outcomes) on EVERY rank with all-rank data):
    mastery = #applied / #judged (unjudged excluded) of the row's group (outcome ``mie_labels``, added by mastery_p2d's
    group_outcome wrapper); no_signal = DAPO-dropped OR all judged labels identical OR nothing judged.
    mastery >= 0.75 -> retired; else visits >= 3 -> parked; else queued {due +10, deadline +30, priority 1 - mastery,
    no_signal}; entries past their deadline expire (as curriculum_v2).
  temperature: a no_signal replay row drawn this step is put in HOT (rolled out at 1.1 by mastery_p2d) unless the run's
    HOT prompts would exceed 5 % of all prompts drawn so far (then 1.0, hot_skipped_cap counted).
  refunds of unused DAPO refill positions undo the draw (seen / visits / queue entry / HOT / counters).

Phase II'' copy (2026-10-23, REWARD_BREAKEVEN_v1.md section 5) of train_p2b_mie_4b/stillpen.py. Added mode "p2c"
(mode p2b / ws125 code paths unchanged): on every draw position that is NOT an hca1 boost position (pos % 4 != 0), a
seeded uniform u = _u2(step, pos) (sha256 of STREAM_SEED|step|pos; independent of the hca1 draw) selects
  u < STREAMS["conflict"]                          -> conflict stream (MIE-inventory rows; blocked in the curriculum)
  u < STREAMS["conflict"] + STREAMS["g0"]           -> non-acute G0 replay stream (labels ``g0_replay``: row_type G0*,
                                                       no hazard key action, query_style not panic/calm, i.e. the
                                                       call decision is active; NOT blocked in the curriculum)
  otherwise                                        -> the restored curriculum (unchanged)
Both streams: image round-robin pools, seeded, 10-step gap, outside the curriculum bookkeeping (boost kind, refunds go
back to the stream's front); an exhausted stream position falls back to the curriculum (counted per stream).

Phase II' copy (2026-10-23, PREREG_PHASE3_MIE.md Amendment 2) of train_stillpen_ws_4b_v2/stillpen.py. Changes,
all confined to this file (penalty, hf gate, restore, boost and every v2 constant are unchanged):
  * KEY_ACTIONS / RELEASE_TRAIN -> the p2b release (hca1 train minus the 4 conflict-dev families + 63 conflict rows);
    labels gain ``conflict`` (row in the MIE inventory).
  * mode "p2b" = ws125 + a conflict stream: every draw position with pos % 4 == 2 (pos = index in the step's draw
    stream: batch and DAPO refills) is a boost draw from the ``conflict`` pool (MIE-inventory rows, image round-robin,
    seeded, 10-step gap, outside the curriculum bookkeeping like the hca1 boost). Positions pos % 4 == 0 stay the hca1
    boost, all other positions are the restored curriculum unchanged. Conflict rows are drawn ONLY by that stream:
    the curriculum never emits a conflict row (they are made ineligible) nor a row of the excluded families (in the
    plan, not in the release). Conflict share = 1/4 of the draw positions ~= 2 x the natural share 129/1035 = 12.5 %.
    A blocked conflict position (every pool row within the 10-step gap) falls back to the curriculum (counted).
  * plan: Plan22 is built from the ORIGINAL hca1 release rows (minus hca1) exactly as v2 does, so the schedule sha and
    the restored step-124 state are the original ones; of_rows()/check() map it onto the p2b release (excluded family
    rows out, hca1 + new conflict rows in, outside the plan).
  * mode "ws125" with the original data is the v2 sampler (CPU test: identical draws).

v2 (owner 2026-10-20: copy of train_stillpen_ws_4b/stillpen.py + (a') stillness counts as a specific hazard_first cue
on stillness-supported rows, (c) symmetric P5 = 0.1 for a missing STILL instruction on stillness-supported acute rows).
Stillness / emergency habitual-phrase penalty + acute up-weighting sampler for the CUR22S s125 warm-start run.

Frozen trainer modules are imported read-only and wrapped in-process (never edited):
  * penalty: eval-free, rule-based, applied to HR.answer_text(stage-2 text) of every TRAINING rollout. It is added to
    the per-rollout ledger ``total`` right after the full scoring chain (patch_h G.score -> reward_v4hg.score_episode
    with every wrapper: v4wh_g, AVR, warning credit, bonus v1.1, fa25, ws52u1/credit_gate, AE QA), by wrapping the
    ``score`` callable that trainer_v2.grpo_v2.train_step hands to trainer_v2.algo_v2.dynamic_rollout. The ledger
    gains a top-level ``stillpen`` record (no new component: reward_v5's component set stays exact). Dev eval does
    not go through dynamic_rollout, so it is unpenalised.
  * per-step firing counts: trainer_v2.run_v2.run_steps is wrapped; counts are all-gathered and stored as
    rec["stillpen"] before eval_v1.ddp_main writes steps.jsonl.
  * sampler (MODES): curriculum_v2_1.CurriculumSamplerV21 is replaced by a subclass that fills every
    ``every``-th position of each step's draw stream (batch and DAPO refills) from a boost pool, outside the curriculum
    bookkeeping: with probability ``p_new`` (seeded hash of step + position) the hca1 rows, else the 322 original
    acute key-action rows (image round-robin each, 10-step gap). ws125: starts from the ORIGINAL run's logged
    curriculum state after step 124 (cursor 648 = checkpoint step_000125 data_cursor) with the curriculum clock at
    loop step + 125; scratch: the original curriculum from step 0, boost = hca1 rows only.
  * plan: curriculum_v2_2.Plan22 (after balance_ae) is subclassed so the hca1 rows never enter the curriculum plan
    (same strata, quotas, schedule and sha as v20); they are known to the sampler only through the boost path.
"""
from __future__ import annotations

import ast
import json
import os
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
FC = (Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction')
KEY_ACTIONS_V20 = FC / "v4_build/g0_redesign_v1/rebuild_v20/TRAIN_KEY_ACTIONS_v20.jsonl"
RELEASE_TRAIN_V20 = FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20/gt_retrieved_drop/train.jsonl"
ORIG_KEY_ACTIONS = FC / "v4_build/freeze_shortcut_fix/train_stillpen_ws_4b/hca1/TRAIN_KEY_ACTIONS_v20_hca1b.jsonl"   # v2 run
ORIG_RELEASE_TRAIN = FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1/gt_retrieved_drop/train.jsonl"                # v2 run
KEY_ACTIONS = (Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction') / 'cprime_inputs' / 'TRAIN_KEY_ACTIONS_P2E_v1_gtclean.jsonl'    # hca1b + 63 v1 + 87 v2 conflict rows
RELEASE_TRAIN = FC / "wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1_p2e/gt_retrieved_drop/train.jsonl"
MIE_INVENTORY = FC / "v4_build/conflict_train_aug_v1/p2e_release/MIE_INVENTORY_TRAIN_P2E_v1.jsonl"
CONFLICT = {"every": 4, "offset": 2}             # mode p2b conflict stream positions
STREAMS = {"conflict": 0.23, "g0": 0.06}         # mode p2c: P(stream | non-hca1 position); tuned by the 100-step CPU sim
# (sim: conflict ~1.5x natural 12.5 %; G0 replay total ~12 % = stream ~5 % + curriculum ~7-8 %; the v2 run drew 6.6 %)
STREAM_SEED = "p2c_streams_v1|0"
MASTERY_CFG = {"mastery": 0.75, "min_delay": 10, "max_delay": 30, "max_visits": 2, "p_replay": 0.5,   # p2e: visits 3 -> 2 (owner 2026-10-24)
               "no_signal_temperature": 1.1, "hot_cap_frac": 0.05, "displace_pmr_only": True}
STREAMS_P2D = {"conflict": 0.62, "g0": 0.06}     # mode p2d: conflict stream only converts PMR curriculum positions
STREAMS_P2E = {"conflict": 0.41, "g0": 0.02}     # mode p2e: owner cap ~15 % of prompts (~30 conflict prompts / 50 steps; CPU sim, per size), G0 1/3 of p2d
MASTERY_SEED = "p2d_mastery_v1|0"
# ----------------------------------------------------------------------------- factpc_v1 targeted fact replay (inactive unless outcomes carry fact_labels)
FACT_REPLAY = {"per_step_max": 1,       # at most 1 of the 4 global prompt groups per step; inside existing replay slots only (no new slot)
               "max_replays_per_row": 1, "min_delay": 10, "max_delay": 30}      # delays as the conflict replay (MASTERY_CFG)

HOT: set = set()                                 # no_signal replay rows drawn this step (rolled out at 1.1)
ACTIVE: dict = {"sampler": None}                 # the main (non-fork) p2d sampler of this process
NEUTRAL_STYLES = ("panic", "calm")               # = reward_v5.NEUTRAL_DECISION_STYLES (decision neutral on G0 rows)
HCA1_PREFIX = "heat_ctx_aug_hca1/"
ORIG_RUN = Path(os.environ.get('PREMIND_PHASE1_RUN', 'runs/cprime_p1_4b_s0'))
ORIG_CUR_LOG = Path(str(ORIG_RUN) + ".curriculum") / "steps.jsonl"
ORIG_STATE_STEP, ORIG_CURSOR = 124, 628          # state logged after step 124 = start of step 125
STEP_OFFSET = 125
# Boost config per mode (owner 2026-10-20): every ``every``-th draw position is a boost draw from ``pools`` (weighted,
# seeded hash of step + position; then the other weighted pools), else ``fallback`` pools (counted), else curriculum.
# Pools: hca1 = the 24 heat/cold rows; acute = the 322 v20 acute rows; still / other = v20 acute rows that are /
# are not stillness-supported (for a possible 1/3-1/3-1/3 acute-type balance: pools=[(still,1/3),(hca1,1/3),(other,1/3)]).
# ``balance`` (scratch, owner decision 2026-10-20): every acute row the curriculum emits keeps its position, but its
# TYPE is re-drawn: still (A, stillness-supported v20) / hca1 (B) / other (C, v20 not stillness-supported), 1/3 each
# (seeded hash of step + position); the drawn type's next row (image round-robin, 10-step gap) replaces the emitted
# row unless the type equals the emitted row's own type (then the emitted row is kept); blocked -> fallback order
# B -> C -> A (counted). Non-acute draws are untouched. ``every`` 0 = no boost positions.
MODES = {"ws125": {"restore": True, "offset": STEP_OFFSET, "every": 4, "pools": [["hca1", 1.0]], "fallback": ["acute"],
                   "balance": None},
         "scratch": {"restore": False, "offset": 0, "every": 0, "pools": [], "fallback": [],
                     "balance": {"types": [["still", 1 / 3], ["hca1", 1 / 3], ["other", 1 / 3]],
                                 "fallback": ["hca1", "other", "still"]}}}
MODES["p2b"] = {**MODES["ws125"], "conflict": CONFLICT}
MODES["p2c"] = {**MODES["ws125"], "streams": STREAMS}
MODES["p2d"] = {**MODES["ws125"], "streams": STREAMS_P2D, "mastery": MASTERY_CFG}
# p2e curated conflict stream (owner 2026-10-24): every conflict row has a curated tier (p2e_release/TIERS_P2E_v1.json:
# core = clear must-learn adaptation, late = valid but less salient). Band 1 = the first ``band1_steps`` loop steps: only
# core rows are drawn; band 2 adds late rows. Fresh rows: core first, then late; within a tier move-required and other
# rows are interleaved at their pool proportions (stratified position), seeded order inside each stratum (_u_row). Replays only for core rows (max 2 visits = 1 replay); late rows are drawn once (max 1 visit, never replayed).
# Per-size profiling tiers (owner 2026-10-24) REPLACE the curation-only order: DIFFICULTY_P2E_4b.json (profile_p2e.py:
# 8 rollouts per curated row with this size's Phase I init; 8/8 -> drop (never drawn), 1-7/8 -> core (a curated-late row
# stays late), 0/8 -> late; difficulty = 1 - success rate). Within a band fresh rows go easy -> hard (difficulty bucket of
# 1/8), move-required rows interleaved at their share inside a bucket. "learned" (8/8 kept at advantage weight 0.5 when
# < 40 rows would remain; not used by either size) is band 2, one visit.
CUR_TIERS = FC / "v4_build/conflict_train_aug_v1/p2e_release/TIERS_P2E_v1.json"      # owner curation (core / late)
TIERS = FC / "v4_build/conflict_train_aug_v1/p2e_release/DIFFICULTY_P2E_4b.json"
BANDS = {"band1_steps": 15, "band1_tiers": ["core"], "max_visits_by_tier": {"core": 2, "late": 1, "learned": 1, "drop": 0}}
TIER_RANK = {"core": 0, "late": 1, "learned": 2}


def tier_rows(path: Path = TIERS) -> dict:
    return json.loads(Path(path).read_text())["rows"]


def tier_map(path: Path = TIERS) -> dict:
    return {r: v["tier"] for r, v in tier_rows(path).items()}
MODES["p2e"] = {**MODES["ws125"], "streams": STREAMS_P2E, "mastery": MASTERY_CFG, "bands": BANDS}
BOOST_EVERY = MODES["ws125"]["every"]
BOOST = "boost"
BOOST_SEED = "stillpen_boost_v1|0"
LAMBDA = 0.2
LAMBDA_P5 = 0.1                                  # (c) symmetric: no STILL instruction on a stillness-supported acute row

STILL_V1 = re.compile(r"\b(?:do not|don'?t|never)\s+(?:try to\s+)?(?:move|reposition|shift)\s+(?:him|her|them|the "
                   r"(?:person|man|woman|child|boy|girl|patient|victim)|yourself)\b"
                   r"|\b(?:do not|don'?t)\s+(?:try to\s+)?(?:move|get up|stand up|sit up)\b(?!\s+(?:the|this|that|it|any|anything)\b)"
                   r"|\b(?:stay|keep (?:still|him still|her still|them still)|remain)\s+(?:still|put|where (?:you|they|he|she) (?:are|is)"
                   r"|in place|lying down)\b"
                   r"|\bkeep (?:him|her|them|the person) (?:still|lying (?:down|still)|where (?:he|she|they) (?:is|are))\b"
                   r"|\bavoid moving (?:him|her|them|the person)\b", re.I)   # owner-narrowed 2026-10-20: the PERSON only
# v2 (owner hacking review 2026-10-20): valid paraphrased stillness instructions also count (P1, P3, P4, P5 and the
# (a') cue). Not included on purpose: "avoid lifting" / "without pulling" / "stay seated" (seated rest is not stillness).
# Deviation from the owner's pattern: the verb "walk" is left out of the let/have/make clause, because the
# owner-approved hca1 severe-row first aid ("Do not make them walk: carry or assist them ...") would otherwise be a
# STILL match and the gold itself would cost P1 + P4 = 0.4.
STILL_PARA = (r"\b(?:don'?t|do not|never)\s+(?:let|have|make|help|allow)\s+(?:him|her|them|the person)\s+"
              r"(?:get up|stand|sit up|move)\b"          # 'walk' dropped: hca1 severe golds say "do not make them walk; carry them"
              r"|\b(?:leave|keep|have)\s+(?:him|her|them|the person)\s+(?:where (?:he|she|they) (?:is|are)|on the "
              r"(?:floor|ground)|lying(?: down)?|in (?:that|this|the same) position|as (?:he|she|they) (?:is|are))\b"
              r"|\b(?:stay|remain)\s+(?:lying|lying down|on the (?:floor|ground))\b"
              r"|\bwait (?:for help )?before (?:moving|getting up)\b|\bkeep (?:him|her|them) from (?:moving|getting up)\b")
STILL = re.compile(STILL_V1.pattern + "|" + STILL_PARA, re.I)          # = STILL_V2
EMERG = re.compile(r"911|\bemergency (?:services|help|number|line)\b|\bambulance\b|\bparamedics?\b|\bpoison control\b",
                   re.I)


# ----------------------------------------------------------------------------- labels
def _labels(v: Any) -> set:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = ast.literal_eval(v)
    return {x.get("label") for x in (v or []) if isinstance(x, dict)}


def conflict_ids(path: Path = MIE_INVENTORY) -> set:
    return {json.loads(x)["modes"]["gt_retrieved_drop"]["sample_id"] for x in Path(path).read_text().splitlines() if x.strip()}


def load_labels(key_actions: Path = KEY_ACTIONS, release_train: Path = RELEASE_TRAIN,
                conflict: Optional[set] = None) -> dict[str, dict]:
    """{training item id (release sample_id): {acute, still_ok, emerg_contra, safe, move_req, hca1}} per train row.
    A key-action row maps by sample_id; the evaluator row_id must agree (checked)."""
    ka = [json.loads(x) for x in Path(key_actions).read_text(encoding="utf-8").splitlines() if x.strip()]
    ka = {r["sample_id"]: r for r in ka if r.get("split") == "train"}
    out = {}
    for line in Path(release_train).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ev = row["evaluator"] if isinstance(row["evaluator"], dict) else json.loads(row["evaluator"])
        k = ka.get(row["sample_id"])
        if k is not None and k["row_id"] != ev.get("row_id"):
            raise ValueError(f"key-action row {k['row_id']} != evaluator row {ev.get('row_id')} ({row['sample_id']})")
        spec, contra = (_labels(k["specific"]), _labels(k["contraindicated"])) if k else (set(), set())
        out[row["sample_id"]] = {"acute": k is not None, "image": k["image"] if k else None,
                                 "still_ok": "dont_move_keep_still" in spec or "move_injured" in contra,
                                 "emerg_contra": "emergency_call_unhurt" in contra,
                                 "safe": k is None and row.get("risk") == "safe",   # release field (items: risk_label)
                                 "move_req": bool(k and k.get("move_required")),
                                 "hca1": row["sample_id"].startswith(HCA1_PREFIX),
                                 "conflict": row["sample_id"] in (conflict or ()),
                                 "g0_replay": (str(row.get("row_type", "")).startswith("G0") and k is None
                                               and str(row.get("query_style")) not in NEUTRAL_STYLES)}
        if out[row["sample_id"]]["move_req"] and k.get("stillness_supported", False):
            raise ValueError(f"{row['sample_id']}: move_required and stillness_supported")
    return out


def terms(text: str, lab: Optional[dict]) -> dict:
    """{p1..p5} booleans (each at most once) for one final answer. P4 (owner 2026-10-20) = a STILL match on a
    move-required row (hca1), ADDITIVE to P1 (those rows are acute and not stillness-supported): -0.4 in total.
    P5 (v2) = NO STILL match on a stillness-supported acute row (weight LAMBDA_P5)."""
    if not lab:
        return {"p1": False, "p2": False, "p3": False, "p4": False, "p5": False}
    still, emerg = bool(STILL.search(text or "")), bool(EMERG.search(text or ""))
    return {"p1": lab["acute"] and not lab["still_ok"] and still,
            "p2": lab["acute"] and lab["emerg_contra"] and emerg,
            "p3": (not lab["acute"]) and lab["safe"] and (still or emerg),
            "p4": lab["acute"] and lab.get("move_req", False) and still,
            "p5": lab["acute"] and lab["still_ok"] and not still}


def penalty(text: str, lab: Optional[dict], lam: float = LAMBDA) -> tuple[float, dict]:
    t = terms(text, lab)
    return lam * sum(v for k, v in t.items() if k != "p5") + LAMBDA_P5 * t["p5"], t


# ----------------------------------------------------------------------------- hazard_first gate (owner 2026-10-20)
HF_EXCLUDED = ("call_emergency", "dont_move_keep_still", "keep_in_place")


def still_supported(ra: Any) -> bool:
    """Stillness-supported row: specific dont_move_keep_still OR contraindicated move_injured (= load_labels still_ok)."""
    return (any(a.label == "dont_move_keep_still" for a in ra.actions if a.specific)
            or any(lab == "move_injured" for lab, _ in (ra.contra or ())))


def hf_specific_lead(ra: Any, answer: str) -> bool:
    """True iff the first two answer sentences (hazard_match.first_sentence(.., 2), the span reward_v5's hazard_first
    reads) hit a cue of the row's hazard-SPECIFIC key actions, excluding call_emergency and stillness labels; (a') on a
    stillness-supported row the dont_move_keep_still cues and the STILL regex also count. call_emergency never counts.
    A row without such cues never qualifies."""
    from trainer_v2 import hazard_match as HM
    lead = HM.first_sentence(answer, 2)
    cues = [c for a in ra.actions if a.specific and a.label not in HF_EXCLUDED for c in a.cues]
    if still_supported(ra):
        cues += [c for a in ra.actions if a.specific and a.label == "dont_move_keep_still" for c in a.cues]
        cues.append(STILL.pattern)
    return bool(lead) and any(re.search(c, lead, re.I) for c in cues)


def install_hf_gate() -> None:
    """reward_v4h.hazard_info -> info["hf_specific_lead"]; reward_v4h.adjust -> after the existing gate (state specific
    and nothing contraindicated), the remaining hazard_first bonus on an acute row is zeroed unless hf_specific_lead;
    the total is re-summed exactly as adjust / the v4wh_g guard sum it. Both the plain v4wh path (score_flags) and the
    audited v4wh_g guard (reward_v4h_guard.score_target) call these two module attributes."""
    from trainer_v2_v4 import hazard_rubric as HR
    from trainer_v2_v4 import reward_v4h as RH
    if getattr(RH.adjust, "_stillpen_hf", False):
        raise RuntimeError("hazard_first gate installed twice")
    info0, adj0 = RH.hazard_info, RH.adjust

    def hazard_info(hz, stage2, hazard_pattern):
        info = info0(hz, stage2, hazard_pattern)
        if info is not None:
            info = {**info, "hf_specific_lead": hf_specific_lead(hz["ra"], HR.answer_text(stage2))}
        return info

    def adjust(ledger, f, info):
        out = adj0(ledger, f, info)
        comps = out["components"]
        hf = comps.get("hazard_first") or {}
        if (info is not None and not comps["degenerate"]["applicable"] and hf.get("applicable") and hf.get("value")
                and hf["value"] > 0 and not info.get("hf_specific_lead", False)):
            comps["hazard_first"] = {**hf, "value": 0.0, "status": f"{hf['status']}:gated_no_specific_lead(stillpen)"}
            out["total"] = round(sum(c["value"] for c in comps.values() if c["applicable"] and c["value"] is not None), 10)
        return out
    adjust._stillpen_hf = True
    RH.hazard_info, RH.adjust = hazard_info, adjust


# ----------------------------------------------------------------------------- reward hook + step counts
_S: dict = {"labels": None, "counts": Counter(), "orig": {}}


def score_with_penalty(score: Any, ep: Any, cfg: Any) -> dict:
    from trainer_v2_v4 import hazard_rubric as HR
    led = score(ep, cfg)
    lab = _S["labels"].get(ep.item.row_id)
    if lab is None:
        raise KeyError(f"stillpen: no label for training row {ep.item.row_id}")
    pen, t = penalty(HR.answer_text(ep.stage2_text), lab)
    c = _S["counts"]
    c["n_scored"] += 1
    c["n_acute"] += lab["acute"]
    c["n_safe"] += lab["safe"]
    c.update({k: 1 for k, v in t.items() if v})
    c["n_penalised"] += pen > 0
    c["pen_sum"] += pen
    return {**led, "total": led["total"] - pen, "stillpen": {"lambda": LAMBDA, "lambda_p5": LAMBDA_P5, "penalty": -pen, **t}}


def install_penalty() -> None:
    from trainer_v2 import algo_v2 as AL
    from trainer_v2 import run_v2 as RV
    if _S["orig"]:
        raise RuntimeError("stillpen hooks installed twice")
    _S["labels"] = load_labels()
    dyn, run = AL.dynamic_rollout, RV.run_steps
    _S["orig"] = {"dyn": dyn, "run": run}

    def dynamic_rollout(items, images, backend, cfg, refill, rollout, score, *a, **kw):
        return dyn(items, images, backend, cfg, refill, rollout, lambda ep, c: score_with_penalty(score, ep, c),
                   *a, **kw)

    def run_steps(*a, **kw):
        _S["counts"] = Counter()
        recs = run(*a, **kw)
        dist = kw.get("dist")
        parts = dist.gather(dict(_S["counts"])) if dist is not None else [dict(_S["counts"])]
        tot = Counter()
        for p in parts:
            tot.update(p)
        if not tot.get("n_scored"):
            raise RuntimeError("stillpen: penalty hook scored no episode this step (dynamic sampling off?)")
        recs[0]["stillpen"] = {"lambda": LAMBDA, "lambda_p5": LAMBDA_P5, **{k: round(v, 6) for k, v in sorted(tot.items())}}
        return recs
    AL.dynamic_rollout, RV.run_steps = dynamic_rollout, run_steps


# ----------------------------------------------------------------------------- sampler
POOLS = {"hca1": lambda l: l["acute"] and l.get("hca1", False),
         "acute": lambda l: l["acute"] and not l.get("hca1", False) and not l.get("conflict", False),
         "conflict": lambda l: l.get("conflict", False),
         "g0": lambda l: l.get("g0_replay", False) and not l.get("conflict", False),
         "still": lambda l: l["acute"] and not l.get("hca1", False) and l["still_ok"],
         "other": lambda l: l["acute"] and not l.get("hca1", False) and not l["still_ok"]}


def boost_order(labels: dict, item_ids: set, pool: str = "acute") -> list[str]:
    """Rows of one boost pool present in the items, image round-robin (images and rows within an image seeded)."""
    by: dict = {}
    for r, lab in sorted(labels.items()):
        if POOLS[pool](lab) and r in item_ids:
            by.setdefault(lab["image"], []).append(r)
    if not by:
        return []
    rng = random.Random(BOOST_SEED + ("" if pool == "acute" else "|" + pool))
    imgs = sorted(by)
    rng.shuffle(imgs)
    for i in imgs:
        rng.shuffle(by[i])
    return [by[i][k] for k in range(max(map(len, by.values()))) for i in imgs if k < len(by[i])]


def _u2(step: int, pos: int) -> float:
    """p2c stream draw: own seed, independent of _u (the hca1 boost / balance draws)."""
    import hashlib
    return int(hashlib.sha256(f"{STREAM_SEED}|{step}|{pos}".encode()).hexdigest()[:8], 16) / 16 ** 8


def _u3(step: int, pos: int) -> float:
    """p2d replay-vs-fresh draw: own seed."""
    import hashlib
    return int(hashlib.sha256(f"{MASTERY_SEED}|{step}|{pos}".encode()).hexdigest()[:8], 16) / 16 ** 8


def _u_row(row_id: str) -> float:
    """p2e fresh order within a tier: a seeded uniform key per row (rows of all images mixed at their natural shares;
    the image round-robin would starve the 6 hca1 images, which carry 7-8 rows each)."""
    import hashlib
    return int(hashlib.sha256(f"{MASTERY_SEED}|fresh|{row_id}".encode()).hexdigest()[:8], 16) / 16 ** 8


def pmr(it: Any) -> bool:
    """Present memory-required row (the only curriculum positions a p2d conflict draw may displace)."""
    return it.row.get("condition") != "gt_absent" and bool(it.target.get("memory_required"))


def _u(step: int, pos: int) -> float:
    import hashlib
    return int(hashlib.sha256(f"{BOOST_SEED}|{step}|{pos}".encode()).hexdigest()[:8], 16) / 16 ** 8


def make_sampler_class(base: type, labels: dict, orig_log: Path = ORIG_CUR_LOG, mode: str = "ws125") -> type:
    from trainer_v2_v4 import curriculum_v1 as CV
    m = MODES[mode]

    class StillpenSampler(base):
        SHARED = base.SHARED + ("orders",)

        def __init__(self, items: list, plan: Any, log_path: Optional[Path] = None) -> None:
            plan_rows = {r for rows in plan.strata.values() for r in rows}
            super().__init__([it for it in items if not it.row_id.startswith(HCA1_PREFIX) and it.row_id in plan_rows],
                             plan, log_path)
            self.items, self.by_id = items, {it.row_id: it for it in items}      # hca1 rows: boost path only
            cf = m.get("conflict") or m.get("streams")
            self.blocked = (plan_rows - set(self.by_id)) | ({r for r in plan_rows if labels.get(r, {}).get("conflict")}
                                                            if cf else set())
            self.conflict_fallbacks = 0
            self.stream_fallbacks = Counter()
            self.mst = {"visits": {}, "seen": [], "status": {}, "queue": [], "stats": {}, "last": {}, "fact_queue": [], "fact_done": [], "fact_allfail": [], "fact_results": []}
            self.cf_src, self.cf_dropped = {}, set()
            rows_t = tier_rows() if m.get("bands") else {}
            self.cf_tier = {r: v["tier"] for r, v in rows_t.items()}
            self.cf_diff = {r: float(v["difficulty"]) for r, v in rows_t.items()}
            self.fresh_key = {}                    # p2e: (tier rank, difficulty bucket, stratified position, tiebreak)
            for t in TIER_RANK:
                for b in sorted({round(8 * self.cf_diff[r]) for r, v in self.cf_tier.items() if v == t}):
                    for mv in (True, False):
                        grp = sorted((r for r, v in self.cf_tier.items() if v == t and round(8 * self.cf_diff[r]) == b
                                      and bool(labels.get(r, {}).get("move_req")) == mv), key=_u_row)
                        self.fresh_key.update({r: (TIER_RANK[t], b, (i + 0.5) / len(grp), _u_row(r))
                                               for i, r in enumerate(grp)})
            if m.get("mastery"):
                ACTIVE["sampler"] = self
            if m["restore"]:
                hit = None
                for line in Path(orig_log).read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if rec["step"] == ORIG_STATE_STEP:
                        hit = rec
                if hit is None or hit["cursor_after"] != ORIG_CURSOR or hit["schedule_sha256"] != plan.sha:
                    raise SystemExit("REFUSED: original curriculum state (step 124, cursor 628, same schedule) not found")
                self.load_state(hit["state"])
            names = [p for p, _ in m["pools"]] + list(m["fallback"]) + (["conflict"] if m.get("conflict") else []) + \
                (list(m["streams"]) if m.get("streams") else [])
            bal = m.get("balance")
            names += [p for p, _ in bal["types"]] if bal else []
            names = list(dict.fromkeys(names))
            self.orders = {p: boost_order(labels, set(self.by_id), p) for p in names}
            self.labels_type = {r: ("still" if v["still_ok"] else "other") for r, v in labels.items()
                                if v["acute"] and not v.get("hca1")}
            self.subst, self.n_type, self.bal_fallbacks = {}, Counter(), 0
            self.pos, self.boost_front, self.boost_last = {p: 0 for p in names}, [], {}
            self.boost_n, self.n_by_pool, self.fallbacks = 0, Counter(), 0

        def set_step(self, step: int) -> None:
            super().set_step(step + m["offset"])
            self.subst = {}
            self.fact_step_n = 0
            if m.get("mastery"):
                HOT.clear()
                self.cf_src, self.cf_dropped = {}, set()
                st = self.mst
                for e in [e for e in st["queue"] if e["deadline"] < self.step]:
                    st["queue"].remove(e)
                    st["status"][e["row_id"]] = "expired"
                    self._bump("expired")
                for e in [e for e in st.get("fact_queue", []) if e["deadline"] < self.step]:
                    st["fact_queue"].remove(e)
                    self._bump("fact_expired")

        def _bump(self, k: str, d: int = 1) -> None:
            self.mst["stats"][k] = self.mst["stats"].get(k, 0) + d

        def band(self) -> Optional[int]:
            """p2e: 1 during the first band1_steps loop steps, else 2 (None without bands)."""
            b = m.get("bands")
            return None if not b else (1 if self.step - m["offset"] < b["band1_steps"] else 2)

        def _allowed(self, r: str) -> bool:
            if not m.get("bands"):
                return True
            t = self.cf_tier.get(r)
            return t in TIER_RANK and (self.band() != 1 or t in m["bands"]["band1_tiers"])

        def _max_visits(self, r: str) -> int:
            mc = m["mastery"]
            return min(mc["max_visits"], m["bands"]["max_visits_by_tier"].get(self.cf_tier.get(r), mc["max_visits"])) \
                if m.get("bands") else mc["max_visits"]

        def _mastery_pick(self) -> Optional[tuple[str, str]]:
            mc, st = m["mastery"], self.mst
            now = {d[0] for d in self.step_draws} | {v["y"] for v in self.subst.values() if v["y"]}
            first_replay = _u3(self.step, len(self.step_draws)) < mc["p_replay"]
            for side in (("replay", "fresh") if first_replay else ("fresh", "replay")):
                if side == "replay":
                    el = [e for e in st["queue"] if e["due"] <= self.step <= e["deadline"] and e["row_id"] not in now
                          and st["visits"].get(e["row_id"], 0) < self._max_visits(e["row_id"]) and self._boost_ok(e["row_id"], now)
                          and self._allowed(e["row_id"])]
                    if not el:
                        fr = self._fact_pick(now)
                        if fr is None:
                            continue
                        return fr, "conflict"
                    e = min(el, key=lambda e: (-e["priority"], e["due"], e["row_id"]))
                    st["queue"].remove(e)
                    r = e["row_id"]
                    hot = False
                    if e["no_signal"]:
                        n_all = st["stats"].get("n_draws", 0) + 1
                        if st["stats"].get("hot", 0) + 1 <= mc["hot_cap_frac"] * n_all:
                            HOT.add(r)
                            hot = True
                            self._bump("hot")
                        else:
                            self._bump("hot_skipped_cap")
                    self.cf_src[r] = {"src": "replay", "entry": e, "hot": hot}
                    self._bump("draw_replay")
                else:
                    seen, order, n = set(st["seen"]), self.orders["conflict"], len(self.orders["conflict"])
                    if m.get("bands"):                  # easiest allowed fresh row, ties in round-robin order
                        cand = [(self.fresh_key[x], x) for k in range(n)
                                for x in [order[(self.pos["conflict"] + k) % n]]
                                if x not in seen and self._allowed(x) and self._boost_ok(x, now)]
                        r = min(cand)[1] if cand else None
                    else:
                        r = next((order[(self.pos["conflict"] + k) % n] for k in range(n)
                                  if order[(self.pos["conflict"] + k) % n] not in seen
                                  and self._boost_ok(order[(self.pos["conflict"] + k) % n], now)), None)
                    if r is None:
                        continue
                    self.pos["conflict"] = (order.index(r) + 1) % n
                    st["seen"].append(r)
                    self.cf_src[r] = {"src": "fresh", "entry": None, "hot": False}
                    self._bump("draw_fresh")
                st["visits"][r] = st["visits"].get(r, 0) + 1
                if m.get("bands"):
                    d = self.cf_tier[r]
                    self.cf_src[r] = {**self.cf_src[r], "band": self.band(), "tier": d,
                                      "difficulty": self.cf_diff.get(r)}
                    self._bump(f"draw_{d}")
                return r, "conflict"
            self.stream_fallbacks["conflict"] += 1
            return None

        def _fact_pick(self, now: set) -> Optional[str]:
            """factpc_v1: a due mixed fact-gap row in a replay slot the conflict queue cannot use (no new slot, no change to other draws).
            <= per_step_max per step; one replay per row; 10-step gap (_boost_ok); no tier / MIE bookkeeping (cf_src.fact marks it)."""
            st, fc = self.mst, FACT_REPLAY
            fq = st.setdefault("fact_queue", [])
            if not fq or getattr(self, "fact_step_n", 0) >= fc["per_step_max"]:
                return None
            el = [e for e in fq if e["due"] <= self.step <= e["deadline"] and e["row_id"] not in now and self._boost_ok(e["row_id"], now)]
            if not el:
                return None
            e = min(el, key=lambda e: (e["due"], e["row_id"]))
            fq.remove(e)
            r = e["row_id"]
            st["visits"][r] = st["visits"].get(r, 0) + 1
            self.cf_src[r] = {"src": "fact_replay", "entry": e, "hot": False, "fact": True}
            self._bump("draw_fact_replay")
            self.fact_step_n = getattr(self, "fact_step_n", 0) + 1
            return r

        def _fact_observe(self, o: dict) -> None:
            """Queue a non-conflict row for ONE targeted replay iff its group (same prompt, same forced/on-demand condition) contained BOTH a safe
            full-use answer (hit) and a fact omission (miss: cited, not withheld, no other blocker).  Groups with eligible rollouts but no hit
            are logged as all-fail (row, step), never replayed here."""
            st, fc, r = self.mst, FACT_REPLAY, o["row_id"]
            fl = o.get("fact_labels") or []
            n_hit, n_miss, n_el = fl.count("hit"), fl.count("miss"), sum(x != "na" for x in fl)
            if not n_el:
                return
            if not n_hit:
                st.setdefault("fact_allfail", []).append([r, self.step, n_el, n_miss])
                self._bump("fact_allfail")
                return
            if n_miss:
                self._bump("fact_mixed_available")
            done, fq = st.setdefault("fact_done", []), st.setdefault("fact_queue", [])
            if r in done or any(e["row_id"] == r for e in fq) or not n_miss:
                return
            done.append(r)
            fq.append({"row_id": r, "src": self.step, "due": self.step + fc["min_delay"], "deadline": self.step + fc["max_delay"],
                       "reason": "mixed", "n_hit": n_hit, "n_miss": n_miss, "n_el": n_el})
            self._bump("fact_queued")

        def _mastery_observe(self, outcomes: Any) -> None:
            mc, st = m["mastery"], self.mst
            for o in sorted(outcomes, key=lambda o: o["row_id"]):
                r = o["row_id"]
                if o.get("fact_labels") is not None and not (r in self.cf_src):
                    self._fact_observe(o)
                if r not in self.cf_src:
                    continue
                if self.cf_src[r].get("fact"):                  # factpc: the one targeted replay of a fact-miss row; no MIE bookkeeping
                    self._bump("fact_replayed")
                    fl = o.get("fact_labels") or []
                    st.setdefault("fact_results", []).append([r, self.step, fl.count("hit"), sum(x != "na" for x in fl)])
                    st["last"][r] = {"step": self.step, "src": "fact_replay", "hot": False, "fact_labels": o.get("fact_labels")}
                    continue
                judged = [x for x in (o.get("mie_labels") or []) if x != "unjudged"]
                mastery = sum(x == "applied" for x in judged) / len(judged) if judged else 0.0
                no_signal = r in self.cf_dropped or not judged or len(set(judged)) == 1
                st["queue"] = [e for e in st["queue"] if e["row_id"] != r]
                if judged and mastery >= mc["mastery"] - 1e-9:
                    status = "mastered"
                elif st["visits"].get(r, 0) >= self._max_visits(r):
                    status = "parked"
                else:
                    status = "no_signal" if no_signal else "failed"
                    st["queue"].append({"row_id": r, "src": self.step, "due": self.step + mc["min_delay"],
                                        "deadline": self.step + mc["max_delay"], "priority": round(1.0 - mastery, 6),
                                        "mastery": round(mastery, 6), "no_signal": no_signal})
                st["status"][r] = status
                self._bump(status)
                st["last"][r] = {"step": self.step, "src": self.cf_src[r]["src"], "hot": self.cf_src[r]["hot"],
                                 "mastery": round(mastery, 6), "no_signal": no_signal, "status": status,
                                 "n_judged": len(judged)}

        def mastery_counts(self) -> dict:
            c = Counter(self.mst["status"].values())
            return {"mastered": c["mastered"], "failed": c["failed"], "no_signal": c["no_signal"],
                    "parked": c["parked"], "expired": c["expired"], "queue": len(self.mst["queue"]),
                    "seen": len(self.mst["seen"]),
                    **({"fact_queue": len(self.mst["fact_queue"]), "fact_allfail_rows": self.mst.get("fact_allfail", []), "fact_results": self.mst.get("fact_results", [])} if self.mst.get("fact_queue") or self.mst.get("fact_allfail") or self.mst.get("fact_results") or self.mst["stats"].get("draw_fact_replay") else {}),
                    "fresh_left": len(self.orders.get("conflict", [])) - len(self.mst["seen"]),
                    "max_visits": max(self.mst["visits"].values(), default=0), **self.mst["stats"],
                    **({"band": self.band()} if m.get("bands") else {})}

        def refund(self, unused: list[int]) -> list[str]:
            undone = []
            if m.get("mastery"):
                self._bump("n_draws", -len(unused))
                for i in unused:
                    v = self.subst.get(i)
                    r = v["y"] if v and v["type"] == "conflict" else None
                    if r is not None and r in self.cf_src:
                        s = self.cf_src.pop(r)
                        self.mst["visits"][r] -= 1
                        if s.get("fact"):
                            self.mst["fact_queue"].append(s["entry"])
                            self._bump("draw_fact_replay", -1)
                            self.fact_step_n = max(0, getattr(self, "fact_step_n", 0) - 1)
                        elif s["src"] == "fresh":
                            self.mst["seen"].remove(r)
                            self._bump("draw_fresh", -1)
                        else:
                            self.mst["queue"].append(s["entry"])
                            self._bump("draw_replay", -1)
                        if s.get("tier"):
                            self._bump(f"draw_{s['tier']}", -1)
                        if s["hot"]:
                            self._bump("hot", -1)
                        HOT.discard(r)
                        undone.append(r)
            back = self._refund_base(unused)
            if undone:
                self.boost_front = [x for x in self.boost_front if not (x[1] == "conflict" and x[0] in undone)]
            return back

        def _balance(self, x: str) -> dict:
            """Type re-draw for the curriculum's acute row ``x`` at the current position -> record {x, y (None = keep
            x), type, fb, prev}."""
            bal, tx = m["balance"], self.labels_type[x]
            u, acc, t = _u(self.step, len(self.step_draws) - 1), 0.0, bal["types"][-1][0]
            for name, w in bal["types"]:
                acc += w
                if u < acc:
                    t = name
                    break
            now = {d[0] for d in self.step_draws} | {v["y"] for v in self.subst.values() if v["y"]}
            for p in [t] + [q for q in bal["fallback"] if q != t]:
                if p == tx:
                    return {"x": x, "y": None, "type": tx, "fb": p != t, "prev": None}
                r = self._from(p, now)
                if r is not None:
                    return {"x": x, "y": r, "type": p, "fb": p != t, "prev": self.boost_last.get(r)}
            return {"x": x, "y": None, "type": tx, "fb": True, "prev": None}

        def _ok(self, r: str) -> bool:
            b = self.boost_last.get(r)
            return super()._ok(r) and (b is None or self.step - b >= CV.MIN_GAP_STEPS)

        def _boost_ok(self, r: str, now: set) -> bool:
            return r not in now and self._ok(r)

        def _elig(self, r: str, *a: Any, **kw: Any) -> bool:
            """p2b: the curriculum never emits a blocked row (excluded-family rows; conflict rows in mode p2b)."""
            return r not in self.blocked and super()._elig(r, *a, **kw)

        def _stream_pick(self, pool: str) -> Optional[tuple[str, str]]:
            """p2c: the stream's refunded front first, else its pool (10-step gap); None -> curriculum (counted)."""
            now = {d[0] for d in self.step_draws} | {v["y"] for v in self.subst.values() if v["y"]}
            for j, (r, p) in enumerate(self.boost_front):
                if p == pool and self._boost_ok(r, now):
                    return tuple(self.boost_front.pop(j))
            r = self._from(pool, now)
            if r is None:
                self.stream_fallbacks[pool] += 1
                return None
            return r, pool

        def _conflict_pick(self) -> Optional[tuple[str, str]]:
            now = {d[0] for d in self.step_draws}
            for j, (r, p) in enumerate(self.boost_front):
                if p == "conflict" and self._boost_ok(r, now):
                    return tuple(self.boost_front.pop(j))
            r = self._from("conflict", now)
            if r is None:
                self.conflict_fallbacks += 1               # every conflict row within the gap: curriculum draw
                return None
            return r, "conflict"

        def _from(self, pool: str, now: set) -> Optional[str]:
            order, n = self.orders[pool], len(self.orders[pool])
            for k in range(n):
                r = order[(self.pos[pool] + k) % n]
                if self._boost_ok(r, now):
                    self.pos[pool] = (self.pos[pool] + k + 1) % n
                    return r
            return None

        def _boost_pick(self) -> Optional[tuple[str, str]]:
            now = {d[0] for d in self.step_draws}
            for j, (r, p) in enumerate(self.boost_front):
                if p not in ("conflict", "g0") and self._boost_ok(r, now):
                    return tuple(self.boost_front.pop(j))
            u, acc, first = _u(self.step, len(self.step_draws)), 0.0, 0
            for j, (_, w) in enumerate(m["pools"]):
                acc += w
                if u < acc:
                    first = j
                    break
            names = [p for p, _ in m["pools"]]
            for p in names[first:] + names[:first] + list(m["fallback"]):
                r = self._from(p, now)
                if r is not None:
                    self.fallbacks += p in m["fallback"]
                    return r, p
            return None

        def draw(self, k: int) -> list:
            out = []
            for _ in range(max(k, 0)):
                pos, cf = len(self.step_draws), m.get("conflict")
                if m["every"] and pos % m["every"] == 0:
                    got = self._boost_pick()
                elif cf and pos % cf["every"] == cf["offset"]:
                    got = self._conflict_pick()
                elif m.get("streams") and m.get("mastery"):
                    u, sm = _u2(self.step, pos), m["streams"]
                    pool = "conflict" if u < sm["conflict"] else "g0" if u < sm["conflict"] + sm["g0"] else None
                    if pool is None:
                        got = None
                    else:
                        x = super().draw(1)[0]             # the curriculum row of this position (consumed)
                        ok = pmr(x) if pool == "conflict" else not x.target.get("withhold")
                        y = ((self._mastery_pick() if pool == "conflict" else self._stream_pick("g0"))
                             if ok else None)
                        if not ok:
                            self._bump("displace_skip_" + pool)
                        if y is not None:
                            r, i = y[0], len(self.step_draws) - 1
                            self.subst[i] = {"x": x.row_id, "y": r, "type": pool, "fb": False,
                                             "prev": self.boost_last.get(r)}
                            self.boost_last[r] = self.step
                            self.n_type[pool] += 1
                            self.events.append(["p2d_subst", x.row_id, r, pool])
                            self._bump("subst_" + pool)
                            x = self.by_id[r]
                        out.append(x)
                        continue
                elif m.get("streams"):
                    u, sm = _u2(self.step, pos), m["streams"]
                    got = (self._stream_pick("conflict") if u < sm["conflict"] else
                           self._stream_pick("g0") if u < sm["conflict"] + sm["g0"] else None)
                else:
                    got = None
                if got is None:
                    it = super().draw(1)[0]
                    if m.get("balance") and self.meta_acute.get(it.row_id):
                        v = self._balance(it.row_id)
                        self.subst[len(self.step_draws) - 1] = v
                        self.n_type[v["type"]] += 1
                        self.bal_fallbacks += v["fb"]
                        if v["y"]:
                            self.boost_last[v["y"]] = self.step
                            self.events.append(["balance_subst", it.row_id, v["y"], v["type"]])
                            it = self.by_id[v["y"]]
                    out.append(it)
                    continue
                r, p = got
                self.step_draws.append([r, BOOST, {"prev": self.boost_last.get(r), "pool": p}])
                self.boost_last[r] = self.step
                self.boost_n += 1
                self.n_by_pool[p] += 1
                out.append(self.by_id[r])
            if m.get("mastery"):
                self._bump("n_draws", len(out))
            return out

        def _refund_base(self, unused: list[int]) -> list[str]:
            for i in sorted(unused, reverse=True):          # balance substitutions: undo the replacement row
                v = self.subst.pop(i, None)
                if v is not None:
                    if v["y"] and v["prev"] is None:
                        self.boost_last.pop(v["y"])
                    elif v["y"]:
                        self.boost_last[v["y"]] = v["prev"]
                    self.n_type[v["type"]] -= 1
                    self.bal_fallbacks -= v["fb"]
            self.subst = {i - sum(j < i for j in unused): v for i, v in self.subst.items()}
            boost = [i for i in sorted(unused) if self.step_draws[i][1] == BOOST]
            rest = [i for i in unused if i not in boost]
            back = super().refund(rest)
            shift = [i - sum(j < i for j in rest) for i in boost]
            for i in reversed(shift):
                r, _, pe = self.step_draws[i]
                if pe["prev"] is None:
                    self.boost_last.pop(r)
                else:
                    self.boost_last[r] = pe["prev"]
                self.boost_n -= 1
                self.n_by_pool[pe["pool"]] -= 1
                self.fallbacks -= pe["pool"] in m["fallback"]
                self.boost_front.insert(0, [r, pe["pool"]])
            gone = set(shift)
            self.step_draws = [d for i, d in enumerate(self.step_draws) if i not in gone]
            if shift:
                self.events.append(["boost_refunded", [x[0] for x in self.boost_front[:len(shift)]]])
            return back

        def _boosted(self) -> set:
            return {r for r, kind, _ in self.step_draws if kind == BOOST} | {v["y"] for v in self.subst.values() if v["y"]}

        def requeue(self, filtered: list) -> list:
            if m.get("mastery"):                       # DAPO-dropped conflict groups -> no_signal (observe)
                self.cf_dropped |= {x[0] for x in filtered if x[0] in self.cf_src}
            b = self._boosted()
            return super().requeue([x for x in filtered if x[0] not in b])

        def observe(self, outcomes: Any) -> list:
            if m.get("mastery"):
                self._mastery_observe(outcomes)
            b = self._boosted()
            return super().observe([o for o in outcomes if o["row_id"] not in b])

        def payload(self, batch: list) -> dict:
            known = [it for it in batch if it.row_id in self.of]
            return {**super().payload(known), "train/data/cur_n_hca1": len(batch) - len(known)}

        def state(self) -> dict:
            return {**super().state(), "boost": {"pos": self.pos, "front": self.boost_front, "last": self.boost_last,
                                                 "n": self.boost_n, "n_by_pool": dict(self.n_by_pool),
                                                 "fallbacks": self.fallbacks,
                                                 **({"conflict_fallbacks": self.conflict_fallbacks}
                                                    if m.get("conflict") else {}),
                                                 **({"stream_fallbacks": dict(self.stream_fallbacks)}
                                                    if m.get("streams") else {})},
                    "balance": {"n_type": dict(self.n_type), "fallbacks": self.bal_fallbacks},
                    **({"mastery": self.mst} if m.get("mastery") else {})}

        def load_state(self, st: dict) -> None:
            super().load_state(st)
            b = st.get("boost") or {}
            if b:                                      # the original run's logged state has no boost block
                self.pos, self.boost_front = dict(b["pos"]), [list(x) for x in b["front"]]
                self.boost_last, self.boost_n = dict(b["last"]), b["n"]
                self.n_by_pool, self.fallbacks = Counter(b["n_by_pool"]), b["fallbacks"]
                self.conflict_fallbacks = b.get("conflict_fallbacks", 0)
                self.stream_fallbacks = Counter(b.get("stream_fallbacks") or {})
            if st.get("mastery"):
                self.mst = json.loads(json.dumps(st["mastery"]))
            bl = st.get("balance") or {}
            if bl:
                self.n_type, self.bal_fallbacks = Counter(bl["n_type"]), bl["fallbacks"]

        def summary(self) -> dict:
            return {**super().summary(), "stillpen_boost": {"mode": mode, **m, "draws": self.boost_n,
                                                            "by_pool": dict(self.n_by_pool), "fallbacks": self.fallbacks,
                                                            "pool_sizes": {p: len(o) for p, o in self.orders.items()},
                                                            "balance_types": dict(self.n_type),
                                                            "balance_fallbacks": self.bal_fallbacks,
                                                            "conflict_fallbacks": self.conflict_fallbacks,
                                                            "stream_fallbacks": dict(self.stream_fallbacks),
                                                            "blocked_rows": len(self.blocked)}}
    return StillpenSampler


def make_plan_class(base: type) -> type:
    """Plan22 (balance_ae) over the rows WITHOUT the hca1 rows: identical schedule / sha; of_rows() and check() also
    accept the hca1 ids so main_v4hcur21's item / release-id checks pass with the hca1 release."""

    class Hca1Plan(base):
        def __init__(self, items: list, acute_ids: set, *a: Any, **kw: Any) -> None:
            self.hca1 = sorted(it.row_id for it in items if it.row_id.startswith(HCA1_PREFIX))
            super().__init__([it for it in items if not it.row_id.startswith(HCA1_PREFIX)],
                             {r for r in acute_ids if not r.startswith(HCA1_PREFIX)}, *a, **kw)

        def of_rows(self) -> list[str]:
            return super().of_rows() + self.hca1

        def check(self, release_ids: set) -> dict:
            return super().check(set(release_ids) - set(self.hca1))

        def report(self) -> dict:
            return {**super().report(), "hca1_outside_plan": len(self.hca1)}
    return Hca1Plan


def make_plan_class_p2b(base: type, orig_release: Path = ORIG_RELEASE_TRAIN.parents[1],
                        orig_key_actions: Path = ORIG_KEY_ACTIONS) -> type:
    """Plan22 built from the ORIGINAL v2 plan rows (hca1 release minus hca1, hca1b acute ids minus hca1): identical
    schedule / sha / restorable state. of_rows() = plan rows minus the rows missing from the p2b release (excluded
    families) + every p2b row outside the plan (hca1 + new conflict rows); check() runs on the original id set."""
    from eval_v1 import release_data as D
    from trainer_v2_v4 import key_actions_v4h as KA

    class P2bPlan(base):
        def __init__(self, items: list, acute_ids: set, *a: Any, **kw: Any) -> None:
            orig, _ = D.load_release(orig_release, "train", "gt_retrieved_drop", arms=(items[0].arm,))
            orig_ids = {it.row_id for it in orig}
            o_acute = set(KA.load(orig_key_actions, orig_release, "gt_retrieved_drop"))
            self.extra = sorted(it.row_id for it in items if it.row_id.startswith(HCA1_PREFIX) or it.row_id not in orig_ids)
            super().__init__([it for it in orig if not it.row_id.startswith(HCA1_PREFIX)],
                             {r for r in o_acute if not r.startswith(HCA1_PREFIX)}, *a, **kw)
            self.excluded = sorted(set(super().of_rows()) - {it.row_id for it in items})

        def of_rows(self) -> list[str]:
            ex = set(self.excluded)
            return [r for r in super().of_rows() if r not in ex] + self.extra

        def check(self, release_ids: set) -> dict:
            return super().check((set(release_ids) - set(self.extra)) | set(self.excluded))

        def report(self) -> dict:
            return {**super().report(), "p2b_outside_plan": len(self.extra), "p2b_excluded_plan_rows": len(self.excluded)}
    return P2bPlan


def install_sampler(mode: str = "ws125") -> None:
    from trainer_v2_v4 import curriculum_v2_1 as C21
    from trainer_v2_v4 import curriculum_v2_2 as C22
    if getattr(C21.CurriculumSamplerV21, "_stillpen", False) or getattr(C22.Plan22, "_stillpen", False):
        raise RuntimeError("stillpen sampler installed twice")
    cls = make_sampler_class(C21.CurriculumSamplerV21, load_labels(conflict=conflict_ids()), mode=mode)
    cls._stillpen = True
    C21.CurriculumSamplerV21 = cls
    plan = make_plan_class_p2b(C22.Plan22)             # after balance_ae.install (Plan22 = Balanced22)
    plan._stillpen = True
    C22.Plan22 = plan


def install(mode: str = "p2e") -> None:
    install_hf_gate()
    install_penalty()
    install_sampler(mode)
