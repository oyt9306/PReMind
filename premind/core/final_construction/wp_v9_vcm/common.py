"""S2 v9 VCM canary: shared paths, pinned constants, config, write-once IO and seeded helpers (CPU only).

Binding spec: docs/SPEC_S2V9_CONCEPT_ANCHORED_VCM_DRAFT_2026-10-02.md (R2 + owner decisions + addendum decisions) and
the owner correction of 2026-10-02 (retrieval-aware stores; offline top-5 stored per row; labels by rule from the
stored context; legacy CLIP contexts with their own rule labels). Holdout files are never opened by this package.
"""
from __future__ import annotations

import hashlib
import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parent
V3 = FC.parent
PROD = V3.parent
V2 = PROD / "benchmark_v2"
RELEASE_OPEN = FC / "wp_g_final_benchmark_v1/s2_run_v8/release_v1_0_1/open"
RELEASE_FILES = {"train": RELEASE_OPEN / "final_train_v1_0_1.json", "dev": RELEASE_OPEN / "final_dev_v1_0_1.json"}
AUDIT_DIR = FC / "retriever_audit_v1"
REFS = AUDIT_DIR / "inputs/open_image_to_refs.json"          # query image -> governed_object_references (train/dev)
QUARANTINE = PROD / "_archive_2026-10-01/custom101/custom101_quarantined_images.json"
CUSTOM101_DIR = Path(os.environ.get("PREMIND_CUSTOM101_DIR", "custom101"))  # construction only
COLLAB = V3 / "docs/PIPELINE_AUDIT_HANDOFF_2026-09-29/COLLABORATION.md"
SPEC = V3 / "docs/SPEC_S2V9_CONCEPT_ANCHORED_VCM_DRAFT_2026-10-02.md"
DIALOGUE_SRC = Path(os.environ.get("PREMIND_DIALOGUE_SRC", "generate_dialogue.py"))  # construction only
CAP_GEN_SRC = Path(os.environ.get("PREMIND_CAP_GEN_SRC", "gpt_cap_gen.py"))  # construction only
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
QWEN8B = HUB / "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ZS_MODELS = {"4b": HUB / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             "9b": HUB / "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
OUT_DEFAULT = HERE / "canary_v9"
USAGE_DEFAULT = HERE / "usage_s2v9_canary.jsonl"
PREREG_DEFAULT = HERE / "PREREG_S2V9_CANARY.json"

RUN_ID = "s2v9_vcm_canary_v1"
SEED = 20261002
ENTRY_SIZE = 448                       # R2 B4/M5: every entry image 448x448, one padding rule
PAD_RGB = (128, 128, 128)
# FINAL K RULE (owner-delegated, supersedes per-concept K): the governed-concept count is unknown at inference, so
# K is FIXED for every image (K + 1 = 9 images per prompt incl. the query). Retrieval = the selected retriever over the
# whole query image plus generic detector crops (USE_CROPS; keep only if the retriever audit shows crops help), merged
# round-robin by rank across query views, deduped, top-K. Labels by rule from the stored K.
K = 8
USE_CROPS, MAX_CROPS = True, 10       # = retriever_audit_v1 crop mode (max_boxes 10, whole image first)
STORE_SIZE = {1: 11, 2: 12}            # owner: 10-12 so the K retrieved entries are mostly real candidates
N_NEG = {1: 4, 2: 4}                   # same-fine-type hard negatives per target (owner correction: 3-4)
TOPK_CAND = 8                          # hard negatives sampled (seeded) from the top-k most similar
TARGET_ELSEWHERE_WEIGHT = 5.0          # R2 B4: prefer targets-elsewhere (negatives AND others; CPU smoke: 2.0 left a
                                       # target-frequency role leak, BA 0.63 -> 0.56 at 5.0)
SIM_MIX = 0.5                          # hard-negative score = mix*sim(query image) + (1-mix)*sim(target reference)
FACT_SLOTS = ("s1", "s2")              # M1: every concept carries exactly two facts (s1 decisive, s2 safety/health)
CANARY_N, CANARY_MIN_TWO_TARGET, CANARY_MIN_RELAXED = 10, 2, 1
CANARY_KIND_MIN = {"object": 3, "pet": 2, "person": 3}
DIALOGUE_T, DIALOGUE_MAX_TOKENS, DIALOGUE_MAX_REGEN = 0.7, 1024, 2   # optional Qwen wrapping (owner change 1)
DIALOGUE_WRAPPING_DEFAULT = False
TEXT_STYLES = ("note", "short_dialogue", "record_list")   # seeded per concept: style can never encode the role
DETECTOR = "yoloe (retriever_audit_v1/build/crops.json, reused as-is)"
TOKEN_CAP = 900_000                    # canary paid cap (raised for R5: one review-driven revise round + its re-review)
PER_CALL_RESERVE = 25_000
RETRIEVERS = {"siglip2": {"id": "google/siglip2-so400m-patch14-384", "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
                          "role": "main (owner amendment: audit tie with WeMM at ctxR@8 0.695, faster, person recall "
                                  "0.83 vs 0.58; whole image + detector crops)"},
              "wemm": {"id": "tencent/WeMM-Embedding-2B", "revision": "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
                       "role": "alternative (reported); selected the frozen stores' hard negatives"},
              "clip": {"id": "openai/clip-vit-large-patch14-336", "revision": "hf-cache refs/main",
                       "role": "legacy_ablation_#5 (RAP-MLLM retriever.py semantics: normalised image features, L2)"}}
MAIN_RETRIEVER = "siglip2"             # context arm used for labels, review, render, export
STORE_RETRIEVER = "wemm"               # frozen: hard-negative selection + finetype nn of the authored plan (plan_sha cc8154fa..)
PLAN_RETRIEVERS = ("wemm", "clip")     # arms stored inside plan.json (unchanged)
EXTRA_RETRIEVERS = ("siglip2",)        # arms added by the `retrieve` phase (s1/retrieval_<r>.json), plan.json untouched
AUDIT_CROPS = AUDIT_DIR / "build/crops.json"   # YOLOE-v8l-seg substitute, conf 0.1, <= 10 boxes, raw boxes (audit)
THRESHOLDS = {
    "v8a_positive_acceptance_min": 0.80, "v8a_control_acceptance_min": 0.80,
    "i1_i7_pass_on_accepted": 1.0,
    "composition_probe_ba_max": 0.60,
    "text_only_identification_max": round(1 / 3 + 0.10, 4),
    "swap_test_min": 1.0,
    "retriever_label_rule_consistency_min": 1.0,
    "role_probe_ba_max": 0.60,
    "identification_band": [0.40, 0.90],
    "query_only_pair_accuracy_reported_only": 0.25,          # owner change 3: reported audit, not a gate
    "degeneration_gate_tokens": 512,
    "max_model_len_budget": 8192,
}
EXIT_BUDGET, EXIT_GATE = 3, 4


# Owner HYBRID split (per-stage backend field; "qwen" = local Qwen3-VL-8B BF16 GPU job, "gpt" = paid gpt-6-sol).
# F1/F6: memory text (memtext_family) and the one repair round (revise) are quality-critical -> GPT; selfcheck is a
# diagnostic only (F2).
BACKENDS = {"describe": "qwen", "salvage_rewrite": "qwen", "selfcheck": "qwen", "author": "gpt", "revise": "gpt",
            "memtext_family": "gpt", "review_primary": "gpt", "review_second": "qwen", "adjudicate": "gpt",
            "revise_review": "gpt", "author_delta": "gpt", "extra_text": "gpt"}


@dataclass(frozen=True)
class Cfg:
    out: Path = OUT_DEFAULT
    usage_log: Path = USAGE_DEFAULT
    prereg: Path = PREREG_DEFAULT
    collab: Path = COLLAB
    release_files: dict = field(default_factory=lambda: dict(RELEASE_FILES))
    open_files: dict = field(default_factory=lambda: dict(RELEASE_FILES))      # holdout identity rule: train/dev
    refs: Path = REFS
    quarantine: Path = QUARANTINE
    audit_crops: Path = AUDIT_CROPS
    custom101_dir: Path = CUSTOM101_DIR
    cap: int = TOKEN_CAP
    mock: bool = False
    backends: dict = field(default_factory=lambda: dict(BACKENDS))
    round: int = 1                     # R5: 1 = first pass; 2 = review-driven revise round (versioned outputs)
    run_id: str = RUN_ID
    profile: str = "canary"
    s0_dir: Path | None = None         # prod: the canary's stage-0 tree, shared read-only (pool, types, embeddings)
    pilot: int | None = None           # P3: only the first N families of the plan
    delta: bool = False                # P2: delta retries (failing items + touched facts only)
    chunk: int | None = None           # production run: 1-based chunk of PROD_CHUNK families after the pilot
    s1_dir: Path | None = None         # production chunks: the shared prod_v9/s1 (plan, overlays, stop files)
    regen: str | None = None           # regeneration run: "pilot" or "chunk01" (prod_v9/regen_<regen>/, gen "g2")
    retry: bool = False                # production retry round (round 2 of a final unit; s5_final_r2, s7_retry)

    def p(self, *parts: str) -> Path:
        if parts and parts[0] == "s0" and self.s0_dir is not None:
            return Path(self.s0_dir).joinpath(*parts[1:])
        if parts and parts[0] == "s1" and self.s1_dir is not None:
            return Path(self.s1_dir).joinpath(*parts[1:])
        return self.out.joinpath(*parts)


PROD_RUN_ID = "s2v9_vcm_prod_v1"
PROD_PILOT_N = 30
PROD_PILOT_CAP = 1_400_000             # pilot ledger cap: 30 families x 35k target x ~1.33 headroom
PROD_TOKENS_PER_FAMILY_MAX = 35_000    # P3 stop rule: projected mean paid tokens per family
# Production run (after the pilot): chunks of PROD_CHUNK families, each run end-to-end into prod_v9/chunk_<KK>/; after
# every chunk `prod_check` (s7_prod) projects the total spend (ledger + remaining families x running mean under the
# current PREREG): > PROD_BUDGET_STOP -> stop file; > PROD_BUDGET_RECOMMEND -> RECOMMEND_NO_REVIEW_ROUND flag; any
# row type's accepted unsafe share < PROD_COVERAGE_STOP -> stop. The ledger is the pilot's, so PROD_CAP (hard ceiling)
# bounds pilot + production: <= $75 at PRICE_USD_PER_M = 4.9 $/1M -> 15.3M tokens.
PROD_CHUNK = 50
PROD_CAP = 19_000_000                  # ~ $93 hard cap: train/dev incl. retry + acute/G0 repair round (owner 2026-10-03; was 18.5M)
PRICE_USD_PER_M = 4.9
PROD_BUDGET_STOP = 18_900_000          # ~ $93 total train/dev incl. the repair round (owner 2026-10-03; was 18.0M)
PROD_BUDGET_RECOMMEND = 11_600_000     # ~ $57: run the following chunks with --no-review-round
PROD_COVERAGE_STOP = 0.20
# Sealed VCM holdout (audit WP-P0b): every holdout stage writes only inside HOLDOUT_ROOT; CLI output is redacted to
# counts and hashes (h_holdout.redact); separate ledger, PREREG, prelaunch line and budget stop.
HOLDOUT_ROOT = HERE / "holdout_v9_sealed"
HOLDOUT_FILE = FC / "wp_g_final_benchmark_v1/s2_run_v8/release_v1_0_1/holdout_sealed/final_holdout_v1_0_1.json"
HOLDOUT_SEAL_V1 = HOLDOUT_FILE.with_name("HOLDOUT_SEAL_MANIFEST_v1_0_1.json")
HOLDOUT_AGGREGATE_V1 = "a91b41a5347610b80f6b486f4209cfe30cf9f54b0cd7b118d73761063d730994"
HOLDOUT_RUN_ID = "s2v9_vcm_holdout_v1"
HOLDOUT_CAP = 10_000_000
HOLDOUT_BUDGET_STOP = 9_500_000          # owner-approved raise 2026-10-03 (≈$47); cap 10.0M unchanged
PROFILES = {"canary": {},
            "prod": {"out": HERE / "prod_v9", "usage_log": HERE / "usage_s2v9_prod.jsonl",
                     "prereg": HERE / "PREREG_S2V9_PROD.json", "run_id": PROD_RUN_ID, "profile": "prod",
                     "s0_dir": OUT_DEFAULT / "s0", "cap": PROD_PILOT_CAP, "delta": True},
            "holdout": {"out": HOLDOUT_ROOT, "usage_log": HERE / "usage_s2v9_holdout.jsonl",
                        "prereg": HERE / "PREREG_S2V9_HOLDOUT.json", "run_id": HOLDOUT_RUN_ID, "profile": "holdout",
                        "cap": HOLDOUT_CAP, "delta": True, "release_files": {"holdout": HOLDOUT_FILE},
                        "refs": HOLDOUT_ROOT / "s0" / "holdout_image_to_refs.json",
                        "audit_crops": HOLDOUT_ROOT / "s0" / "holdout_crops.json"}}


class BudgetBlocked(RuntimeError):
    """Paid call refused by the token cap."""


def round_tag(cfg: Cfg) -> str:
    return "" if cfg.round == 1 else f"_r{cfg.round}"


def rid(cfg: Cfg, fid: str) -> str:
    """Paid-request id of a family in the current round (a consumed round-1 tag is never re-used); a regeneration run
    tags "@g2" so its calls never collide with the original generation's consumed tags."""
    base = f"{fid}@{GEN_REGEN}" if cfg.regen else fid
    if holdout_gen(cfg) > 1:                          # holdout re-author (owner 2026-10-03): fresh "@h<g>" tags
        base = f"{base}@h{holdout_gen(cfg)}"
    return base if cfg.round == 1 else f"{base}@r{cfg.round}"


GEN_ORIGINAL, GEN_REGEN = "g1", "g2"
REGEN_TARGETS = {"pilot": (0, PROD_PILOT_N), **{f"chunk{k:02d}": (PROD_PILOT_N + PROD_CHUNK * (k - 1),
                                                                   PROD_PILOT_N + PROD_CHUNK * k) for k in (1, 2, 3)}}


RETRY_FIRST_CHUNK = 4                  # retry units: regen pilot + regen chunks 1-3 (g2) and chunks 4..last (g1)
FIX_ROUND = 3                          # acute/G0 repair round of the final units (`--round 3 --retry`, s2_fix)
FIX_ROUNDS = (3, 4)                    # pass 1 (round 3) and pass 2 (round 4: round-3 affected rows not accepted)


EXTRA_ROUND = 7                        # additive rows (safe generic G0 + G1_panic_memory), text-only GPT (s2_extra)
REREVIEW_ROUND = 9                     # re-review of the rejected safe generic rows of round 7 (s5_rereview)
EXTRA2_ROUND = 10                      # extended safe generic G0 rows (owner 2026-10-03: query-only must not predict)
BALANCE_TAG_V5 = "_r11"                # dataset-wide label-independent query assignment after round 10 (code-only)
ADDITIVE_ROUNDS = (EXTRA_ROUND, REREVIEW_ROUND, EXTRA2_ROUND)


def holdout_gen(cfg: "Cfg") -> int:
    """Holdout chunk generation from its tree name (chunk_KK = 1, chunk_KK_g2 = 2; h_holdout.chunk_dir)."""
    import re as _re
    m = _re.search(r"_g(\d+)$", Path(cfg.out).name) if cfg.profile == "holdout" else None
    return int(m.group(1)) if m else 1


def chunk_dir(root: Path, k: int, profile: str = "prod") -> Path:
    """Tree of chunk K; holdout re-author (owner 2026-10-03): s1/HOLDOUT_REAUTHOR.json {"chunks": {"1": 2}} sends
    chunk 1 to chunk_01_g2 (fresh "@h2" tags; the superseded tree stays untouched)."""
    d = Path(root) / f"chunk_{k:02d}"
    f = Path(root) / "s1" / "HOLDOUT_REAUTHOR.json"
    g = int(read(f)["chunks"].get(str(k), 1)) if profile == "holdout" and f.exists() else 1
    return d.with_name(f"{d.name}_g{g}") if g > 1 else d


def is_rereview(cfg: "Cfg") -> bool:
    return cfg.round == REREVIEW_ROUND


def is_fix(cfg: "Cfg") -> bool:
    return cfg.retry and cfg.round in FIX_ROUNDS


def is_extra(cfg: "Cfg") -> bool:
    return cfg.round in (EXTRA_ROUND, EXTRA2_ROUND)


# Owner 2026-10-03: the general-rule control (salvaged general_rule_closed rows) is removed from the benchmark in
# train, dev, eval and the holdout (vague, forced golds). Prod / holdout never build it, ignore its risk request
# (plan.json unchanged), skip the salvage step that only fed it, and release/export filter stored rows.
DROPPED_ROW_TYPES = ("general_rule_closed",)
DROP_REASON = "owner 2026-10-03: general-rule control removed from the benchmark (vague, forced golds)"


def no_general_rule(cfg: "Cfg") -> bool:
    return cfg.profile in ("prod", "holdout")


def gen_version(cfg: Cfg) -> str:
    return GEN_REGEN if cfg.regen else GEN_ORIGINAL


def rows_path(cfg: Cfg) -> Path:
    """Assembled rows of the current round for the main arm (round 1 file name unchanged; round 2 = *_r2)."""
    return cfg.p("s4", f"rows_{MAIN_RETRIEVER}{round_tag(cfg)}.json")


def s5_dir(cfg: Cfg) -> str:
    return "s5" + round_tag(cfg)


def final_tag(cfg: Cfg) -> str:
    """Latest finalized production selection: "_r3" (acute/G0 repair, s5_final_r3/) > "_r2" (retry, s5_final_r2/) >
    "": render/export then read that selection and write s6<tag>/ and release<tag>/ (older ones stay as they are)."""
    for tag in ("_r11", "_r10", "_r9", "_r8", "_r7", "_r6", "_r5", "_r4", "_r3", "_r2"):   # newest selection first
        if cfg.p(f"s5_final{tag}", "gate.json").exists():
            return tag
    return ""


def final_rows_path(cfg: Cfg) -> Path:
    """Rows consumed by render/export/probes: the retry selection (rows_*_final_r2.json) when finalized, else the
    finalized merge of rounds 1-2 when it exists, else round 1."""
    for f in (cfg.p("s4", f"rows_{MAIN_RETRIEVER}_final{final_tag(cfg)}.json"),
              cfg.p("s4", f"rows_{MAIN_RETRIEVER}_final_r2.json"), cfg.p("s4", f"rows_{MAIN_RETRIEVER}_final.json")):
        if f.exists():
            return f
    return cfg.p("s4", f"rows_{MAIN_RETRIEVER}.json")


def final_gate_path(cfg: Cfg) -> Path:
    f = cfg.p(f"s5_final{final_tag(cfg)}", "gate.json")
    return f if f.exists() else cfg.p("s5", "gate.json")


def plan_raw(cfg: Cfg) -> dict:
    """plan.json with the R3 per-concept relations (s1/relations_v2.json) applied when present (plan.json unchanged)."""
    plan = read(cfg.p("s1", "plan.json"))
    if cfg.pilot:
        plan["families"] = plan["families"][:cfg.pilot]                 # P3 pilot: the first N families only
    elif cfg.chunk:                                    # production chunk K (after the pilot); holdout chunk K
        a = (PROD_PILOT_N if cfg.profile == "prod" else 0) + PROD_CHUNK * (cfg.chunk - 1)
        plan["families"] = plan["families"][a:a + PROD_CHUNK]
    elif cfg.regen:                                                      # regeneration: the same family ids
        a, b = REGEN_TARGETS[cfg.regen]
        plan["families"] = plan["families"][a:b]
    rel = cfg.p("s1", "relations_v2.json")
    if rel.exists():
        doc = read(rel)
        if doc["plan_sha256"] != sha_file(cfg.p("s1", "plan.json")):
            raise RuntimeError(f"{rel} was computed for another plan.json")
        for fam in plan["families"]:
            for k, r in doc["families"].get(fam["family_id"], {}).items():
                fam["concepts"][k]["relation"] = r["relation"]
    for v in (2, 3):                       # P6 labels (v2 pilot) + v3 neutral handles after the pilot (overlay)
        lab = cfg.p("s1", f"concept_labels_v{v}.json")
        if not lab.exists():
            continue
        doc = read(lab)
        if doc["plan_sha256"] != sha_file(cfg.p("s1", "plan.json")):
            raise RuntimeError(f"{lab} was computed for another plan.json")
        for fam in plan["families"]:
            for k, r in doc["families"].get(fam["family_id"], {}).items():
                fam["concepts"][k].update(name=r["name"], relation=r["relation"], handle=r["handle"])
    if cfg.regen == "pilot":            # pilot regen: v3 handles + v2 risk requests, as for every production family
        from final_construction.wp_v9_vcm import s7_regen
        s7_regen.apply_pilot_overlay(cfg, plan)
    rv2 = cfg.p("s1", "risk_plan_v2.json")                # pilot fix: raised unsafe requests after the pilot only
    if rv2.exists():
        doc = read(rv2)
        if doc["plan_sha256"] != sha_file(cfg.p("s1", "plan.json")):
            raise RuntimeError(f"{rv2} was computed for another plan.json")
        for fam in plan["families"]:
            fam["risk_plan"] = doc["families"].get(fam["family_id"], fam["risk_plan"])
    if no_general_rule(cfg):                              # general-rule control removed: its request is ignored
        for fam in (f for f in plan["families"] if "risk_plan" in f):
            fam["risk_plan"] = {k: v for k, v in fam["risk_plan"].items() if k not in DROPPED_ROW_TYPES}
    return plan


def load_plan(cfg: Cfg, require_main: bool = True) -> dict:
    """plan_raw with the EXTRA_RETRIEVERS arms merged into every family's `retrieval` (plan.json itself is never
    modified, so the paid author stage's plan hash stays valid)."""
    plan = plan_raw(cfg)
    for r in EXTRA_RETRIEVERS:
        f = cfg.p("s1", f"retrieval_{r}.json")
        if not f.exists():
            if require_main and r == MAIN_RETRIEVER:
                raise RuntimeError(f"{f} missing: run `embed --retriever {r}` (GPU) and `retrieve --retriever {r}`")
            continue
        doc = read(f)
        if doc["plan_sha256"] != sha_file(cfg.p("s1", "plan.json")):
            raise RuntimeError(f"{f} was computed for another plan.json")
        for fam in plan["families"]:
            fam["retrieval"][r] = doc["families"][fam["family_id"]]
    return plan


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_once(path: Path, value: object) -> None:
    """Same contract as canary_construct_v1.write_once (refuse to overwrite), atomic via a .partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"write-once output exists: {path}")
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def read(path: Path) -> object:
    return json.loads(Path(path).read_text())


def rng(key: str) -> random.Random:
    return random.Random(int(h(f"{SEED}:{key}")[:16], 16))


def seeded_order(items: list, key: str) -> list:
    return sorted(items, key=lambda x: h(f"{SEED}:{key}:{x}"))


def weighted_sample(items: list[str], weights: list[float], n: int, key: str) -> list[str]:
    """Seeded weighted sampling without replacement (Efraimidis-Spirakis keys)."""
    r = rng(key)
    keyed = sorted(((r.random() ** (1.0 / w), it) for it, w in zip(items, weights)), reverse=True)
    return [it for _, it in keyed[:n]]


def safe_name(row_or_family_id: str) -> str:
    return row_or_family_id.replace(":", "__").replace("/", "_")


def assert_open_only(path: Path | str, holdout_ok: bool = False) -> None:
    """Holdout guard: only the sealed `holdout` profile (holdout_ok) may open holdout files."""
    if "holdout" in str(path).lower() and not holdout_ok:
        raise RuntimeError(f"holdout path refused: {path}")
