"""resplit_v1 configuration: every threshold, target and path lives here (frozen dataclasses, no magic numbers).

Owner decision 2026-10-04: the holdout is not final; re-split RELEASE_TRAIN_DEV_v6d + RELEASE_HOLDOUT_v2 at the
image / leakage-component level into RELEASE_TRAIN_DEV_v7 + RELEASE_HOLDOUT_v3 with matched marginals.
Existing releases are read-only inputs; outputs are write-once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parent
VCM = FC / "wp_v9_vcm"
PROD_ROOT = VCM / "prod_v9"
SEALED_ROOT = VCM / "holdout_v9_sealed"
SRC_TRAIN_DEV = PROD_ROOT / "RELEASE_TRAIN_DEV_v6d"
SRC_HOLDOUT = SEALED_ROOT / "RELEASE_HOLDOUT_v2"
OUT_TRAIN_DEV = PROD_ROOT / "RELEASE_TRAIN_DEV_v7"
OUT_HOLDOUT = SEALED_ROOT / "RELEASE_HOLDOUT_v3"
STAGING = HERE / "out"
AUDIT_KEYS = FC / "holdout_audit_v3" / "results" / "resplit_keys.json"   # scene + view group (read-only)
MODE_MIXED, MODE_SCENE = "mixed", "scene_disjoint"
COLLAB_LOG = FC.parent / "docs" / "PIPELINE_AUDIT_HANDOFF_2026-09-29" / "COLLABORATION.md"

# render mode -> sub-directory of a release (same layout as release_v6.common.MODES)
MODES: dict[str, str] = {"retrieved": ".", "gt_retrieved": "gt_retrieved", "gt_retrieved_drop": "gt_retrieved_drop"}
SPLITS: tuple[str, ...] = ("train", "dev", "holdout")
MEDIA_DIR = "media"  # inside each new release dir; holds hard links of images whose source root differs


@dataclass(frozen=True)
class Pool:
    """One source release. `tag` is used in rewritten media paths and must never contain 'holdout'
    (trainer_v2.paths_v2.refuse_holdout_path rejects any train/dev path containing that word)."""
    name: str
    release: Path
    tag: str
    splits: tuple[str, ...]


POOLS: tuple[Pool, ...] = (
    Pool("RELEASE_TRAIN_DEV_v6d", SRC_TRAIN_DEV, "pool_t6d", ("train", "dev")),
    Pool("RELEASE_HOLDOUT_v2", SRC_HOLDOUT, "pool_s2", ("holdout",)),
)


@dataclass(frozen=True)
class LeakageConfig:
    phash_max_hamming: int = 8          # near-duplicate query images: 64-bit DCT pHash Hamming distance <= this
    phash_size: int = 32                # pHash: grayscale resize side before the 2-D DCT
    phash_lowfreq: int = 8              # pHash: top-left low-frequency block side (8x8 = 64 bits)
    # row fields that link rows into one component (None values never link). Persons own the concept store,
    # so person_id also covers "store / household"; the release rows carry no separate store / household /
    # query-template ids (styled generic query texts are shared across splits by design and are NOT linked).
    link_fields: tuple[str, ...] = ("family_id", "person_id", "pair_id", "evidence_pair_id")


@dataclass(frozen=True)
class FeatureConfig:
    acute_priority: str = "acute_incident_or_injury_escalation"
    escalation_action: str = "safety_escalation"
    unsafe_label: str = "unsafe"
    strict_stratum: str = "strict"
    person_types: frozenset[str] = frozenset({"person"})
    pet_types: frozenset[str] = frozenset({"cat", "dog", "parrot"})
    no_style: str = "none"
    absent_type_suffix: str = "_absent"   # row types whose target is removed by design
    present_condition: str = "gt_present"
    regen_unit_prefix: str = "regen"    # evaluator.unit of regenerated v6d units (generation provenance)


@dataclass(frozen=True)
class OptimizerConfig:
    seed: int = 20261004
    target_rows: tuple[tuple[str, int], ...] = (("train", 1000), ("dev", 130), ("holdout", 1000))
    split_weights: tuple[tuple[str, float], ...] = (("train", 1.0), ("dev", 0.5), ("holdout", 1.0))
    size_weight: float = 4.0            # penalty weight on squared relative size error
    iterations: int = 400_000
    t_start: float = 0.1                # annealing temperature (loss units), geometric cooling to t_end
    t_end: float = 1e-7
    swap_prob: float = 0.5
    min_count_weight: float = 4.0       # penalty on a split's shortfall below a ResplitConfig.min_counts minimum              # else: move one component to another split

    def targets(self) -> dict[str, int]:
        return dict(self.target_rows)

    def weights(self) -> dict[str, float]:
        return dict(self.split_weights)


@dataclass(frozen=True)
class Tolerance:
    """Acceptance bars checked by the build and the tests (absolute share deviation from the pooled share)."""
    max_dev_large: float = 0.03         # train / holdout, every feature
    max_dev_dev: float = 0.08           # dev (about 130 rows of 17-row components: coarser)
    size_rel: float = 0.05              # |rows - target| / target


@dataclass(frozen=True)
class HarmonizeConfig:
    query_key: str = "resplit_v1"        # H4: s4_g0query.assign key (v5 used one key per pool)
    reassign_queries: bool = True        # H4
    family_rule: bool = True             # H3 (v6c acute family rule)
    image_acute_rule: bool = False       # H3b: opt-in extension to safe rows of OTHER families on the same image
    plan_roots: tuple[tuple[str, str], ...] = ((str(PROD_ROOT), "prod"), (str(SEALED_ROOT), "holdout"))


@dataclass(frozen=True)
class ResplitConfig:
    harmonize: HarmonizeConfig = field(default_factory=HarmonizeConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tolerance: Tolerance = field(default_factory=Tolerance)
    pools: tuple[Pool, ...] = POOLS
    train_dev_name: str = OUT_TRAIN_DEV.name
    holdout_name: str = OUT_HOLDOUT.name
    mode: str = MODE_MIXED              # mixed: in-distribution, scene as a stratum; scene_disjoint: holdout scenes
    holdout_scenes: tuple[str, ...] = ("bicycle", "bonsai", "london", "nyc", "room")  # audit's best alternative
    keys_path: Path | None = AUDIT_KEYS  # None: no scene stratum / view-group unit (synthetic tests)
    overlay_new_fields: tuple[str, ...] = ("risk_not_scored",)
    # minimum per-split counts of a feature's numerator; panic E rows = G1_panic_memory with safety_escalation (A9)
    min_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
        ("escalation_given_type:G1_panic_memory", (("holdout", 15), ("dev", 3))),)
    extra_origin: str = "harmonize_v1_panic"   # pool name of --extra-rows rows (provenance / origin feature)
    extra_tag: str = "pool_x1"                 # media tag of --extra-rows rows
    extra_allowed_new_keys: tuple[str, ...] = ("origin", "risk_not_scored")   # overlay may ADD these top-level row fields
    seen_source_splits: tuple[str, ...] = ("train",)   # seen/unseen identity flags of holdout rows come from these
    hash_media: bool = True             # sha256 of every linked media file in MEDIA_INDEX.jsonl
    io_threads: int = 16


__all__ = ["ResplitConfig", "HarmonizeConfig", "LeakageConfig", "FeatureConfig", "OptimizerConfig", "Tolerance", "Pool", "POOLS",
           "MODES", "SPLITS", "AUDIT_KEYS", "MODE_MIXED", "MODE_SCENE", "MEDIA_DIR", "PROD_ROOT", "SEALED_ROOT", "OUT_TRAIN_DEV", "OUT_HOLDOUT", "STAGING",
           "COLLAB_LOG", "HERE"]
