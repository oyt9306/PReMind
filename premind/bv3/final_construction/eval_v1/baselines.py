"""External baseline VLMs (EVAL ONLY): registry, local-snapshot resolution and loader.

The baselines are evaluated with the same items, prompts, two-stage memory_call protocol, reward_v5 scoring and
metrics as the Qwen3.5 models (eval_v1/evaluate.py); only the model / processor and the backend's chat-template
kwargs differ (baseline_backend.py). No LoRA, no training, bf16, eval mode. Everything resolves from the local HF
cache (HF_HUB_OFFLINE=1): an unknown key, a missing snapshot, an unpinned revision that does not match the cached
``refs/main`` or a missing weight shard fails closed. A revision of None (RePIC, downloaded separately) is resolved
from the local ``refs/main`` at load time and recorded in the eval meta.

Decoding parity: the Qwen3.5 checkpoints ship no generation_config.json, so their ``generate`` defaults are
``GenerationConfig.from_model_config``. The baselines' shipped generation configs carry sampling defaults
(top_k / top_p / temperature) and RePIC a repetition_penalty of 1.05 that HFBackend does not override, so the loader
replaces them by the same model-config default (the dropped values are recorded in the meta).

Non-Qwen families (InternVL3.5, Gemma 4, GLM-4.6V, Phi-4-MM; ``Family.backend == "generic"``) run through
eval_v1/generic_backend.py: same conversation text through each processor's own chat template, EOS = the shipped
generation_config ``eos_token_id`` plus the template's turn-end token (``Family.turn_end``). A ``Baseline.blocked``
reason refuses the key at resolve time (fail closed, e.g. remote code that cannot run under the installed stack).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


QWEN_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
GEMMA_FILES = ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json")
PHI4_BLOCKED = ("remote modeling_phi4mm.py (written for transformers 4.48) cannot be imported in the qwen35_trl env: "
                "it requires the package 'backoff', which is not installed (no new dependencies without owner OK), "
                "and its cache / attention code is unvalidated under transformers 5.12; processor + config load on CPU")


@dataclass(frozen=True)
class Family:
    model_type: str          # config.json model_type
    model_class: str         # transformers class used to load the weights
    processor_class: str     # expected AutoProcessor class
    backend: str = "qwen"    # "qwen": baseline_backend.BaselineBackend; "generic": generic_backend.GenericBackend
    turn_end: str = "<|im_end|>"                 # token that closes an assistant turn in the chat template
    template_kwargs: tuple[tuple[str, Any], ...] = ()   # chat-template kwargs (thinking switch OFF where present)
    files: tuple[str, ...] = QWEN_FILES          # non-weight files that must be in the snapshot
    remote_code: bool = False                    # trust_remote_code for processor / model


FAMILIES = {
    "qwen3_vl": Family("qwen3_vl", "Qwen3VLForConditionalGeneration", "Qwen3VLProcessor"),
    "qwen2_5_vl": Family("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLProcessor"),
    "internvl": Family("internvl", "InternVLForConditionalGeneration", "InternVLProcessor", backend="generic"),
    "gemma4": Family("gemma4", "Gemma4ForConditionalGeneration", "Gemma4Processor", backend="generic",
                     turn_end="<turn|>", template_kwargs=(("enable_thinking", False),), files=GEMMA_FILES),
    "glm4v": Family("glm4v", "Glm4vForConditionalGeneration", "Glm46VProcessor", backend="generic",
                    turn_end="<|user|>", template_kwargs=(("enable_thinking", False),)),
    "phi4mm": Family("phi4mm", "AutoModelForCausalLM", "Phi4MMProcessor", backend="generic", turn_end="<|end|>",
                     remote_code=True),
}


@dataclass(frozen=True)
class Baseline:
    key: str
    model_id: str
    revision: Optional[str]  # None: resolve from the local refs/main at load time (recorded in meta)
    family: str
    dtype: str = "bfloat16"
    note: str = ""
    blocked: str = ""        # non-empty: refuse at resolve time with this reason


BASELINES = {b.key: b for b in (
    Baseline("qwen3vl_4b", "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", "qwen3_vl"),
    Baseline("qwen3vl_8b", "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "qwen3_vl"),
    Baseline("covip_8b", "Yeongtak/CoViP-Qwen3-VL-8B-GSPO", "2283a3061626481e0d62ac48cb9f2bc027b3ea83", "qwen3_vl",
             note="personalized captioning RL model (Qwen3-VL-8B base)"),
    Baseline("qwen2_5vl_7b", "Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", "qwen2_5_vl",
             note="RePIC-7B base model (owner 2026-10-07: attribute RePIC's call prior)"),
    Baseline("repic_7b", "Yeongtak/RePIC_Qwen2.5VL_7B", None, "qwen2_5_vl",
             note="personalized captioning RL model (Qwen2.5-VL-7B base); revision resolved from local refs/main"),
    Baseline("internvl3_5_4b", "OpenGVLab/InternVL3_5-4B-HF", "6bd4487402110ef9889ba50eb7aefeb302526fed", "internvl"),
    Baseline("internvl3_5_8b", "OpenGVLab/InternVL3_5-8B-HF", "741a7d03020411e666c6109218ab71e08151ef86", "internvl"),
    Baseline("gemma4_e4b", "google/gemma-4-E4B-it", None, "gemma4", note="revision resolved from local refs/main"),
    Baseline("glm4_6v_flash", "zai-org/GLM-4.6V-Flash", None, "glm4v",
             note="9B; thinking off (template switch); revision resolved from local refs/main"),
    Baseline("phi4_mm", "microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81", "phi4mm",
             blocked=PHI4_BLOCKED),
)}


class BaselineUnavailableError(RuntimeError):
    """Unknown baseline key or incomplete local snapshot (fail closed; nothing is downloaded)."""


def hub_cache() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def get(key: str) -> Baseline:
    if key not in BASELINES:
        raise BaselineUnavailableError(f"unknown baseline {key!r}; known {sorted(BASELINES)}")
    return BASELINES[key]


def _repo_dir(model_id: str, cache: Path) -> Path:
    return cache / ("models--" + model_id.replace("/", "--"))


def resolve_revision(spec: Baseline, cache: Optional[Path] = None) -> str:
    """Pinned revision, or the local refs/main for an unpinned one; the snapshot directory must exist."""
    repo = _repo_dir(spec.model_id, cache or hub_cache())
    rev = spec.revision
    if rev is None:
        ref = repo / "refs" / "main"
        if not ref.is_file():
            raise BaselineUnavailableError(f"{spec.key}: no local refs/main under {repo} (not downloaded)")
        rev = ref.read_text().strip()
    if not (repo / "snapshots" / rev).is_dir():
        raise BaselineUnavailableError(f"{spec.key}: snapshot {rev} not in the local cache ({repo})")
    return rev


def snapshot_files(spec: Baseline, rev: str, cache: Optional[Path] = None) -> Path:
    """Snapshot dir after checking config / processor / tokenizer files and EVERY weight shard are present."""
    snap = _repo_dir(spec.model_id, cache or hub_cache()) / "snapshots" / rev
    need = list(FAMILIES[spec.family].files)
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        need += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        need.append("model.safetensors")
    missing = [n for n in need if not (snap / n).is_file()]     # dangling symlinks (partial download) count
    if missing:
        raise BaselineUnavailableError(f"{spec.key}@{rev}: missing local files {missing} (download incomplete?)")
    model_type = json.loads((snap / "config.json").read_text()).get("model_type")
    if model_type != FAMILIES[spec.family].model_type:
        raise BaselineUnavailableError(f"{spec.key}: config model_type {model_type!r} != family {spec.family!r}")
    return snap


def resolve(key: str, cache: Optional[Path] = None) -> dict:
    """Eval-meta record of a baseline (key, HF id, resolved revision, family, dtype, classes, snapshot)."""
    spec = get(key)
    if spec.blocked:
        raise BaselineUnavailableError(f"{key}: BLOCKED - {spec.blocked}")
    rev = resolve_revision(spec, cache)
    snap = snapshot_files(spec, rev, cache)
    fam = FAMILIES[spec.family]
    meta = {**asdict(spec), "revision": rev, "revision_pinned": spec.revision is not None,
            "model_class": fam.model_class, "processor_class": fam.processor_class, "snapshot": str(snap)}
    if fam.backend == "generic":     # EOS source for generic_backend (the shipped file, not the dropped defaults)
        gc = snap / "generation_config.json"
        meta["shipped_generation_config"] = json.loads(gc.read_text()) if gc.is_file() else None
        meta["backend"], meta["turn_end"] = fam.backend, fam.turn_end
        meta["template_kwargs"] = dict(fam.template_kwargs)
    return meta


def load_processor(meta: dict) -> Any:
    """CPU-safe: the processor of a resolved baseline, checked against the family's processor class."""
    from transformers import AutoProcessor
    remote = FAMILIES[meta["family"]].remote_code if "family" in meta else False
    proc = AutoProcessor.from_pretrained(meta["model_id"], revision=meta["revision"], trust_remote_code=remote)
    if type(proc).__name__ != meta["processor_class"]:
        raise BaselineUnavailableError(f"{meta['key']}: processor {type(proc).__name__} != {meta['processor_class']}")
    return proc


def load_baseline(key: str) -> tuple[Any, Any, dict]:  # pragma: no cover - GPU (scheduler job only)
    """(model, processor, meta): family model class, bf16, cuda, eval, no grad, no LoRA, neutral generation config."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("REFUSED: baselines load only with HF_HUB_OFFLINE=1")
    import torch
    import transformers
    from transformers import GenerationConfig
    meta = resolve(key)
    proc = load_processor(meta)
    cls = getattr(transformers, meta["model_class"])
    if FAMILIES[meta["family"]].remote_code:   # unreachable while blocked; never auto-trust remote modeling code
        raise BaselineUnavailableError(f"{key}: remote modeling code is not loaded by this evaluator")
    model = cls.from_pretrained(meta["model_id"], revision=meta["revision"],
                                dtype=getattr(torch, meta["dtype"])).to("cuda")
    shipped = model.generation_config.to_diff_dict()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    meta = {**meta, "shipped_generation_config_dropped": {k: v for k, v in shipped.items()
                                                           if k in ("do_sample", "temperature", "top_k", "top_p",
                                                                    "repetition_penalty")},
            "transformers_version": transformers.__version__}
    logger.info("baseline %s: %s@%s (%s)", key, meta["model_id"], meta["revision"], meta["model_class"])
    return model.eval().requires_grad_(False), proc, meta
