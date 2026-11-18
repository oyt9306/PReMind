"""CoViP auxiliary samples (multi_vision; optional vision) for the PReMind v4 trainer, ported VERBATIM from the CoViP
release code (owner 2026-10-13: CoViP's own prompt template AND reward template).

Source of truth: ``CoViP-main/VLM-R1-Qwen3/src/open-r1-multimodal/src/open_r1`` (= ``COVIP``):
  * multi_vision prompt   ``grpo_pmllm.py`` L99-127 (byte-identical to ``vlm_modules/qwen_module.py`` L138-166 and to
                          the older ``VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_pmllm.py`` L258-286);
  * multi_vision messages ``grpo_pmllm.py`` L276-302 (``make_mul_vision_conversation``): per concept a text part
                          "===== Concept {idx} =====\\n" (idx from 0) + an image, then the prompt text, then the query
                          image; one user turn, no system prompt;
  * multi_vision reward   ``vlm_modules/qwen_module.py`` L373-416 (``yes_no_reward_multi_vis``): first ``\\boxed{...}``,
                          ``ast.literal_eval``, set F1 over indices (any parse error -> 0.0; no repetition / length
                          gate for this method). NOTE: the older VLM-R1 (Qwen2.5) copy, ``grpo_pmllm.py`` L626-643,
                          used RECALL |gt & pred| / |gt| (exploitable by listing every index); the released CoViP code
                          (the paper's) uses F1, which is what is ported here;
  * vision prompt         ``grpo_pmllm.py`` L59-61 (``return_prompt_vision``) + L38-57 (``return_random_prompt``, Faker
                          ko_KR/es_ES names); messages L228-240 (image, image, text);
  * vision reward         ``vlm_modules/qwen_module.py`` L418-436 (``yes_no_reward_vis``).
Deviations (documented in REPORT.md): Faker is not installed in the training env, so the names come from a fixed pool
generated once with the same Faker locales (``out/covip_vision_names.json``) and are drawn per item with a seeded RNG
(CoViP draws a new name at every ``__getitem__``); the chat template is Qwen3.5's (CoViP: Qwen3-VL's), thinking off as
in every PReMind run. The CoViP prompt contains a literal backspace (``Inside \\boxed{}`` in a non-raw string = "\\x08oxed"):
kept, because the port is verbatim.

Each CoViP prompt is its own GRPO group (G samples), rewarded only by its own target; advantages are normalised inside
the group (grpo_v2.grouped_advantages), so the CoViP and PReMind rewards are never combined or compared.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from trainer_v2 import reward_v5 as R5
from trainer_v2 import vcm_data as VD

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
COVIP = Path(os.environ.get("COVIP_OPEN_R1", "open_r1"))  # CoViP aux loss; unused (--aux-ratio 0)
DEFAULT_DATA = Path(os.environ.get("COVIP_AUX_DATA", "pmllm_training_data_multi_vision.json"))  # unused (--aux-ratio 0)
NAMES = HERE / "out" / "covip_vision_names.json"
METHODS = ("multi_vision", "vision")
SOURCE = {"multi_vision": "covip_multi_vision", "vision": "covip_vision"}
LEDGER_SCHEMA = "covip_aux.ledger_v1"


# ----------------------------------------------------------------------------- verbatim templates
def get_multi_vision_qa():
    prompt_multi_choice = """You are given multiple images.
Each image corresponds, in order, to a specific concept.

Your task is to identify which concept images are present in the final query image.
Include the numbers (indices) of the concept images that appear in the query image.

The query image may contain up to four concept images randomly selected from the given concepts.
However, some of the concept images are irrelevant to the query.
Do NOT include those irrelevant images in your answer.

[Answering Rules]

Observe all the concept images and the final query image carefully.

Determine which concept images appear in the query image.

For each question:

You may briefly explain your reasoning in natural language.

Then, on a separate line, output the final choice in the exact format:

Answer: \\boxed{X}

where X is something like [0, 1, 3, 5] (for example, if the query image contains the 0th, 1st, 3rd, and 5th concept images).

Inside \boxed{} you must include only the indices of the selected concept images, written as a list in square brackets, with no extra text."""  # noqa: W605,E501  (verbatim, incl. the "\b")
    return prompt_multi_choice


def naming_prompts(name: str) -> list[str]:
    """``return_random_prompt`` L40-54 templates (the name is drawn by the caller)."""
    return [
        f"Is {name} visible in this picture?",
        f"Is {name} in this image?",
        f"Do you see {name} in the photo?",
        f"Is {name} present in this photograph?",
        f"Can you identify if {name} is captured in this picture?",
        f"Is {name} depicted in this image?",
        f"Does the picture feature {name}?",
        f"Can you confirm if {name} appears in this photo?",
        f"Is {name} included in this shot?",
        f"Is {name} shown in this image?",
        f"Can you detect {name} in the photo?",
        f"Is {name} captured in this image?",
        f"Do you recognize {name} in this picture?"
    ]


def return_prompt_vision(name, naming_prompts):
    prompt = f"As an evaluation expert, your task is to verify whether the object identified as {name} in the first image is also present in the second image. {naming_prompts}"  # noqa: E501
    return prompt


def yes_no_reward_multi_vis(completions, answer, **kwargs) -> float:
    """qwen_module.py L373-416, verbatim logic (comments translated)."""
    out = completions
    gt = answer

    def to_list(x):
        if isinstance(x, (list, tuple, set)):
            return list(x)
        else:
            return [x]

    try:
        matches = re.findall(r"\\boxed\{(.*?)\}", out, flags=re.DOTALL)[0]
        pred = ast.literal_eval(matches)
        pred_list = to_list(pred)
        gt_list = to_list(gt)
        pred_set = set(pred_list)
        gt_set = set(gt_list)
    except Exception:  # noqa: BLE001 - verbatim CoViP behaviour: any parse failure scores 0
        return 0.0
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return f1


def yes_no_reward_vis(completions, answer, **kwargs) -> float:
    """qwen_module.py L418-436, verbatim."""
    gt = (answer or "").lower()
    out = (completions or "").lower()
    gt_m = re.search(r"(yes|no)", gt, flags=re.IGNORECASE)
    out_m = re.search(r"(yes|no)", out, flags=re.IGNORECASE)
    gt_ans = (gt_m.group(1).lower() if gt_m else "")
    out_ans = (out_m.group(1).lower() if out_m else "")
    return 1.0 if gt_ans and out_ans and (gt_ans == out_ans) else 0.0


def boxed_parse_ok(text: str) -> bool:
    """Logging only: the multi_vision answer has a ``\\boxed{...}`` that literal_evals."""
    try:
        ast.literal_eval(re.findall(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)[0])
        return True
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- messages / items
def multi_vision_messages(images: list[str], c_num: int) -> list[dict]:
    """``make_mul_vision_conversation`` with the image paths filled in (concepts 0..c_num-1, then the query)."""
    if len(images) != c_num + 1:
        raise ValueError(f"multi_vision sample needs c_num + 1 images, got {len(images)} for c_num {c_num}")
    content: list[dict] = []
    for idx in range(c_num):
        content.append({"type": "text", "text": f"===== Concept {idx} =====\n"})
        content.append({"type": "image", "image": images[idx]})
    content.append({"type": "text", "text": get_multi_vision_qa()})
    content.append({"type": "image", "image": images[c_num]})
    return [{"role": "user", "content": content}]


def vision_messages(images: list[str], problem: str) -> list[dict]:
    """``make_conversation_vis``: left image, right image, prompt."""
    if len(images) != 2:
        raise ValueError("vision sample needs 2 images")
    return [{"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "image", "image": images[1]},
                                         {"type": "text", "text": problem}]}]


def _name_pool() -> list[str]:
    return json.loads(NAMES.read_text(encoding="utf-8"))["names"]


def make_item(ex: dict, idx: int, seed: int = 0, names: Optional[list[str]] = None) -> VD.VcmItem:
    """One CoViP sample as a passive-arm VcmItem; ``row["aux_source"]`` routes scoring to ``score_aux``."""
    method = ex["accu_reward_method"]
    images = list(ex["images"])
    if method == "multi_vision":
        messages, answer, problem = multi_vision_messages(images, ex["c_num"]), ex["answer"], None
    elif method == "vision":
        rng = random.Random(f"covip_vision:{seed}:{idx}")
        pool = names or _name_pool()
        name = pool[rng.randrange(len(pool))]
        problem = return_prompt_vision(name, rng.choice(naming_prompts(name)))
        messages, answer = vision_messages(images, problem), ex["answer"]
    else:
        raise ValueError(f"unsupported accu_reward_method {method!r} (supported: {METHODS})")
    rid = f"covip:{method}:{idx}"
    row = {"row_id": rid, "aux_source": SOURCE[method], "aux_method": method, "aux_answer": answer,
           "aux_index": idx, "row_type": SOURCE[method], "condition": "aux", "image_path": images[-1],
           "c_num": ex.get("c_num", 2), "problem": problem}
    return VD.VcmItem(rid, "passive", messages, None, None, tuple(images), _target(rid), row, {})


def _target(rid: str) -> dict:
    return {"row_id": rid, "risk_label": None, "safety_priority": None, "expected_action": None, "withhold": False,
            "gold": [], "gold_action_set": [], "memory_required": False, "rendered_keys": []}


def is_aux(item: Any) -> bool:
    return bool(getattr(item, "row", {}).get("aux_source"))


def item_images(item: VD.VcmItem) -> dict[str, Any]:
    """path -> PIL (same loader as the release items; no sha pin: CoViP ships none)."""
    return {p: VD.load_image(p) for p in dict.fromkeys(item.image_paths)}


def load_items(path: Path = DEFAULT_DATA, methods: Iterable[str] = ("multi_vision",), *,
               exclude_images: Optional[set[str]] = None, seed: int = 0) -> tuple[list[VD.VcmItem], dict]:
    """CoViP samples of ``methods`` minus every sample with an image in ``exclude_images`` (overlap exclusion).
    Returns (items in file order, manifest)."""
    methods = tuple(methods)
    bad = set(methods) - set(METHODS)
    if bad:
        raise ValueError(f"unsupported methods {sorted(bad)}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = _name_pool() if "vision" in methods else None
    ex = exclude_images or set()
    items, n_raw, n_excl = [], 0, 0
    for idx, e in enumerate(data):
        if e.get("accu_reward_method") not in methods:
            continue
        n_raw += 1
        if any(p in ex for p in e["images"]):
            n_excl += 1
            continue
        items.append(make_item(e, idx, seed, names))
    man = {"path": str(path), "sha256": VD.P.sha256_file(Path(path)), "methods": list(methods), "n_raw": n_raw,
           "n_excluded_overlap": n_excl, "n_items": len(items), "n_exclude_images": len(ex)}
    logger.info("covip_aux: %s", json.dumps({k: v for k, v in man.items() if k != "sha256"}))
    return items, man


# ----------------------------------------------------------------------------- scoring (ledger compatible with the
# trainer_v2 consumers: records_v2.episode_record, metrics_v9.episode_record / summarize, opsd.failure_reasons)
def score_aux(item: VD.VcmItem, text: str, n_tokens: int) -> dict:
    method = item.row["aux_method"]
    if method == "multi_vision":
        value, ok = yes_no_reward_multi_vis(text, item.row["aux_answer"]), boxed_parse_ok(text)
    else:
        value = yes_no_reward_vis(text, item.row["aux_answer"])
        ok = bool(re.search(r"(yes|no)", (text or "").lower()))
    comps = {c: {"applicable": False, "value": None, "status": "not_applicable:covip_aux"} for c in R5.COMPONENTS}
    return {"ledger_schema": LEDGER_SCHEMA, "reward_version": f"covip_release:{method}", "flags": {"name": "covip_aux"},
            "row_id": item.row_id, "arm": "passive", "branch": "passive", "decision": None, "memory_required": False,
            "r": None, "n_stage2_tokens": n_tokens,
            "target": {"risk_label": None, "safety_priority": None, "expected_action": None, "withhold": False,
                       "gold": [], "gold_action_set": []},
            "cited": [], "cited_keys": [], "cited_ranks": [],
            "parse": {"format_strict": ok, "citation_valid": True, "risk_pred": None, "action_pred": None,
                      "withholds": False, "wrong_concept_fired": False, "correct_withhold": None},
            "components": comps, "applicable_components": [], "renormalized": False, "complete": True,
            "total": round(float(value), 10), "aux": {"source": item.row["aux_source"], "reward": float(value),
                                                      "parse_ok": ok}}
