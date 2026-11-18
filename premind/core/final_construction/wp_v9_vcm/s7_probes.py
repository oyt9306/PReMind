"""Stage 7 (CPU): canary probes (spec R2 PASS checks, adapted to retrieval-aware contexts and K=8).

1 composition probe     same-type count, entry kinds, memory-text lengths per display slot -> GT-present/absent,
                        leave-family-out logistic regression, balanced accuracy <= 0.60.
2 text-only identification  lexical baseline here (query vs the target + 2 displayed look-alikes' texts, chance 1/3;
                        the model version runs in s7_zs) <= 1/3 + 0.10; swap test (I4 swap proxy) = 100%.
3 retriever label rule   I6 holds for both arms (100%); condition agreement main vs legacy CLIP is reported.
4 entry-image role probe source, resolution, aspect, background, generation-input flag, target-elsewhere count ->
                        "is target" (leave-family-out), BA <= 0.60.
5 identification band    filled from s7_zs (0.40-0.90 per kind, strict stratum only).
6 memory/length budget   prompt tokens (text + 9 images at 448^2 + query image) + 512 completion vs the budget; the
                        real A40 training step needs trainer_v1 (pending).
+ target recall@K per kind, GT-absent / natural-miss rates, query-only pair probe (reported, owner change 3),
  crossed safety coverage (unsafe G1-visual / GT-absent / G0 rows), 1- vs 2-concept strata.
"""
from __future__ import annotations

import math
from collections import Counter

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s4_checks

# owner feedback 4 (+ change 5): at least one unsafe row in each of these row types
SAFETY_REQUIRED = ("G1_visual", "G1G2_visual", "GT_absent", "G0_closed")   # general-rule control removed (2026-10-03)
IMAGE_TOKENS_448 = (cm.ENTRY_SIZE // 32) ** 2          # Qwen3-VL: patch 16 x merge 2 -> 196 tokens per entry


def grouped_ba(X: list[list[float]], y: list[int], groups: list[str]) -> float | None:
    """Leave-family-out (<= 5 folds) logistic-regression balanced accuracy; None if a class or groups are missing."""
    if len(set(y)) < 2 or len(set(groups)) < 2:
        return None
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xa, ya, ga = np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)
    pred = np.zeros_like(ya)
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(Xa, ya, ga):
        if len(set(ya[tr])) < 2:
            pred[te] = ya[tr][0]
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xa[tr], ya[tr])
        pred[te] = m.predict(Xa[te])
    return round(float(balanced_accuracy_score(ya, pred)), 4)


def _personal(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["arms"][cm.MAIN_RETRIEVER]["expected_memory_behavior"] in ("use", "clarify_or_abstain")]


def entry_texts(a: dict) -> list[str]:
    """Memory texts in display order from the rendered Stage-2 blocks (header, image, text) x K + final."""
    blocks = a["render"]["memory_call_stage2_call"][:-1]
    return [blocks[i + 2]["text"] for i in range(0, len(blocks), 3)]


def composition_probe(rows: list[dict], plan: dict) -> dict:
    X, y, g = [], [], []
    for r in _personal(rows):
        fam, a = plan[r["family_id"]], r["arms"][cm.MAIN_RETRIEVER]
        keys = a["display_keys"]
        ttypes = {fam["concepts"][t]["fine_type"] for t in fam["targets"]}
        kinds = Counter(fam["concepts"][k]["kind"] for k in keys)
        lens = [len(t.split()) for t in entry_texts(a)]
        X.append([sum(fam["concepts"][k]["fine_type"] in ttypes for k in keys), kinds["person"], kinds["pet"],
                  kinds["object"], *lens, sum(lens) / max(1, len(lens)), max(lens or [0])])
        y.append(int(a["condition"] == "gt_present"))
        g.append(r["family_id"])
    ba = grouped_ba(X, y, g)
    return {"n": len(y), "ba": ba, "pass": ba is not None and ba <= cm.THRESHOLDS["composition_probe_ba_max"]}


def id_options(r: dict, fam: dict, t: str) -> list[int] | None:
    """Display indices (1-based) of the target and its two best-ranked displayed look-alikes, or None."""
    a = r["arms"][cm.MAIN_RETRIEVER]
    keys, rank = a["display_keys"], {int(j): v for j, v in a["display_to_rank"].items()}
    look = sorted((j + 1 for j, k in enumerate(keys) if fam["concepts"][k]["role"] in ("negative", "replacement")
                   and fam["concepts"][k]["of_target"] == t), key=lambda j: rank[j])[:2]
    return [keys.index(t) + 1, *look] if t in keys and len(look) == 2 else None


def text_only_lexical(rows: list[dict], plan: dict) -> dict:
    hits, n = 0, 0
    for r in rows:
        a = r["arms"][cm.MAIN_RETRIEVER]
        if a["expected_memory_behavior"] != "use" or not r["row_type"].startswith("G1_"):
            continue
        fam, texts = plan[r["family_id"]], entry_texts(a)
        for t in fam["targets"]:
            opts = id_options(r, fam, t)
            if not opts:
                continue
            q = s4_checks.content(r["query"])
            score = {j: len(q & s4_checks.content(texts[j - 1])) for j in opts}
            tie = sorted(j for j, v in score.items() if v == max(score.values()))
            hits += tie[cm.rng(f"tie:{r['row_id']}:{t}").randrange(len(tie))] == opts[0]
            n += 1
    acc = round(hits / n, 4) if n else None
    return {"n": n, "accuracy": acc, "chance": round(1 / 3, 4),
            "pass": acc is not None and acc <= cm.THRESHOLDS["text_only_identification_max"]}


def role_probe(rows: list[dict], plan: dict) -> dict:
    srcs = sorted({c["source_family"] for f in plan.values() for c in f["concepts"].values()})
    X, y, g, seen = [], [], [], set()
    for r in _personal(rows):
        a = r["arms"][cm.MAIN_RETRIEVER]
        if a["condition"] != "gt_present" or (r["family_id"], "x") in seen:
            continue
        seen.add((r["family_id"], "x"))
        fam = plan[r["family_id"]]
        for k in a["display_keys"]:
            c = fam["concepts"][k]
            X.append([*(float(c["source_family"] == s) for s in srcs), math.log(c["orig_w"]), math.log(c["orig_h"]),
                      c["orig_w"] / c["orig_h"], c["border_mean"], c["border_std"],
                      float(c["gallery_is_generation_input"]), c["target_elsewhere_count"]])
            y.append(int(k in fam["targets"]))
            g.append(r["family_id"])
    ba = grouped_ba(X, y, g)
    return {"n_entries": len(y), "ba": ba, "pass": ba is not None and ba <= cm.THRESHOLDS["role_probe_ba_max"]}


def length_budget(rows: list[dict]) -> dict:
    from PIL import Image
    tok = _tokenizer()
    worst, totals = 0, []
    for r in rows:
        a = r["arms"][cm.MAIN_RETRIEVER]["render"]
        text = " ".join(c["text"] for m in a["passive"] for c in m["content"] if c["type"] == "text")
        n_text = len(tok(text)) if tok else int(len(text.split()) * 1.35)
        with Image.open(r["image_path"]) as im:
            w, h = im.size
        q_tokens = round(h / 32) * round(w / 32)
        total = n_text + (a["n_images"] - 1) * IMAGE_TOKENS_448 + q_tokens + cm.THRESHOLDS["degeneration_gate_tokens"]
        totals.append(total)
        worst = max(worst, total)
    budget = cm.THRESHOLDS["max_model_len_budget"]
    return {"images_per_prompt": cm.K + 1, "entry_image_tokens": IMAGE_TOKENS_448, "tokenizer": bool(tok),
            "max_total_tokens": worst, "median_total_tokens": sorted(totals)[len(totals) // 2] if totals else None,
            "budget": budget, "pass": worst <= budget,
            "training_step_fit": "pending: one real 9-image training step on an A40 via trainer_v1 (not run here)"}


def _tokenizer():  # noqa: ANN202
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(str(cm.QWEN8B), local_files_only=True)
        return lambda s: t(s)["input_ids"]
    except Exception:  # noqa: BLE001 - optional dependency; fall back to a word-count estimate
        return None


def pair_probe(rows: list[dict]) -> dict:
    """Query-only memory-call pair probe (reported only, owner change 3): bag-of-words -> needs memory."""
    data = [r for r in rows if r["memory_call_pair_id"]]
    if len({r["family_id"] for r in data}) < 2:
        return {"n": len(data), "pair_accuracy": None}
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    y = np.asarray([int(r["grounding_requirement"] in ("G1", "G1+G2")) for r in data])
    X = CountVectorizer(min_df=1).fit_transform([r["query"] for r in data])
    gr = [r["family_id"] for r in data]
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=min(5, len(set(gr)))).split(X, y, gr):
        pred[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict(X[te]) if len(set(y[tr])) > 1 else y[tr][0]
    by = {}
    for r, ok_ in zip(data, pred == y):
        by.setdefault(r["memory_call_pair_id"], []).append(bool(ok_))
    full = [v for v in by.values() if len(v) == 2]
    acc = round(sum(all(v) for v in full) / len(full), 4) if full else None
    return {"n_pairs": len(full), "pair_accuracy": acc, "reference": cm.THRESHOLDS["query_only_pair_accuracy_reported_only"],
            "gate": False}


def run_probes(cfg: cm.Cfg) -> dict:
    out = cfg.p("s7", "probes.json")
    if out.exists():
        return cm.read(out)
    plan_doc = cm.load_plan(cfg)
    plan = {f["family_id"]: f for f in plan_doc["families"]}
    rendered = cm.read(cfg.p("s6", "rendered_rows.json"))["rows"]
    rows = [r for r in rendered if r["accepted"]] or rendered
    all_rows = cm.read(cm.final_rows_path(cfg))["rows"]
    swap = [r for r in all_rows if r["deterministic_pass"] and r["expected_memory_behavior"] == "use"]
    i6 = {arm: round(sum(r["checks"]["I6_labels_by_rule"]["pass"] for r in all_rows) / max(1, len(all_rows)), 4)
          for arm in cm.RETRIEVERS}
    personal = [r for r in all_rows if r["expected_memory_behavior"] != "ignore"]
    agree = {arm: round(sum(r["arms"][cm.MAIN_RETRIEVER].get("condition") == r["arms"][arm].get("condition")
                            for r in personal) / max(1, len(personal)), 4)
             for arm in cm.RETRIEVERS if arm != cm.MAIN_RETRIEVER}
    extra = {r: cm.read(cfg.p("s1", f"retrieval_{r}.json"))["report"] for r in cm.EXTRA_RETRIEVERS
             if cfg.p("s1", f"retrieval_{r}.json").exists()}
    unsafe = Counter(r["row_type"] for r in rows if r["risk_label"] == "unsafe")
    res = {"composition_probe": composition_probe(rows, plan),
           "text_only_identification_lexical": text_only_lexical(rows, plan),
           "swap_test": {"n": len(swap), "rate": round(sum(r["checks"].get("I4_swap_proxy", {"pass": False})["pass"]
                                                           for r in swap) / len(swap), 4) if swap else None},
           "retriever_label_rule": {"i6_pass_rate": i6, "pass": min(i6.values()) >= 1.0,
                                    "condition_agreement_main_vs_other_arms": agree},
           "role_probe": role_probe(rows, plan), "length_budget": length_budget(rows),
           "retrieval": {**plan_doc["report"]["retrieval_stats_all_candidates"], "extra_arms_canary": extra,
                         "main_arm": cm.MAIN_RETRIEVER,
                         "canary_gt_absent_rate": round(sum(r["condition"] == "gt_absent" for r in all_rows
                                                            if r["expected_memory_behavior"] != "ignore")
                                                        / max(1, sum(r["expected_memory_behavior"] != "ignore" for r in all_rows)), 4),
                         "canary_natural_miss_rate": cm.read(cm.final_rows_path(cfg))["report"]["natural_miss_rate"]},
           "query_only_pair_probe": pair_probe(rows),
           "safety_coverage": {"unsafe_by_row_type": dict(unsafe), "required": list(SAFETY_REQUIRED),
                               "pass": all(any(k == p or (p == "GT_absent" and k.endswith("_absent")) for k in unsafe)
                                           for p in SAFETY_REQUIRED)},
           "strata": dict(Counter(f"{r['n_targets']}_concept/{r['stratum']}" for r in rows)),
           "n_rows_probed": len(rows), "accepted_only": any(r["accepted"] for r in rendered)}
    res["swap_test"]["pass"] = res["swap_test"]["rate"] == cm.THRESHOLDS["swap_test_min"]
    cm.write_once(out, res)
    return res
