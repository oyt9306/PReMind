"""Stage 1 (CPU): retrieval-aware person stores, offline retrieval contexts for both retrievers, canary selection.

Store (owner correction 2026-10-02): target(s) + 3-4 same-fine-type hard negatives per target, scored by
SIM_MIX*max-over-query-views sim + (1-SIM_MIX)*sim(target reference) with STORE_RETRIEVER (WeMM-Embedding-2B, frozen) and sampled
with a seed from the top-k (identities that are targets elsewhere weighted up; persons: same source + gender + age
group) + other concepts of different types, to a fixed store size. Fewer than 2 same-type candidates -> relaxed tier
(separate stratum, M2). The GT-absent store removes the target(s) and adds the next unsampled same-type hard negative
per target (store size and same-type count constant; B1).
Context = offline top-K over the store (FINAL K RULE: K=8 for every image; whole image + generic detector crops,
merged round-robin by rank, deduped), for the main and the legacy retriever; labels come later by rule.
"""
from __future__ import annotations

from collections import Counter

from final_construction.wp_v9_vcm import common as cm
from final_construction.wp_v9_vcm import s0_pool
from final_construction.wp_v9_vcm import taxonomy as tx


class World:
    def __init__(self, cfg: cm.Cfg) -> None:
        pool = cm.read(cfg.p("s0", "pool.json"))
        self.ids, self.images, self.families = pool["identities"], pool["images"], pool["families"]
        self.desc = cm.read(cfg.p("s0", "descriptions.json"))["descriptions"]
        ft = cm.read(cfg.p("s0", "fine_types.json"))                # stage 0b (finetype.py), never the raw Qwen enum
        if ft.get("preview"):
            raise RuntimeError("plan needs the final s0/fine_types.json, not the preview")
        self.final_type = {i: r["fine_type"] for i, r in ft["fine_types"].items()}
        self.emb = {r: s0_pool.load_embeddings(cfg, r) for r in cm.PLAN_RETRIEVERS}
        self.target_count = Counter(t for im in self.images.values() for t in im["targets"])

    def ft(self, iid: str) -> str:
        return self.final_type[iid]

    def person_group(self, iid: str) -> tuple[str, str]:
        return self.desc[iid]["gender"], self.desc[iid]["age_group"]

    def tiers(self, t: str) -> list:
        if self.ids[t]["kind"] == "person":
            src, (g, a) = self.ids[t]["source_family"], self.person_group(t)
            return [lambda c: self.ids[c]["source_family"] == src and self.person_group(c) == (g, a),
                    lambda c: self.person_group(c) == (g, a), lambda c: self.person_group(c)[0] == g, lambda c: True]
        ft = self.ft(t)
        return [lambda c: self.ft(c) == ft, lambda c: True]

    def sim(self, retr: str, qkey: str, iid: str) -> float:
        g, q = self.emb[retr]
        return float(q[qkey] @ g[iid])


def views(w: World, image_key: str, retr: str = cm.STORE_RETRIEVER) -> list[str]:
    """Query views of an image: the whole image, then its generic detector crops (FINAL K RULE; no use of the
    governed-concept count)."""
    q = w.emb[retr][1]
    crops = sorted((k for k in q if k.startswith(image_key + "#crop:")), key=lambda k: int(k.rsplit(":", 1)[1]))
    return [image_key] + (crops if cm.USE_CROPS else [])


def hard_negatives(w: World, image_key: str, t: str, exclude: set[str], n: int, seed_key: str | None = None) -> dict:
    kind = w.ids[t]["kind"]
    pool = sorted(c for c in w.ids if c not in exclude and w.ids[c]["kind"] == kind)
    tiers = w.tiers(t)
    n_same = sum(tiers[0](c) for c in pool)
    level, cands = next(((i, [c for c in pool if f(c)]) for i, f in enumerate(tiers) if sum(f(c) for c in pool) >= 2),
                        (len(tiers) - 1, pool))
    vs = views(w, image_key)
    g = w.emb[cm.STORE_RETRIEVER][0]
    score = {c: cm.SIM_MIX * max(w.sim(cm.STORE_RETRIEVER, v, c) for v in vs) + (1 - cm.SIM_MIX) * float(g[t] @ g[c])
             for c in cands}
    ranked = sorted(cands, key=lambda c: (-score[c], c))
    top = ranked[:cm.TOPK_CAND]
    weights = [cm.TARGET_ELSEWHERE_WEIGHT if w.target_count[c] > 0 else 1.0 for c in top]
    chosen = cm.weighted_sample(top, weights, min(n, len(top)), f"neg:{seed_key or image_key}:{t}")
    chosen = sorted(chosen, key=lambda c: (-score[c], c))
    replacement = next((c for c in ranked if c not in chosen), None)
    return {"negatives": chosen, "replacement": replacement, "relaxed_level": level, "relaxed": level > 0,
            "n_same_type_available": n_same, "scores": {c: round(score[c], 5) for c in top}}


def build_store(w: World, image_key: str, seed_key: str | None = None) -> dict:
    """Store of one family; seeded by image (canary, unchanged) or by family (production: families of one image get
    different look-alike/other draws)."""
    targets = w.images[image_key]["targets"]
    n_t = len(targets)
    exclude, info = set(targets), {}
    for t in targets:
        info[t] = hard_negatives(w, image_key, t, exclude, cm.N_NEG[n_t], seed_key)
        exclude |= set(info[t]["negatives"]) | ({info[t]["replacement"]} - {None})
    banned_types = {w.ft(c) for c in exclude}
    others_pool = sorted(c for c in w.ids if c not in exclude and w.ids[c]["kind"] != "person"
                         and w.ft(c) not in banned_types)
    n_other = cm.STORE_SIZE[n_t] - n_t - sum(len(i["negatives"]) for i in info.values())
    others = sorted(cm.weighted_sample(others_pool, [cm.TARGET_ELSEWHERE_WEIGHT if w.target_count[c] else 1.0
                                                     for c in others_pool], n_other, f"others:{seed_key or image_key}"))
    store = sorted(targets + [c for t in targets for c in info[t]["negatives"]] + others)
    absent = sorted([c for c in store if c not in targets] + [info[t]["replacement"] for t in targets
                                                              if info[t]["replacement"]])
    return {"targets": targets, "target_info": info, "others": others, "store": store, "absent_store": absent}


def rank(w: World, retr: str, qkey: str, store: list[str]) -> list[tuple[str, float]]:
    return sorted(((c, w.sim(retr, qkey, c)) for c in store), key=lambda x: (-x[1], x[0]))


def retrieve(w: World, retr: str, image_key: str, store: list[str]) -> dict:
    """Offline top-K (stored per row), FINAL K RULE: K fixed for every image; rankings of every query view (whole
    image + detector crops) merged round-robin by rank, deduped, first K. Entry sources never name an identity."""
    lists = [(v, rank(w, retr, v, store)) for v in views(w, image_key, retr)]
    merged: list[dict] = []
    for r in range(len(store)):
        for n, (v, lst) in enumerate(lists):
            if r < len(lst) and lst[r][0] not in {e["iid"] for e in merged}:
                merged.append({"iid": lst[r][0], "score": round(lst[r][1], 6),
                               "source": "whole" if n == 0 else f"crop:{n}", "source_rank": r + 1})
    whole = lists[0][1]
    return {"retriever": retr, **cm.RETRIEVERS[retr], "k": cm.K, "query_views": len(lists),
            "entries": [dict(e) for e in merged[:cm.K]], "merged_ranking": merged,
            "whole_ranking": [{"iid": c, "score": round(s, 6)} for c, s in whole]}


def condition(context_iids: list[str], targets: list[str]) -> str:
    """Label rule (I6): all targets in the stored context -> gt_present; none -> gt_absent; otherwise partial."""
    hit = sum(t in context_iids for t in targets)
    return "gt_present" if hit == len(targets) else "gt_absent" if hit == 0 else "partial"


def candidate(w: World, image_key: str, seed_key: str | None = None) -> dict:
    s = build_store(w, image_key, seed_key)
    s["retrieval"] = {r: {"full": retrieve(w, r, image_key, s["store"]),
                          "absent": retrieve(w, r, image_key, s["absent_store"])} for r in cm.PLAN_RETRIEVERS}
    s["natural_condition"] = {r: condition([e["iid"] for e in s["retrieval"][r]["full"]["entries"]], s["targets"])
                              for r in cm.PLAN_RETRIEVERS}
    return s


def select_canary(w: World, cands: dict[str, dict]) -> list[str]:
    order = cm.seeded_order(sorted(cands), "canary_select")
    picked: list[str] = []

    def kinds(k: str) -> set[str]:
        return {w.ids[t]["kind"] for t in cands[k]["targets"]}
    quotas = [(lambda k: len(cands[k]["targets"]) == 2, cm.CANARY_MIN_TWO_TARGET),
              (lambda k: any(i["relaxed"] for i in cands[k]["target_info"].values()), cm.CANARY_MIN_RELAXED)]
    quotas += [(lambda k, kd=kd: kd in kinds(k), n) for kd, n in cm.CANARY_KIND_MIN.items()]
    for pred, need in quotas:
        for k in order:
            if sum(pred(p) for p in picked) >= need or len(picked) >= cm.CANARY_N:
                break
            if k not in picked and pred(k):
                picked.append(k)
    picked += [k for k in order if k not in picked][:cm.CANARY_N - len(picked)]
    return sorted(picked)


def assign_concepts(w: World, fam: dict, s: dict) -> dict:
    """Pseudonymous keys, names (fixed lists, unique in store), one shared person relation, seeded text style."""
    fid = fam["family_id"]
    members = s["store"] + [i["replacement"] for i in s["target_info"].values() if i["replacement"]]
    role: dict[str, tuple[str, str | None]] = {c: ("other", None) for c in s["others"]}
    for t, i in s["target_info"].items():
        role[t] = ("target", None)
        role.update({c: ("negative", t) for c in i["negatives"]})
        if i["replacement"]:
            role[i["replacement"]] = ("replacement", t)
    persons = [c for c in members if w.ids[c]["kind"] == "person"]
    rels = tx.relation_options({w.person_group(c) for c in persons}) if persons else ("",)
    relation = rels[cm.rng(f"relation:{fid}").randrange(len(rels))] if persons else ""
    used, concepts = set(), {}
    for n, c in enumerate(cm.seeded_order(members, f"keys:{fid}")):
        kind = w.ids[c]["kind"]
        names = (tx.PERSON_NAMES.get(w.person_group(c)[0], tx.PERSON_NAMES["other"]) if kind == "person"
                 else tx.PET_NAMES if kind == "pet" else tx.OBJECT_NAMES)
        name = next(x for x in cm.seeded_order(list(names), f"name:{fid}:{c}") if x not in used)
        used.add(name)
        concepts[f"C{n + 1:02d}"] = {
            "iid": c, "role": role[c][0], "of_target": role[c][1],
            "kind": kind, "fine_type": w.ft(c), "name": name, "relation": relation if kind == "person" else "",
            "style": cm.TEXT_STYLES[cm.rng(f"style:{fid}:{c}").randrange(len(cm.TEXT_STYLES))],
            "description": w.desc[c]["description"], "visible_text": w.desc[c]["visible_text"],
            "gender": w.desc[c]["gender"], "age_group": w.desc[c]["age_group"],
            **{k: w.ids[c][k] for k in ("entry", "entry_sha256", "source_family", "orig_w", "orig_h", "border_mean",
                                        "border_std", "gallery_is_generation_input")},
            "target_elsewhere_count": w.target_count[c] - (1 if c in s["targets"] else 0)}
    return concepts


def risk_plan(index: int) -> dict:
    """Crossed safety axis (owner change 5): requested Option-B risk per row type, alternating over the canary so
    G1-visual, GT-absent (derived from G1/G1+G2) and G0 all get unsafe rows; the author may decline as infeasible."""
    return {"G1_visual": "unsafe" if index % 2 == 0 else "safe", "G1G2_visual": "unsafe",
            "G0_closed": "unsafe" if index % 2 == 1 else "safe", "general_rule_closed": "unsafe" if index % 3 == 0 else "safe"}


def family_plan(w: World, fam: dict, k: str, s: dict, risk: dict) -> dict:
    """One family's plan entry: pseudonymous concept keys, retrieval contexts keyed by concept, display permutation."""
    concepts = assign_concepts(w, fam, s)
    key_of = {c["iid"]: key for key, c in concepts.items()}
    for c in concepts.values():
        c["of_target"] = key_of[c["of_target"]] if c["of_target"] else None
    for r in cm.PLAN_RETRIEVERS:
        for v in ("full", "absent"):
            ctx = s["retrieval"][r][v]
            for e in ctx["entries"] + ctx["merged_ranking"] + ctx["whole_ranking"]:
                e["key"] = key_of[e.pop("iid")]
    im = w.images[k]
    return {**fam, "image_key": k, "image": im["image"], "image_sha256": im["image_sha256"],
            "targets": [key_of[t] for t in s["targets"]], "n_targets": len(s["targets"]),
            "k": cm.K, "concepts": concepts, "retrieval": s["retrieval"],
            "store": [key_of[c] for c in s["store"]], "absent_store": [key_of[c] for c in s["absent_store"]],
            "target_info": {key_of[t]: {kk: v for kk, v in i.items() if kk not in ("negatives", "replacement", "scores")}
                            | {"negatives": [key_of[c] for c in i["negatives"]],
                               "replacement": key_of.get(i["replacement"])}
                            for t, i in s["target_info"].items()},
            "display_perm": cm.seeded_order(list(range(cm.K)), f"display:{fam['family_id']}"),
            "risk_plan": risk, "natural_condition": s["natural_condition"]}


def run_plan(cfg: cm.Cfg) -> dict:
    out = cfg.p("s1", "plan.json")
    if out.exists():
        return {"status": "exists"}
    w = World(cfg)
    cands = {k: candidate(w, k) for k in sorted(w.images)}
    stats = {}
    for r in cm.PLAN_RETRIEVERS:
        per = Counter()
        for k, s in cands.items():
            for t in s["targets"]:
                kd = w.ids[t]["kind"]
                per[(kd, "n")] += 1
                per[(kd, "hit")] += t in [e["iid"] for e in s["retrieval"][r]["full"]["entries"]]
        stats[r] = {"target_recall_at_K_by_kind": {kd: round(per[(kd, "hit")] / per[(kd, "n")], 4)
                                                   for kd in tx.KINDS if per[(kd, "n")]},
                    "natural_condition": dict(Counter(s["natural_condition"][r] for s in cands.values()))}
    families = []
    for idx, k in enumerate(select_canary(w, cands)):
        fam = sorted(w.families[k], key=lambda f: f["family_id"])[0]
        families.append(family_plan(w, fam, k, cands[k], risk_plan(idx)))
    rep = {"candidates": len(cands), "selected": len(families), "retrieval_stats_all_candidates": stats,
           "selected_two_concept": sum(f["n_targets"] == 2 for f in families),
           "selected_relaxed": sum(any(i["relaxed"] for i in f["target_info"].values()) for f in families),
           "selected_kinds": dict(Counter(f["concepts"][t]["kind"] for f in families for t in f["targets"])),
           "selected_splits": dict(Counter(f["split"] for f in families))}
    cm.write_once(out, {"run_id": cfg.run_id, "params": {"K": cm.K, "STORE_SIZE": cm.STORE_SIZE, "N_NEG": cm.N_NEG,
                                                        "TOPK_CAND": cm.TOPK_CAND, "SIM_MIX": cm.SIM_MIX,
                                                        "USE_CROPS": cm.USE_CROPS, "MAX_CROPS": cm.MAX_CROPS},
                        "families": families, "report": rep})
    return rep
