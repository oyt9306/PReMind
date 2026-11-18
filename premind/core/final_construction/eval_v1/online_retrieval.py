"""Online SigLIP2 retrieval at evaluation time, K in {1, 2, 4, 8} (v4 rebuild, owner 2026-10-13; opt-in only).

Per release row (retrieved mode), at load time of the evaluation:
  * store     = the family's frozen plan store (holdout_v9_sealed/s1 or prod_v9/s1 plan.json); designed target-absent
                rows (``<type>_absent``) retrieve over the family's ``absent_store`` - a store in which the target was
                never saved (look-alikes stay). Target iids replaced by regen_v1 / regen_v2 / blite_v4 are swapped in
                (row block ``replaced_identity``: plan iid -> new id), exactly as recompute/retrieve_relabel;
  * memory    = frozen s0 SigLIP2 gallery vectors (cached) + the regen / blite sidecars' new gallery vectors;
  * query     = SigLIP2 vectors of every query view (whole image + YOLOE crops): ``frozen`` (s0 store + sidecars) or
                ``fresh`` (a per-release sidecar re-embedded from the release's query image files on GPU,
                v4_build/online_query_embed.py) overlaid on top;
  * ranking   = wp_v9_vcm.s1_plan.retrieve (rankings of all views merged round-robin by rank, deduped); top-K =
                its first K entries (the merge is prefix-stable, so top-K of the K=8 ranking).
Display order: the family's seeded ``display_perm`` restricted to the retrieved K (= release display order of those
entries, renumbered 1..K) - never the retrieval rank.
Equivalence gate (owner: "check online top-K equals the stored top-K when the store is identical"): the online top-K
keys must equal the release row's stored ``evaluator.retriever.ranked`` prefix; the context is then rendered from the
release row's stored entries (images / texts / concepts). A mismatch is reported (``online_stored_mismatch``) and
``release_data.load_release`` refuses the run: a different ranking would need entries the row does not store.
Labels at K (retrieve_relabel / s4_rows.arm_labels semantics, applied to the release row):
  * no-memory rows (G0, ``ignore``): context restricted, context_condition re-derived (I6), labels unchanged;
  * designed-absent rows: context restricted, labels unchanged (target never in the store);
  * ``use`` rows: all targets kept -> citations / support / wrong-concept patterns re-keyed; one of two kept ->
    EXCLUDE (partial_retrieval_two_concept); none kept (saved but missed) -> G1_panic_memory: EXCLUDE
    (panic_row_retrieval_miss); partner ``<row_id>_absent`` in the release -> A4 relabel from the partner
    (gold answer / action / action set / B7 subtype; reference_only True, absent_reason retrieval_miss, memory
    behaviour clarify_or_abstain, no keyphrases / citations / support; the ROW's risk is kept and safety_priority is
    recomputed with s4_rows.safety_priority); no partner -> EXCLUDE (retrieval_miss_a4_unverified).
Per-row tag ``absent_case``: ``never_saved`` (designed absent: real retrieval over a store without the target),
``saved_but_missed`` (target in the store, not in the online top-K -> relabelled), ``present`` (use row, targets
retrieved), ``no_memory`` (G0). Excluded rows carry their reason and no item.
K=8 returns the release row object unchanged (byte-identical to the default path) after the equivalence check.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from eval_v1 import topk_context as TK

FC = Path(__file__).resolve().parents[1]
BENCH = FC.parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

logger = logging.getLogger(__name__)
RETR = "siglip2"
SIDECARS = (FC / "eval_image_correction_v1" / "regen_v1" / "recompute" / "sidecar",
            FC / "v4_build" / "blite_recompute" / "sidecar")
SWAP_TAGS = ("regen_v1", "regen_v2", "blite_v4", "acute_add_v4c", "collapsed_v1", "acute_v5")  # acute_add_v4c: orchestrator 2026-10-14 (RUNBOOK §6)
# collapsed_v1 (v4h): a second query image of an existing family; its views live under the row block's
# ``query_image_key`` ("<pool image_key>@collapsed_v1") so the family's original query views stay untouched
QUERY_KEY_TAGS = ("collapsed_v1", "acute_v5")  # acute_v5 (v4i): same mechanism
# memswap_v4f (2026-10-14, v4_build/build_v4f): NON-target store entries replaced (flagged identities -> substitutes),
# row block ``replaced_store_identity`` (plan iid -> new id); rows without the block are unaffected
STORE_SWAP_TAGS = ("memswap_v4f",)
CASES = ("present", "saved_but_missed", "never_saved", "no_memory")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_sidecar(d: Path) -> Optional[tuple[dict, dict, dict]]:
    """(gallery, queries, info) of ``d/embed_siglip2.{npz,json}`` (sha-checked), or None if absent."""
    import numpy as np
    js, npz = Path(d) / f"embed_{RETR}.json", Path(d) / f"embed_{RETR}.npz"
    if not js.is_file():
        return None
    meta = json.loads(js.read_text())
    if _sha(npz) != meta["npz_sha256"]:
        raise RuntimeError(f"sidecar changed: {npz}")
    with np.load(npz) as a:
        g, q = a["gallery"], a["queries"]
    return ({i: g[n] for n, i in enumerate(meta["gallery_ids"])}, {k: q[n] for n, k in enumerate(meta["query_keys"])},
            {"dir": str(d), "npz_sha256": meta["npz_sha256"], "n_queries": len(meta["query_keys"]),
             "n_gallery": len(meta["gallery_ids"])})


class _World:
    """s1_plan.views / retrieve interface (``emb[retr] = (gallery, queries)``) over one profile."""

    def __init__(self, g: dict, q: dict) -> None:
        self.emb = {RETR: (g, q)}

    def sim(self, retr: str, qkey: str, iid: str) -> float:
        g, q = self.emb[retr]
        return float(q[qkey] @ g[iid])


def overlay(g: dict, q: dict, side: tuple[dict, dict, dict]) -> tuple[dict, dict]:
    """retrieve_relabel.World rule: new gallery ids added; ALL views of a re-embedded image replaced."""
    sg, sq, _ = side
    new_imgs = {k.split("#", 1)[0] for k in sq}
    return {**g, **sg}, {**{k: v for k, v in q.items() if k.split("#", 1)[0] not in new_imgs}, **sq}


class OnlineRetriever:
    def __init__(self, query_emb: Optional[Path] = None, sidecars: tuple[Path, ...] = SIDECARS,
                 release: Optional[Path] = None) -> None:
        """``release``: a release may declare extra gallery sidecars in RETRIEVAL_SIDECARS.json (v4 memdiv variant:
        re-embedded memory photos); releases without that file use ``sidecars`` only."""
        from final_construction.wp_v9_vcm import common as cm
        self.cm = cm
        decl = Path(release) / "RETRIEVAL_SIDECARS.json" if release is not None else None
        if decl is not None and decl.is_file():
            extra = [Path(p) for p in json.loads(decl.read_text())["gallery_sidecars"]]
            missing = [str(p) for p in extra if load_sidecar(p) is None]
            if missing:
                raise FileNotFoundError(f"{decl} declares sidecars without embeddings: {missing}")
            sidecars = (*sidecars, *extra)
        self.plan: dict[str, dict] = {}
        self.worlds: dict[str, _World] = {}
        self.cache: dict[tuple, list[str]] = {}
        sides = [s for s in (load_sidecar(d) for d in sidecars) if s is not None]
        fresh = load_sidecar(query_emb) if query_emb else None
        if query_emb and fresh is None:
            raise FileNotFoundError(f"fresh query embeddings missing: {query_emb}")
        self.info = {"retriever": RETR, **cm.RETRIEVERS[RETR], "sidecars": [s[2] for s in sides],
                     "query_source": f"fresh:{fresh[2]['dir']}" if fresh else "frozen_s0+sidecars",
                     "fresh": fresh[2] if fresh else None, "display_order": "seeded display_perm restricted to top-K"}
        for prof in ("holdout", "prod"):
            cfg = cm.Cfg(**cm.PROFILES[prof])
            for f in cm.load_plan(cfg)["families"]:
                self.plan[f["family_id"]] = f
            meta = cm.read(cfg.p("s0", f"embed_{RETR}.json"))
            npz = cfg.p("s0", f"embed_{RETR}.npz")
            if cm.sha_file(npz) != meta["npz_sha256"]:
                raise RuntimeError(f"frozen embedding file changed: {npz}")
            import numpy as np
            with np.load(npz) as a:
                ga, qa = a["gallery"], a["queries"]
            g = {i: ga[n] for n, i in enumerate(meta["gallery_ids"])}
            q = {k: qa[n] for n, k in enumerate(meta["query_keys"])}
            for s in sides + ([fresh] if fresh else []):
                g, q = overlay(g, q, s)
            self.worlds[prof] = _World(g, q)

    @staticmethod
    def profile(fid: str) -> str:
        return "holdout" if fid.startswith("family:s2h") else "prod"

    def ranking(self, row: dict) -> list[str]:
        """Concept keys of the merged online ranking (rank order) for this row's store variant."""
        from final_construction.wp_v9_vcm import s1_plan
        ev = TK.evaluator(row)
        fam = self.plan[row["family_id"]]
        swap: dict = {}
        for t in SWAP_TAGS:
            if isinstance(row.get(t), dict):
                swap.update(row[t].get("replaced_identity") or {})
        store_swap: dict = {}
        for t in STORE_SWAP_TAGS:
            if isinstance(row.get(t), dict):
                store_swap.update(row[t].get("replaced_store_identity") or {})
        variant = "absent" if row["row_type"].endswith("_absent") else "full"
        qkey = next((row[t]["query_image_key"] for t in QUERY_KEY_TAGS
                     if isinstance(row.get(t), dict) and row[t].get("query_image_key")), fam["image_key"])
        key = (row["family_id"], variant, qkey, tuple(sorted(swap.items())), tuple(sorted(store_swap.items())))
        if key not in self.cache:
            iid = {k: (swap.get(c["iid"], c["iid"]) if k in fam["targets"] else store_swap.get(c["iid"], c["iid"]))
                   for k, c in fam["concepts"].items()}
            key_of = {v: k for k, v in iid.items()}
            store = fam["absent_store"] if variant == "absent" else fam["store"]
            ctx = s1_plan.retrieve(self.worlds[self.profile(row["family_id"])], RETR, qkey,
                                   [iid[k] for k in store])
            self.cache[key] = [key_of[e["iid"]] for e in ctx["merged_ranking"]]
        if list(ev.get("target_keys") or []) != list(fam["targets"]):
            raise ValueError(f"{ev['row_id']}: target keys differ from the plan")
        return self.cache[key]


def partner_index(rows: list[dict]) -> dict[str, dict]:
    return {TK.evaluator(r)["row_id"]: r for r in rows}


def _info(k: int, case: Optional[str], online: list[str], eq_k: bool, eq_8: bool, retained: Optional[bool] = None,
          relabelled: bool = False, excluded: Optional[str] = None, source: Optional[str] = None) -> dict:
    return {"context_k": k, "online_k": k, "target_retained": retained, "relabelled": relabelled,
            "topk_excluded": excluded is not None, "topk_exclusion": excluded, "relabel_source": source,
            "absent_case": case, "online_topk": online, "online_equals_stored_topk": eq_k,
            "online_equals_stored_top8": eq_8}


def online_row(row: dict, k: int, retr: OnlineRetriever, partners: dict[str, dict]) -> tuple[Optional[dict], dict]:
    """(row restricted to the online top-K with retrieve_relabel labels | None if excluded, per-row info)."""
    from final_construction.wp_v9_vcm import s1_plan, s4_rows
    TK.check_k(k)
    ev = TK.evaluator(row)
    emb = ev.get("expected_memory_behavior")
    rt = row["row_type"]
    merged = retr.ranking(row)
    stored = [e["key"] for e in ev["retriever"]["ranked"]]
    n = len(row["contexts"])
    online = merged[:min(k, n)]
    eq_k, eq_8 = online == stored[:min(k, n)], merged[:n] == stored
    designed_absent = rt.endswith("_absent")
    case0 = "no_memory" if emb == "ignore" else "never_saved" if designed_absent else (
        "saved_but_missed" if ev.get("absent_reason") == "retrieval_miss" else "present")
    if not eq_k:
        return None, _info(k, case0, online, eq_k, eq_8, excluded="online_stored_mismatch")
    targets = list(ev.get("target_keys") or [])
    cond = s1_plan.condition(online, targets)
    use = emb == "use"
    if designed_absent and cond != "gt_absent":
        return None, _info(k, case0, online, eq_k, eq_8, excluded="absent_store_contains_target")
    if k >= n:
        return row, _info(k, case0, online, eq_k, eq_8, retained=True if use else None)
    keep = TK.kept_display(ev, n, k)
    new = {old: j + 1 for j, old in enumerate(keep)}
    gold = [int(g) for g in row["gold_citations"]]
    kept_gold = [g for g in gold if g in new]
    if use and cond == "partial":
        return None, _info(k, "present", online, eq_k, eq_8, retained=False, excluded="partial_retrieval_two_concept")
    ev2 = {**ev, "context_concepts": [ev["context_concepts"][i - 1] for i in keep],
           "display_to_rank": {str(new[int(d)]): r for d, r in ev["display_to_rank"].items() if int(d) in new},
           "support_patterns_by_display_index": TK._rekey(ev.get("support_patterns_by_display_index") or {}, new),
           "wrong_concept_patterns": TK._rekey(ev.get("wrong_concept_patterns") or {}, new)}
    out = {**row, "contexts": [row["contexts"][i - 1] for i in keep],
           "context_texts": [row["context_texts"][i - 1] for i in keep],
           "gold_citations": sorted(new[g] for g in kept_gold), "k": k}
    if not use:                                          # G0 / designed absent: display + condition only
        if emb == "ignore":
            out["context_condition"] = cond
        out["evaluator"] = ev2
        return out, _info(k, case0, online, eq_k, eq_8)
    if cond == "gt_present":
        out["evaluator"] = ev2
        return out, _info(k, case0, online, eq_k, eq_8, retained=True)
    # saved but missed at K (cond == gt_absent): retrieve_relabel A4 rule
    if rt == "G1_panic_memory":
        return None, _info(k, "saved_but_missed", online, eq_k, eq_8, retained=False,
                           excluded="panic_row_retrieval_miss")
    partner = partners.get(ev["row_id"] + "_absent")
    if partner is None:
        return None, _info(k, "saved_but_missed", online, eq_k, eq_8, retained=False,
                           excluded="retrieval_miss_a4_unverified")
    pev = TK.evaluator(partner)
    out.update(gold_answer=partner["gold_answer"], gold_action=partner["gold_action"],
               gold_action_set=partner["gold_action_set"], b7_subtype=partner["b7_subtype"], reference_only=True,
               gold_citations=[], context_condition="gt_absent")
    risk = "true" if row["risk"] == "unsafe" else "false"
    ev2.update(absent_reason="retrieval_miss", expected_memory_behavior="clarify_or_abstain", decisive_keyphrases=[],
               support_patterns_by_display_index={},
               safety_priority=s4_rows.safety_priority(risk, partner["gold_action"], row["grounding_requirement"], True))
    if pev.get("expected_memory_behavior") != "clarify_or_abstain":
        return None, _info(k, "saved_but_missed", online, eq_k, eq_8, retained=False,
                           excluded="partner_not_gt_absent")
    out["evaluator"] = ev2
    return out, _info(k, "saved_but_missed", online, eq_k, eq_8, retained=False, relabelled=True,
                      source=partner["sample_id"])


def summarize(infos: dict[str, dict]) -> dict:
    vals = list(infos.values())
    cases: dict[str, int] = {}
    excl: dict[str, int] = {}
    for i in vals:
        if i["topk_excluded"]:
            excl[i["topk_exclusion"]] = excl.get(i["topk_exclusion"], 0) + 1
        else:
            cases[i["absent_case"]] = cases.get(i["absent_case"], 0) + 1
    use = [i for i in vals if i["absent_case"] in ("present", "saved_but_missed")]       # saved-target rows
    missed = [i for i in use if i["target_retained"] is False]                             # incl. excluded misses
    relab = sum(i["relabelled"] for i in vals)
    return {"n_rows": len(vals), "absent_case": cases, "excluded": excl,
            # old-test convention (11.1 / 3.9 / 0.7 % at K = 1 / 2 / 4): relabelled rows / all rows
            "saved_but_missed_relabelled_share_all_rows": round(relab / len(vals), 4) if vals else None,
            # target (fully) not retrieved among saved-target rows, relabelled + excluded (partial counts as missed)
            "target_miss_rate_saved_rows": round(len(missed) / len(use), 4) if use else None,
            "n_online_stored_mismatch_topk": sum(not i["online_equals_stored_topk"] for i in vals),
            "n_online_stored_mismatch_top8": sum(not i["online_equals_stored_top8"] for i in vals)}


def equivalence_report(rows: list[dict], retr: OnlineRetriever, ks: tuple[int, ...] = TK.CONTEXT_KS) -> dict:
    """CPU check over a release split file: online top-K vs stored prefix, labels, cases, for every K."""
    def pending(r: dict) -> bool:          # b-lite / memdiv rows before their recompute (eval refuses them too)
        return any(isinstance(r.get(t), dict) and "retrieval_siglip2" in (r[t].get("pending_recompute") or [])
                   for t in ("blite_v4", "memdiv_v4", "acute_add_v4c", "memswap_v4f", "collapsed_v1", "acute_v5"))
    acc = [r for r in rows if TK.evaluator(r).get("accepted") and TK.evaluator(r).get("expected_memory_behavior")
           and not pending(r)]
    partners = partner_index(rows)
    out = {"retrieval": retr.info, "skipped_pending_recompute": sum(map(pending, rows))}
    for k in ks:
        infos = {}
        for r in acc:
            _, info = online_row(copy.deepcopy(r) if k < 8 else r, k, retr, partners)
            infos[r["sample_id"]] = info
        out[f"k{k}"] = summarize(infos)
    return out


__all__ = ["OnlineRetriever", "online_row", "partner_index", "summarize", "equivalence_report", "CASES"]
