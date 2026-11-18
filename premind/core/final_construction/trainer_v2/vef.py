"""VEF = <visual_evidence> faithfulness reward (general_mix_v1/DESIGN.md §2): QA pools, per-step QA sampling, scoring
and the trainer-side context (judge prefetch, per-step stats, judge-down stop rule). Nothing here runs unless the
reward preset has ``V5Flags.vef`` set (reward_v5 presets "vef05" / "vef08"); every other preset is unchanged.

Pools and scoring are ports of general_mix_v1/vef_qas.py (``row_pool``) and vef_reward.py (``score``), byte-for-byte
in the QA texts and option draws (tests compare against out/vef_pool_*.jsonl). Differences:
  * ``sample_qas`` is keyed by (seed, row_id, step) (the offline helper used (row_id, step)): every rank of a DDP run
    and every resume draws the same 2-3 QAs for a prompt group at a global step;
  * the pool is built in memory from ``item.row["vef_src"]`` (release_data.convert copies it when ``vef=True``).
Per QA (n sampled): positive correct +1/n, D 0, wrong -1/n; negative D 0, any A-C -1/n; vef = W * sum, |vef| <= W.
The positive part needs the v5.2 format term; evidence > MAX_WORDS words = -W without a judge call; a judge failure
(after retries) zeroes the WHOLE prompt group (all G rollouts) and is logged; nothing is paid under the degenerate gate
(reward_v5 skips the component).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
import re
import statistics
from typing import Any, Optional

logger = logging.getLogger(__name__)
PRESETS = {"vef05": 0.05, "vef08": 0.08}      # W; v5.2 positive weight sum = 0.88
MAX_WORDS = 60
D_TEXT = "Cannot be determined from the description"
ANY = ("called", "not_called", "passive")
SALIENT = ("dog", "cat", "parrot", "horse", "bicycle", "guitar", "umbrella", "fish tank", "baby stroller",
           "christmas tree", "skateboard", "birdcage")
SUBJ_SKIP = ("electronics other", "toy other", "outdoor scene", "building")
HAZARD_CATS = {
    "a person who has fallen, is falling or is lying on the floor": ("person", "man", "woman"),
    "a wet or slippery floor": ("wet", "water", "spill", "slipping"),
    "exposed or sparking electrical wiring or cables": ("electrical", "wiring", "sparking", "cord", "cable"),
    "a heater or flame next to flammable material": ("heater", "flame", "fire", "candle", "stove"),
    "toxic vapor, smoke or fumes": ("vapor", "smoke", "fume", "gas"),
}
FALL_WORDS = ("floor", "fall", "lying", "slump", "balance", "tripping", "slipping", "unresponsive")
Q_HAZKIND = "According to the description, what kind of hazard is visible in the scene?"
Q_SUBJ = "According to the description, what kind of subject relevant to the user's request is visible?"
Q_NHAZ = "According to the description, which of these hazards is visible in the scene?"
Q_NOBJ = "According to the description, which of these is visible in the scene?"
JUDGE_TEMPLATE = """You are given a description of an image. Answer the multiple-choice question using ONLY the description.
- Do NOT use outside knowledge and do NOT assume anything the description does not clearly state.
- If one of A/B/C is explicitly or clearly supported by the description, choose it; otherwise choose D.
Output only the final choice in the exact format: Answer: \\boxed{{X}} where X is one of A, B, C, D.

[Description]
{desc}

[Question]
{question}
A. {A}
B. {B}
C. {C}
D. {D}"""
_VE = re.compile(r"<visual_evidence>(.*?)</visual_evidence>", re.S | re.I)
_BOX = re.compile(r"\\boxed\{\s*([ABCD])\s*\}")
STOP_FRAC, STOP_STEPS = 0.10, 3            # > 10 % failed groups on 3 consecutive steps -> VEF JUDGE DOWN
STEP_DEADLINE_S = 90.0
ALARM_WORDS, ALARM_STEPS = 1.5 * 26, 20     # §2.6.4 monitor: median words > 1.5 x 26, or rho(words, vef) > 0, 20 steps


def template_sha() -> str:
    """sha256 of everything that defines the QA texts and the judge prompt (pinned by the VEF PREREG)."""
    blob = json.dumps([D_TEXT, SALIENT, SUBJ_SKIP, HAZARD_CATS, FALL_WORDS, Q_HAZKIND, Q_SUBJ, Q_NHAZ, Q_NOBJ,
                       JUDGE_TEMPLATE, MAX_WORDS, PRESETS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------- pools (port of vef_qas.py)
def hazard_cats(what: str) -> list[str]:
    w = what.lower()
    toks = set(re.findall(r"[a-z]+", w))
    out = []
    for c, kws in HAZARD_CATS.items():
        hit = (bool(toks & set(kws)) and any(f in w for f in FALL_WORDS)) if c.startswith("a person") \
            else any(k in w for k in kws)
        out += [c] if hit else []
    return out


def readable(fine_type: str) -> str:
    return fine_type.replace("_or_", " or ").replace("_", " ")


def _rng(*key: object) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16))


def _row_words(r: dict) -> set[str]:
    ev = r["evaluator"]
    haz = ev.get("acute_visual_hazard") or {}
    text = " ".join([r.get("query_text") or "", r.get("gold_answer") or "", haz.get("evidence") or "",
                     (haz.get("image_verdict") or {}).get("what") or "", ev.get("general_knowledge_rule") or "",
                     *r.get("context_texts", [])])
    return set(re.findall(r"[a-z]+", text.lower()))


def _clean(option: str, words: set[str]) -> bool:
    return not any(w in words for w in re.findall(r"[a-z]{4,}", option.lower()))


def _mcq(qid: str, pol: str, question: str, opts: list[str], correct_text: Optional[str], rng: random.Random) -> dict:
    opts = list(opts)
    rng.shuffle(opts)
    return {"id": qid, "polarity": pol, "question": question, "options": dict(zip("ABC", opts)) | {"D": D_TEXT},
            "correct": "D" if correct_text is None else "ABC"[opts.index(correct_text)], "branches": list(ANY)}


def source_row(row: dict) -> dict:
    """The release-row fields ``row_pool`` reads (copied into ``VcmItem.row["vef_src"]`` by release_data.convert)."""
    ev = row["evaluator"] if isinstance(row["evaluator"], dict) else json.loads(row["evaluator"])
    return {"evaluator": {"row_id": ev["row_id"], "acute_visual_hazard": ev.get("acute_visual_hazard"),
                          "general_knowledge_rule": ev.get("general_knowledge_rule"),
                          "context_concepts": [{"fine_type": c["fine_type"], "role": c["role"]}
                                               for c in ev.get("context_concepts") or []]},
            "row_type": row["row_type"], "gold_citations": list(row.get("gold_citations") or []),
            "query_text": row.get("query_text"), "gold_answer": row.get("gold_answer"),
            "context_texts": list(row.get("context_texts") or [])}


def row_pool(r: dict) -> list[dict]:
    """All verification QAs of one row (vef_qas.row_pool, identical draws)."""
    ev, rid = r["evaluator"], r["evaluator"]["row_id"]
    words = _row_words(r)
    types = {readable(c["fine_type"]) for c in ev.get("context_concepts") or []}
    haz = ev.get("acute_visual_hazard") or {}
    own = hazard_cats((haz.get("image_verdict") or {}).get("what") or "")
    other_h = [c for c in HAZARD_CATS if c not in own and (haz.get("present") or _clean(c, words))]
    free_s = [s for s in SALIENT if s not in types and _clean(s, words)]
    out: list[dict] = []
    if haz.get("present") and own and len(other_h) >= 2:
        c = _rng(rid, "phk-c").choice(own)
        out.append(_mcq("P-HAZKIND", "pos", Q_HAZKIND, [c, *_rng(rid, "phk").sample(other_h, 2)], c, _rng(rid, "phk-o")))
    tgt = [c for c in ev.get("context_concepts") or [] if c["role"] == "target"]
    if r["row_type"] in ("G1_visual", "G1G2_visual") and r.get("gold_citations") and len(tgt) == 1 \
            and readable(tgt[0]["fine_type"]) not in SUBJ_SKIP and not tgt[0]["fine_type"].startswith("singleton") \
            and len(free_s) >= 2:
        t = readable(tgt[0]["fine_type"])
        out.append(_mcq("P-SUBJ", "pos", Q_SUBJ, [t, *_rng(rid, "psj").sample(free_s, 2)], t, _rng(rid, "psj-o")))
    if len(other_h) >= 3:
        out.append(_mcq("N-HAZ", "neg", Q_NHAZ, _rng(rid, "nh").sample(other_h, 3), None, _rng(rid, "nh-o")))
    if len(free_s) >= 3:
        out.append(_mcq("N-OBJ", "neg", Q_NOBJ, _rng(rid, "no").sample(free_s, 3), None, _rng(rid, "no-o")))
    return out


def sample_qas(pool: list[dict], row_id: str, step: int, seed: int) -> list[dict]:
    """2-3 QAs per prompt group and global step (at least one negative, at most one per id), deterministic in
    (seed, row_id, step); every episode of the group gets the same QAs."""
    rng = _rng(seed, row_id, "step", step)
    neg = [q for q in pool if q["polarity"] == "neg"]
    if not neg:
        return []
    first = rng.choice(neg)
    rest = [q for q in pool if q is not first]
    return [first, *rng.sample(rest, min(len(rest), rng.choice((1, 2))))]


# ----------------------------------------------------------------------------- scoring (port of vef_reward.py)
def visual_evidence(text: str) -> tuple[str, int]:
    """(field text, word count); a missing field is an empty description."""
    m = _VE.search(text or "")
    ve = m.group(1).strip() if m else ""
    return ve, len(ve.split())


def judge_prompt(desc: str, qa: dict) -> str:
    return JUDGE_TEMPLATE.format(desc=desc or "(empty)", question=qa["question"], **qa["options"])


def parse_choice(reply: str) -> str:
    m = _BOX.findall(reply or "")
    return m[-1] if m else "D"            # CoViP: unparsable -> D


def qa_score(qa: dict, choice: str) -> int:
    if choice == "D":
        return 0
    return 1 if (qa["polarity"] == "pos" and choice == qa["correct"]) else -1


def score(qas: list[dict], replies: Optional[list[str]], *, preset: str, format_ok: bool,
          judge_failed: bool = False) -> dict:
    """VEF component of one episode. ``replies`` None = over-long evidence (-W, no judge call)."""
    w = PRESETS[preset]
    if not qas:
        return {"applicable": False, "value": None, "status": "not_applicable:no_qa"}
    if judge_failed:
        return {"applicable": True, "value": 0.0, "status": "judge_failed:zero"}
    if replies is None:
        return {"applicable": True, "value": -w, "status": "active:overlong"}
    s = [qa_score(q, parse_choice(r)) for q, r in zip(qas, replies, strict=True)]
    pos = sum(v for v, q in zip(s, qas) if q["polarity"] == "pos" and v > 0)
    neg = sum(v for v in s if v < 0)
    value = max(-w, min(w, w * ((pos if format_ok else 0) + neg) / len(qas)))      # cap |vef| <= W
    return {"applicable": True, "value": round(value, 10), "status": "active", "scores": s}


# ----------------------------------------------------------------------------- trainer context
def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        for _, grp in itertools.groupby(order, key=v.__getitem__):
            idx = list(grp)
            for i in idx:
                r[i] = (order.index(idx[0]) + order.index(idx[-1])) / 2
        return r
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 6) if den else None


class VefContext:
    """Per-run VEF state of one rank: pools, the judge client, per-step counters and the judge-down tracker."""

    def __init__(self, preset: str, seed: int, judge: Any, *, deadline_s: float = STEP_DEADLINE_S) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown VEF preset {preset!r}")
        self.preset, self.seed, self.judge, self.deadline_s = preset, seed, judge, deadline_s
        self.pools: dict[str, list[dict]] = {}
        self.bad_streak = self.len_streak = self.rho_streak = 0
        self.down = False
        self._new_step(None)

    def _new_step(self, step: Optional[int]) -> None:
        self.step = step
        self.counts = {"groups": 0, "failed_groups": 0, "calls": 0, "failed_calls": 0, "cache_hits": 0}
        self.judge_s = 0.0

    def pool(self, item: Any) -> list[dict]:
        if item.row_id not in self.pools:
            src = item.row.get("vef_src")
            if src is None:
                raise ValueError(f"{item.row_id}: no vef_src (load the release with vef=True)")
            self.pools[item.row_id] = row_pool(src)
        return self.pools[item.row_id]

    def prefetch(self, eps: list, step: Optional[int]) -> None:
        """Judge every episode's <visual_evidence> against its group's sampled QAs; sets ``ep.vef``."""
        if step is None:
            raise ValueError("VEF needs the global step index (train_step(step=...))")
        if step != self.step:
            self._new_step(step)
        groups = [list(g) for _, g in itertools.groupby(eps, key=lambda e: id(e.item))]
        need: dict[int, list[str]] = {}
        for gi, grp in enumerate(groups):
            qas = sample_qas(self.pool(grp[0].item), grp[0].item.row_id, step, self.seed)
            need[gi] = []
            for ep in grp:
                desc, n = visual_evidence(ep.stage2_text)
                over = n > MAX_WORDS
                prompts = [] if (over or not qas) else [judge_prompt(desc, q) for q in qas]
                ep.vef = {"qas": qas, "replies": None, "judge_failed": False, "words": n, "overlong": over,
                          "_prompts": prompts}
                need[gi] += prompts
        uniq = list(dict.fromkeys(p for ps in need.values() for p in ps))
        replies, stats = self.judge.run(uniq, max(0.0, self.deadline_s - self.judge_s)) if uniq else ({}, {})
        self.judge_s += stats.get("seconds", 0.0)
        self.counts["calls"] += len(uniq)
        self.counts["cache_hits"] += stats.get("cache_hits", 0)
        self.counts["failed_calls"] += sum(replies.get(p) is None for p in uniq)
        for gi, grp in enumerate(groups):
            failed = any(replies.get(p) is None for p in need[gi])
            if grp[0].vef["qas"]:
                self.counts["groups"] += 1
                self.counts["failed_groups"] += failed
            if failed:
                logger.warning("VEF judge failed for group %s at step %s: group vef = 0", grp[0].item.row_id, step)
            for ep in grp:
                ps = ep.vef.pop("_prompts")
                ep.vef.update(judge_failed=failed, replies=None if (failed or not ps)
                              else [replies[p] for p in ps])

    def raw(self, eps: list) -> dict:
        """Mergeable per-rank record of the step (ddp_v2.merge_step concatenates / sums)."""
        per = []
        for ep in eps:
            v, comp = ep.ledger.get("vef") or {}, ep.ledger["components"].get("vef") or {}
            per.append({"words": v.get("words"), "value": comp.get("value"), "status": comp.get("status"),
                        "pol": v.get("polarity") or [], "scores": v.get("scores")})
        return {"counts": dict(self.counts), "judge_s": round(self.judge_s, 3), "eps": per}

    def observe(self, raw: dict) -> dict:
        """Step summary for steps.jsonl / wandb + the judge-down rule (identical on every rank: merged input)."""
        c, eps = raw["counts"], raw["eps"]
        vals = [e["value"] for e in eps if e["value"] is not None]
        words = [e["words"] for e in eps if e["words"] is not None]
        pos = [s for e in eps if e["scores"] for p, s in zip(e["pol"], e["scores"]) if p == "pos"]
        neg = [s for e in eps if e["scores"] for p, s in zip(e["pol"], e["scores"]) if p == "neg"]
        paired = [(e["words"], e["value"]) for e in eps if e["value"] is not None and e["words"] is not None]
        frac = c["failed_groups"] / c["groups"] if c["groups"] else 0.0
        self.bad_streak = self.bad_streak + 1 if frac > STOP_FRAC else 0
        self.down = self.down or self.bad_streak >= STOP_STEPS
        rho = _spearman([float(a) for a, _ in paired], [float(b) for _, b in paired])
        self.len_streak = self.len_streak + 1 if words and statistics.median(words) > ALARM_WORDS else 0
        self.rho_streak = self.rho_streak + 1 if rho is not None and rho > 0 else 0
        out = {"preset": self.preset, "W": PRESETS[self.preset],
               "reward_mean": round(statistics.fmean(vals), 6) if vals else None, **c,
               "failed_group_frac": round(frac, 6), "judge_s": raw["judge_s"],
               "qa_pos_acc": round(sum(s == 1 for s in pos) / len(pos), 6) if pos else None,
               "qa_pos_wrong": round(sum(s == -1 for s in pos) / len(pos), 6) if pos else None,
               "halluc_rate": round(sum(s == -1 for s in neg) / len(neg), 6) if neg else None,
               "evidence_words_mean": round(statistics.fmean(words), 3) if words else None,
               "evidence_words_median": statistics.median(words) if words else None,
               "overlong_frac": round(sum(e["status"] == "active:overlong" for e in eps) / len(eps), 6) if eps else None,
               "spearman_words_vef": rho, "len_alarm_streak": self.len_streak, "rho_alarm_streak": self.rho_streak,
               "bad_streak": self.bad_streak, "judge_down": self.down}
        if max(self.len_streak, self.rho_streak) >= ALARM_STEPS:   # log only: the owner stops and reviews (no auto-tune)
            logger.warning("VEF MONITOR ALARM (length / length-vef correlation for %d steps): %s", ALARM_STEPS, out)
        if self.down:
            logger.error("VEF JUDGE DOWN: failed-group share > %.0f%% on %d consecutive steps (%s)",
                         100 * STOP_FRAC, STOP_STEPS, out)
        return out


def merge_raw(parts: list[dict]) -> dict:
    """Rank merge of ``VefContext.raw`` (rank order = global prompt order)."""
    keys = parts[0]["counts"]
    return {"counts": {k: sum(p["counts"][k] for p in parts) for k in keys},
            "judge_s": max(p["judge_s"] for p in parts), "eps": [e for p in parts for e in p["eps"]]}


def wandb_scalars(summary: dict) -> dict:
    return {f"train/vef/{k}": (int(v) if isinstance(v, bool) else v) for k, v in summary.items()
            if isinstance(v, (int, float)) and v is not None}

