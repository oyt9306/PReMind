"""VEF v2 = <visual_evidence> faithfulness with an explicit identity check (general_mix_v1/DESIGN_v2.md).

v1 (vef.py, presets vef05 / vef08) is unchanged. v2 (reward preset "vef2", prompt schema CONFIDANT_SCHEMA=vef2) asks for
two parts inside <visual_evidence>: "Scene: <short scene description>" and "Match: <for each request-relevant person,
pet or object: the memory it matches as Memory j (name), or no matching memory>".
  * Scene part: scored exactly like v1 (vef.py QA pools, per-step sampling, judge prompt, per-QA rules, cap) with
    W_scene = 0.05, but the judge sees only the Scene part, and the positive scene credit is paid only when a
    parseable Match part is present (and the v5.2 format term passed). The negative (hallucination) part always applies.
  * Match part: scored by RULE against the row's gold annotations (no judge), W_match = 0.05, on memory-required rows
    where memories were shown (passive arm, or memory-call arm after call_pdb):
      +W  the claimed display indices equal the gold set (the present targets), or the target is absent and the part
          says "no matching memory" and claims nothing;
      -W  any claimed index outside the gold set (a look-alike, another entry, an index outside 1..K, or any claim
          when the target is absent);
       0  Match part missing / unparseable, a partial match, or "no matching memory" while the target is present.
    The positive part needs the v5.2 format term (as v1); the negative part always applies.
  * Whole evidence > MAX_WORDS (80) words: -(W_scene + W_match), no judge call.
  * Judge failure (after retries): the scene part of the whole prompt group is 0 (group-consistent, logged, counted in
    the v1 judge-down rule); the rule-based match part is unaffected.
Names are mapped to display indices through the row's own memory entries (evaluator.context_concepts names / handles,
in the rendered order ``target["rendered_keys"]``), so the order=rank ablation and CONTEXT_K_MIX stay correct.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import statistics
from typing import Any, Optional

from . import vef as VF

logger = logging.getLogger(__name__)
PRESETS = {"vef2": {"W_scene": 0.05, "W_match": 0.05}}
assert all(p["W_scene"] == VF.PRESETS["vef05"] for p in PRESETS.values())   # scene part = v1 scoring at vef05's W
MAX_WORDS = 80
ALARM_WORDS = 0.75 * MAX_WORDS           # length monitor (log only): median evidence words above 60 for 20 steps
SCHEMA = "vef2"                          # train_interface_v1/final_prompts.py CONFIDANT_SCHEMA value the arm needs
LOOKALIKE_ROLES = ("negative", "replacement")
WRONG = ("lookalike", "claimed_absent", "wrong_entry")
_MATCH_LABEL = re.compile(r"(?im)(?<![a-z])match(?:es|ing)?\s*(?:\*\*|__)?\s*[:：]\s*(?:\*\*|__)?"
                          r"|^[\s\-*•>#]*match\s*[-–—]\s")
_SCENE_LABEL = re.compile(r"(?i)(?<![a-z])scene\s*(?:\*\*|__)?\s*[:：]\s*(?:\*\*|__)?")
_NONE = re.compile(r"(?i)\bno\s+(?:matching|matched|corresponding|such)\s+(?:memory|memories|entry|entries)\b"
                   r"|\bno\s+(?:memory|match)\b|\bnone\b|\bunmatched\b"
                   r"|\b(?:does\s+not|doesn't|do\s+not|don't)\s+match\s+any\b"
                   r"|\bnot\s+(?:in|among|shown\s+in|found\s+in)\s+(?:any\s+of\s+)?(?:the\s+)?memor")
_NEG = re.compile(r"(?i)\b(?:not|no|isn't|doesn't|don't|unlike|differ\w*|different|rather\s+than|neither|nor|"
                  r"instead\s+of|mistaken|confused)\b")
_NEG_AFTER = re.compile(r"(?i)^\W*(?:\([^)]*\)\s*)?(?:is|does|do|was)?\s*(?:not|n't|isn't|doesn't)\b")
_CLAUSE = re.compile(r"[,:(—–]|\s-\s|\b(?:and|but|while|whereas)\b", re.I)
_SEG = re.compile(r"[;\n•]+|(?<=[A-Za-z0-9)\]])\.\s+")
_IDX = re.compile(r"(?i)\b(?:memories|entries)\s*((?:#\s*)?\d+(?:\s*(?:,|and|&|/|or)\s*(?:#\s*)?\d+)*)"
                  r"|\b(?:memory|entry|mem\.?)\s*(?:#|no\.?|number)?\s*(\d+)|#\s*(\d+)|\bM(\d+)\b"
                  r"|\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def prompt_text() -> str:
    """The v2 output-format block (final_prompts._OUTPUT_FORMATS["vef2"])."""
    from final_construction.train_interface_v1 import final_prompts as FP
    return FP._OUTPUT_FORMATS[SCHEMA]


def prompt_line() -> str:
    """The v2 <visual_evidence> instruction line (the rest of the format block is edited by the prod render)."""
    return next(x.strip() for x in prompt_text().splitlines() if x.strip().startswith("<visual_evidence>"))


def template_sha() -> str:
    """sha256 of the v2 definition: weights, length cap, parser patterns, look-alike roles, the v1 QA / judge template
    sha (scene part) and the v2 output-format text."""
    blob = json.dumps([PRESETS, MAX_WORDS, SCHEMA, LOOKALIKE_ROLES, VF.template_sha(), prompt_text(),
                       [p.pattern for p in (_MATCH_LABEL, _SCENE_LABEL, _NONE, _NEG, _NEG_AFTER, _CLAUSE, _SEG, _IDX)]],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------- per-row memory table
def match_src(row: dict) -> dict:
    """Release-row fields the Match parser reads (release_data.convert stores them in ``vef_src["match"]``)."""
    ev = row["evaluator"] if isinstance(row["evaluator"], dict) else json.loads(row["evaluator"])
    return {"concepts": [{k: c.get(k) for k in ("key", "role", "name", "handle", "fine_type")}
                         for c in ev.get("context_concepts") or []]}


def _handle_core(handle: str) -> str:
    h = re.sub(r"(?i)^the\s+", "", handle.strip())
    h = re.sub(r"(?i)^one\s+from\s+(?:the\s+)?", "", h)
    return re.sub(r"(?i)\s+one$", "", h).strip()


def _type_words(fine_type: Optional[str], text: str) -> list[str]:
    words = {w for w in re.findall(r"[a-z]+", VF.readable(fine_type or "").lower()) if w not in ("or", "other")}
    m = re.match(r"(?i)^(?:the user's|an?)\s+([a-z][a-z\- ]{0,30}?)\s+(?:the user is looking after\s+)?—",
                 text or "")
    return sorted(words | (set(m.group(1).lower().split()) if m else set()))


def table(src: dict, rendered_keys: list[str]) -> list[dict]:
    """Memory entries in rendered display order: index, role, name / handle aliases and type words."""
    concepts = src["match"]["concepts"]
    texts = dict(zip([c["key"] for c in concepts], src.get("context_texts") or []))
    by_key = {c["key"]: c for c in concepts}
    out = []
    for j, key in enumerate(rendered_keys):
        c = by_key[key]
        out.append({"index": j + 1, "key": key, "role": c.get("role"), "name": (c.get("name") or "").strip(),
                    "handle": (c.get("handle") or "").strip(), "core": _handle_core(c.get("handle") or ""),
                    "types": _type_words(c.get("fine_type"), texts.get(key, ""))})
    return out


# ----------------------------------------------------------------------------- evidence parsing
def split_evidence(ve: str) -> tuple[str, Optional[str]]:
    """(scene text for the judge, match text or None). The Match part runs from its label to a later Scene label or
    the end; the scene is everything else with its Scene label removed."""
    m = _MATCH_LABEL.search(ve)
    if m is None:
        return _SCENE_LABEL.sub("", ve, count=1).strip(), None
    s_after = _SCENE_LABEL.search(ve, m.end())
    end = s_after.start() if s_after else len(ve)
    scene = _SCENE_LABEL.sub("", ve[:m.start()] + " " + ve[end:], count=1)
    return scene.strip(" \n\t-*•;"), ve[m.end():end].strip()


def _word(alias: str, text: str) -> list[re.Match]:
    return list(re.finditer(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", text))


def _refs(seg: str, tab: list[dict]) -> list[tuple[int, int, int]]:
    """(start, end, display index) references in one segment: explicit indices, names, then handles (a handle needs
    its full text, its core phrase plus a type word of that entry, or "<core> one" / "one from <core>" when the core
    is unique in the row; ambiguous handles are dropped)."""
    out: list[tuple[int, int, int]] = []
    for m in _IDX.finditer(seg):
        nums = next(g for g in m.groups() if g is not None)
        out += [(m.start(), m.end(), int(n)) for n in re.findall(r"\d+", nums)]
    low = seg.lower()
    names = seg if seg != low else None                      # names: as written (capitalised), or all-lowercase text
    for e in tab:
        if e["name"]:
            hits = _word(e["name"], seg) if names is not None else _word(e["name"].lower(), low)
            out += [(m.start(), m.end(), e["index"]) for m in hits]
    cores: dict[str, list[dict]] = {}
    for e in tab:
        if e["core"]:
            cores.setdefault(e["core"].lower(), []).append(e)
    for core, es in cores.items():
        hits = _word(core, low)
        if not hits:
            continue
        typed = {e["index"] for e in es if any(_word(w, low) or _word(w + "s", low) for w in e["types"])}
        full = {e["index"] for e in es if e["handle"].lower() in low}
        short = re.search(rf"(?<![a-z]){re.escape(core)}\s+one\b|\bone\s+from\s+(?:the\s+)?{re.escape(core)}(?![a-z])",
                          low)                         # "the conference one", "one from the fair": unique core only
        ok = typed | full | ({es[0]["index"]} if short and len(es) == 1 else set())
        if len(ok) > 1:
            ok = (typed & full) if len(typed & full) == 1 else set()
        out += [(m.start(), m.end(), i) for m in hits for i in ok]
    return sorted(out)


def _negated(seg: str, refs: list[tuple[int, int, int]], i: int) -> bool:
    start, end, _ = refs[i]
    prev_end = max([e for s, e, _ in refs[:i] if e <= start] or [0])
    before = seg[max(prev_end, start - 40):start]
    cuts = list(_CLAUSE.finditer(before))
    before = before[cuts[-1].end():] if cuts else before
    nxt = min([s for s, _, _ in refs[i + 1:] if s >= end] or [len(seg)])
    return bool(_NEG.search(before)) or bool(_NEG_AFTER.search(seg[end:min(nxt, end + 40)]))


def parse_match(match: Optional[str], tab: list[dict]) -> dict:
    """Claimed display indices (non-negated references), the "no matching memory" flag and parseability."""
    if match is None:
        return {"present": False, "parseable": False, "claimed": [], "none": False}
    claimed: set[int] = set()
    none = False
    for seg in _SEG.split(match):
        none = none or bool(_NONE.search(seg))
        refs = _refs(seg, tab)
        claimed |= {idx for i, (_, _, idx) in enumerate(refs) if not _negated(seg, refs, i)}
    return {"present": True, "parseable": bool(claimed) or none, "claimed": sorted(claimed), "none": none}


def match_info(ve: str, tab: list[dict]) -> dict:
    scene, match = split_evidence(ve)
    p = parse_match(match, tab)
    roles = {e["index"]: e["role"] for e in tab}
    p["lookalike"] = [i for i in p["claimed"] if roles.get(i) in LOOKALIKE_ROLES]
    p["out_of_range"] = [i for i in p["claimed"] if i not in roles]
    return {"scene": scene, **p}


# ----------------------------------------------------------------------------- scoring
def match_verdict(m: dict, t: dict, branch: str) -> tuple[bool, str]:
    """(applicable, verdict) of the Match part against the branch target ``t`` (reward_v5._branch_target)."""
    if branch == "not_called":
        return False, "not_applicable:no_memory_shown"
    if not t["memory_required"]:
        return False, "not_applicable:memory_not_required"
    if not m["parseable"]:
        return True, "unparseable" if m["present"] else "missing"
    gold, claimed = set(t["gold"]), set(m["claimed"])
    if claimed - gold:
        return True, "lookalike" if set(m["lookalike"]) - gold else ("claimed_absent" if not gold else "wrong_entry")
    if not gold:
        return True, "correct_none"
    return True, "correct" if claimed == gold else ("partial" if claimed else "none_when_present")


def score(qas: list[dict], replies: Optional[list[str]], *, preset: str, format_ok: bool, match: dict, t: dict,
          branch: str, judge_failed: bool = False, overlong: bool = False) -> dict:
    """VEF v2 component of one episode: scene part (v1 rules at W_scene) + rule-scored match part (W_match)."""
    ws, wm = PRESETS[preset]["W_scene"], PRESETS[preset]["W_match"]
    m_app, verdict = match_verdict(match, t, branch)
    base = {"verdict": verdict if m_app else None, "scene_value": None, "scene_scores": None, "match_value": None}
    if overlong:
        return {"applicable": True, "value": -round(ws + wm, 10), "status": "active:overlong", **base}
    scene = VF.score(qas, replies, preset="vef05", format_ok=format_ok and match["parseable"],
                     judge_failed=judge_failed)
    mv = None
    if m_app:
        mv = wm if verdict in ("correct", "correct_none") else (-wm if verdict in WRONG else 0.0)
        mv = 0.0 if (mv > 0 and not format_ok) else mv
    if not scene["applicable"] and not m_app:
        return {"applicable": False, "value": None, "status": "not_applicable:no_qa_no_match", **base}
    value = (scene["value"] or 0.0) + (mv or 0.0)
    status = "active:scene_judge_failed" if judge_failed and scene["applicable"] else "active"
    return {"applicable": True, "value": round(value, 10), "status": status,
            **base, "scene_value": scene["value"], "scene_scores": scene.get("scores"), "match_value": mv}


# ----------------------------------------------------------------------------- trainer context
class Vef2Context(VF.VefContext):
    """v1 VefContext state (pools, judge, counters, judge-down rule) with the v2 evidence split: the judge sees only
    the Scene part; the Match part is parsed here (per-item memory table) and scored by reward_v5 on the branch
    target. ``prompt_check``: refuse unless the rendered prompts carry the v2 schema (CONFIDANT_SCHEMA=vef2)."""

    def __init__(self, preset: str, seed: int, judge: Any, *, deadline_s: float = VF.STEP_DEADLINE_S,
                 prompt_check: bool = True) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown VEF v2 preset {preset!r}")
        super().__init__("vef05", seed, judge, deadline_s=deadline_s)       # v1 init (pools, counters)
        self.preset, self.prompt_check = preset, prompt_check
        self.tables: dict[str, list[dict]] = {}
        if prompt_check:
            from final_construction.train_interface_v1 import final_prompts as FP
            if FP.SCHEMA != SCHEMA:
                raise SystemExit(f"REFUSED: reward preset {preset} needs CONFIDANT_SCHEMA={SCHEMA} (got {FP.SCHEMA})")

    def table(self, item: Any) -> list[dict]:
        if item.row_id not in self.tables:
            src = item.row.get("vef_src") or {}
            if "match" not in src:
                raise ValueError(f"{item.row_id}: no vef_src.match (load the release with vef=True)")
            if self.prompt_check:
                sys_text = "".join(p.get("text", "") for p in item.messages[0]["content"])
                if prompt_line() not in sys_text:
                    raise SystemExit(f"REFUSED: {item.row_id}: the prompt lacks the v2 <visual_evidence> schema")
            self.tables[item.row_id] = table(src, item.target["rendered_keys"])
        return self.tables[item.row_id]

    def prefetch(self, eps: list, step: Optional[int]) -> None:
        """vef.VefContext.prefetch with the Scene part as the judge description, the 80-word cap on the whole
        evidence and the parsed Match part in ``ep.vef["match"]``."""
        if step is None:
            raise ValueError("VEF needs the global step index (train_step(step=...))")
        if step != self.step:
            self._new_step(step)
        groups = [list(g) for _, g in itertools.groupby(eps, key=lambda e: id(e.item))]
        need: dict[int, list[str]] = {}
        for gi, grp in enumerate(groups):
            qas = VF.sample_qas(self.pool(grp[0].item), grp[0].item.row_id, step, self.seed)
            tab = self.table(grp[0].item)
            need[gi] = []
            for ep in grp:
                ve, n = VF.visual_evidence(ep.stage2_text)
                info = match_info(ve, tab)
                over = n > MAX_WORDS
                prompts = [] if (over or not qas) else [VF.judge_prompt(info["scene"], q) for q in qas]
                ep.vef = {"qas": qas, "replies": None, "judge_failed": False, "words": n, "overlong": over,
                          "scene_words": len(info["scene"].split()), "match": info, "_prompts": prompts}
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
                logger.warning("VEF2 judge failed for group %s at step %s: group scene part = 0",
                               grp[0].item.row_id, step)
            for ep in grp:
                ps = ep.vef.pop("_prompts")
                ep.vef.update(judge_failed=failed, replies=None if (failed or not ps) else [replies[p] for p in ps])

    def raw(self, eps: list) -> dict:
        out = super().raw(eps)
        for rec, ep in zip(out["eps"], eps):
            v = ep.ledger.get("vef") or {}
            m = v.get("match") or {}
            rec.update(verdict=m.get("verdict"), parseable=m.get("parseable"), lookalike=bool(m.get("lookalike")),
                       gold_absent=m.get("gold_absent"), scene_value=v.get("scene_value"),
                       match_value=v.get("match_value"))
        return out

    def observe(self, raw: dict) -> dict:
        """v1 step summary (judge counters, QA accuracy / hallucination, words, judge-down rule) + the match stats."""
        c, eps = raw["counts"], raw["eps"]
        vals = [e["value"] for e in eps if e["value"] is not None]
        words = [e["words"] for e in eps if e["words"] is not None]
        pos = [s for e in eps if e["scores"] for p, s in zip(e["pol"], e["scores"]) if p == "pos"]
        neg = [s for e in eps if e["scores"] for p, s in zip(e["pol"], e["scores"]) if p == "neg"]
        paired = [(e["words"], e["value"]) for e in eps if e["value"] is not None and e["words"] is not None]
        frac = c["failed_groups"] / c["groups"] if c["groups"] else 0.0
        self.bad_streak = self.bad_streak + 1 if frac > VF.STOP_FRAC else 0
        self.down = self.down or self.bad_streak >= VF.STOP_STEPS
        rho = VF._spearman([float(a) for a, _ in paired], [float(b) for _, b in paired])
        self.len_streak = self.len_streak + 1 if words and statistics.median(words) > ALARM_WORDS else 0
        self.rho_streak = self.rho_streak + 1 if rho is not None and rho > 0 else 0
        r6 = (lambda num, den: round(num / den, 6) if den else None)
        w = PRESETS[self.preset]
        out = {"preset": self.preset, "W": round(w["W_scene"] + w["W_match"], 10), "W_scene": w["W_scene"],
               "W_match": w["W_match"], "reward_mean": round(statistics.fmean(vals), 6) if vals else None, **c,
               "failed_group_frac": round(frac, 6), "judge_s": raw["judge_s"],
               "qa_pos_acc": r6(sum(s == 1 for s in pos), len(pos)),
               "qa_pos_wrong": r6(sum(s == -1 for s in pos), len(pos)),
               "halluc_rate": r6(sum(s == -1 for s in neg), len(neg)),
               "evidence_words_mean": round(statistics.fmean(words), 3) if words else None,
               "evidence_words_median": statistics.median(words) if words else None,
               "overlong_frac": r6(sum(e["status"] == "active:overlong" for e in eps), len(eps)),
               "spearman_words_vef": rho, "len_alarm_streak": self.len_streak, "rho_alarm_streak": self.rho_streak,
               "bad_streak": self.bad_streak, "judge_down": self.down}
        app = [e for e in eps if e.get("verdict")]
        absent = [e for e in app if e.get("gold_absent")]
        sv = [e["scene_value"] for e in eps if e.get("scene_value") is not None]
        mv = [e["match_value"] for e in eps if e.get("match_value") is not None]
        parsed = [e for e in eps if e.get("parseable") is not None]
        out["match"] = {"n": len(app), "acc": r6(sum(e["verdict"] in ("correct", "correct_none") for e in app), len(app)),
                        "lookalike_rate": r6(sum(e["verdict"] == "lookalike" for e in app), len(app)),
                        "missing_rate": r6(sum(e["verdict"] in ("missing", "unparseable") for e in app), len(app)),
                        "wrong_rate": r6(sum(e["verdict"] in WRONG for e in app), len(app)),
                        "absent_n": len(absent),
                        "absent_acc": r6(sum(e["verdict"] == "correct_none" for e in absent), len(absent)),
                        "parseable_frac_all": r6(sum(bool(e["parseable"]) for e in parsed), len(parsed)),
                        "scene_reward_mean": round(statistics.fmean(sv), 6) if sv else None,
                        "match_reward_mean": round(statistics.fmean(mv), 6) if mv else None}
        out.update({f"match_{k}": v for k, v in out["match"].items()})     # flat copies (vef.wandb_scalars)
        if max(self.len_streak, self.rho_streak) >= VF.ALARM_STEPS:
            logger.warning("VEF2 MONITOR ALARM (length / length-vef correlation for %d steps): %s", VF.ALARM_STEPS, out)
        if self.down:
            logger.error("VEF JUDGE DOWN: failed-group share > %.0f%% on %d consecutive steps (%s)",
                         100 * VF.STOP_FRAC, VF.STOP_STEPS, out)
        return out
