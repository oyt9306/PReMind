"""Privileged-context OPSD (METHOD_PRIVILEGED_OPSD_v1 draft): hints, leak check, teacher table, selective gate, KL.

Stage-scoped teacher context (round 2, owner-approved hint reinforcement):
  * passive arm: the hint block is appended to the single user turn; every completion token is distilled;
  * memory-call arm: the Stage-1 turn is NEVER hinted (an absence hint there pushed the teacher to direct_answer);
    the block is appended to the Stage-2 environment turn and only Stage-2 tokens are distilled.
Hints (label-free behavioural text; the same template on every eligible row):
  H1 knowledge  rows G1G2_visual, G1G2_visual_absent, general_rule_closed: the row's ``general_knowledge_rule`` +
                fixed principles (match memory to the individual; cite only memories that change the answer;
                emergency only for a visible injury / acute hazard);
  H2 absence    rows whose arm condition is ``gt_absent``: the absence statement + do-not-guess principle;
  H3 identity   (ablation only) rows G1_visual / G1G2_visual with gold citations: "... corresponds to Memory [j]."
Leak check: no label / action names, XML tags, ids, entry keys, row-type or role markers, no gold-answer 5-gram,
no decisive keyphrase; a leaking hint is dropped for that row (recorded). Selective gate: the teacher table says the
hinted frozen base passes the Stage-2 reward checks (greedy, or best-of-n sampled) on the branch the row needs AND the
student episode took that branch and fails a Stage-2 check.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import paths_v2 as P
from . import reward_v5 as R5

TOPK = 16
LAMBDA = 0.2
TABLE_SCHEMA = "trainer_v2.teacher_table_v2"     # v2: stage-scoped hints, Stage-2 checks, best-of-n
HINT_IDS = ("H1", "H2", "H3")
HINT_SETS = {"main": ("H1", "H2"), "minus_H1": ("H2",), "minus_H2": ("H1",), "plus_H3": ("H1", "H2", "H3")}
H1_ROW_TYPES = ("G1G2_visual", "G1G2_visual_absent", "general_rule_closed")
H3_ROW_TYPES = ("G1_visual", "G1G2_visual")
H1_PRINCIPLES = ("Apply it to what is visible, together with a memory only if that memory shows the same individual."
                 " Cite a memory only when its fact changes your answer. Treat the situation as an emergency only if"
                 " an injury or an acute hazard is visible.")
H2_TEXT = ("None of the retrieved memories shows the individual in the new image. Facts in those memories belong to"
           " other individuals and do not apply here. If the correct response depends on a personal fact about this"
           " individual, do not guess; ask the user or say you cannot tell from the memories (after addressing any"
           " visible acute hazard).")
HEADER = "Verified evidence:"
GOLD_NGRAM = 5
_LABEL_LEAKS = re.compile(
    r"direct[_ ]answer|safety[_ ]escalation|\bescalat\w*|\bclarif\w*|\babstain\w*|call[_ ]pdb|memory[_ ]call|"
    r"\bunsafe\b|\bsafe\b|<\s*/?\s*\w+\s*>|\bG0\b|\bG1\b|\bG2\b|G1G2|_absent\b|_closed\b|\babsent\b|general[_ ]rule|"
    r"row[_ ]type|\bgold\b|\blabel\w*|expected[_ ]action|\bstratum\b|\brelaxed\b|knowledge[_ -]gap|closed question",
    re.I)
_TOK = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class TeacherCtx:
    """Teacher-side context of one item: first-turn messages + Stage-2 env contents (memory call)."""
    messages: list
    env_call: Optional[list] = None
    env_no_call: Optional[list] = None


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOK.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def leak_hits(text: str, item: Any) -> list[str]:
    """Label / id / row-type / gold-text hits in a teacher-only hint text (empty list = clean)."""
    from final_construction.wp_v9_vcm import s4_checks
    hits = [m.group(0) for m in _LABEL_LEAKS.finditer(text)] + s4_checks.leak_hits(text, set())
    for key in ("row_id", "family_id", "image_path", "image_sha256"):
        val = str(item.row.get(key) or "")
        if val and val in text:
            hits.append(key)
    hits += [k for k in item.target["rendered_keys"] if re.search(rf"\b{re.escape(k)}\b", text)]
    fields = getattr(item, "arm_fields", {}) or {}
    if _ngrams(text, GOLD_NGRAM) & _ngrams(fields.get("gold_answer") or "", GOLD_NGRAM):
        hits.append("gold_answer_5gram")
    for kp in fields.get("decisive_keyphrases") or []:
        if len(kp.split()) >= 2 and kp.lower() in text.lower():
            hits.append(f"decisive_keyphrase:{kp}")
    return hits


def hint_lines(item: Any, hint_ids: tuple[str, ...]) -> dict[str, str]:
    """Applicable hint texts for an item (before the leak check)."""
    out: dict[str, str] = {}
    rule = (item.row.get("general_knowledge_rule") or "").strip().rstrip(".")
    if "H1" in hint_ids and item.row.get("row_type") in H1_ROW_TYPES and rule:
        out["H1"] = f"Relevant general knowledge: {rule}. {H1_PRINCIPLES}"
    if "H2" in hint_ids and item.row.get("condition") == "gt_absent":
        out["H2"] = H2_TEXT
    gold = item.target["gold"]
    if "H3" in hint_ids and item.row.get("row_type") in H3_ROW_TYPES and gold:
        refs = " and ".join(f"Memory [{j}]" for j in gold)
        out["H3"] = (f"The visible item corresponds to {refs}." if len(gold) == 1
                     else f"The visible items correspond to {refs}.")
    return out


def build_hint(item: Any, hint_ids: tuple[str, ...]) -> tuple[Optional[str], dict]:
    """(hint block or None, audit). Leaking lines are dropped; H3's own "Memory [j]" reference is its content."""
    kept, dropped = {}, {}
    for hid, line in hint_lines(item, hint_ids).items():
        probe = re.sub(r"Memory \[\d+\]", "Memory", line) if hid == "H3" else line
        hits = leak_hits(probe, item)
        (dropped if hits else kept)[hid] = hits if hits else line
    if not kept:
        return None, {"hints": [], "dropped": dropped}
    block = HEADER + "\n" + "\n".join(f"- {kept[h]}" for h in HINT_IDS if h in kept)
    return block, {"hints": [h for h in HINT_IDS if h in kept], "dropped": dropped}


def _append(content: Optional[list], block: str) -> Optional[list]:
    if content is None:
        return None
    return copy.deepcopy(content) + [{"type": "text", "text": "\n\n" + block}]


def teacher_context(item: Any, block: str) -> TeacherCtx:
    """Passive: hint at the end of the user turn. Memory call: Stage 1 untouched, hint at the end of both Stage-2
    env turns. Copies only; the student item is never modified."""
    if item.arm == "passive":
        msgs = copy.deepcopy(item.messages)
        if msgs[-1]["role"] != "user":
            raise ValueError("first-turn prompt must end with the user message")
        msgs[-1]["content"] = _append(msgs[-1]["content"], block)
        return TeacherCtx(msgs)
    return TeacherCtx(item.messages, _append(item.env_call, block), _append(item.env_no_call, block))


def assert_student_untouched(item: Any, rendered_messages: list) -> None:
    """The student prompt is byte-identical to the inference render and carries no hint text."""
    a, b = json.dumps(item.messages, sort_keys=True), json.dumps(rendered_messages, sort_keys=True)
    if a != b:
        raise AssertionError(f"{item.row_id}: student prompt differs from the inference render")
    envs = json.dumps([item.env_call, item.env_no_call])
    if any(h in t for t in (a, envs) for h in (HEADER, H2_TEXT[:40], "Relevant general knowledge:")):
        raise AssertionError(f"{item.row_id}: student context carries teacher-only hint text")


def correct_decision(item: Any) -> str:
    return "call_pdb" if item.target["memory_required"] else "direct_answer"


# ----------------------------------------------------------------------------- reward checks + gate
def failure_reasons(ledger: dict, *, stage2_only: bool = False) -> list[str]:
    """Reward-check failures of one scored episode (empty = passes); ``stage2_only`` ignores the Stage-1 decision."""
    comps, info, tgt = ledger["components"], ledger.get("parse") or {}, ledger["target"]
    if comps["degenerate"]["applicable"]:
        return ["degenerate"]
    why = []
    if not (info.get("format_strict") and info.get("citation_valid")):
        why.append("format")
    if info.get("risk_pred") != tgt["risk_label"] and comps["risk"]["applicable"]:
        why.append("risk")
    act = comps["action"]
    if act["applicable"] and act["value"] < R5.W["action"]:
        why.append("action")
    if set(ledger["cited"]) != set(tgt["gold"]):
        why.append("citation")
    dec = comps["decision"]["value"]
    if not stage2_only and ledger["arm"] == "call" and dec is not None and dec < 0:
        why.append("decision")
    return why


def table_key(item: Any) -> str:
    return f"{item.arm}|{item.row_id}"


def table_entry(item: Any, audit: dict, block: Optional[str], results: list[tuple[str, dict, str]]) -> dict:
    """results = [(kind, ledger, text)]: greedy first, then the sampled best-of-n candidates."""
    if block is None:
        return {"teacher_ok": False, "reason": "no_clean_hint", **audit}
    checks = [(kind, failure_reasons(led, stage2_only=True), led, text) for kind, led, text in results]
    passing = [c for c in checks if not c[1]]
    ref = passing[0] if passing else checks[0]
    greedy_why = checks[0][1]
    return {"teacher_ok": bool(passing), "reason": "pass" if passing else "teacher_fails:" + ",".join(greedy_why),
            **audit, "hint_sha256": P.sha256_text(block), "reference": ref[0] if passing else None,
            "greedy_ok": not greedy_why, "greedy_reasons": greedy_why,
            "sample_reasons": [c[1] for c in checks[1:]], "n_samples_ok": sum(not c[1] for c in checks[1:]),
            "stage1_decision": correct_decision(item) if item.arm == "memory_call" else None,
            "teacher_total": ref[2]["total"], "teacher_text": ref[3]}


@dataclass
class OpsdContext:
    """Training-time OPSD state: hint set, teacher table entries keyed ``arm|row_id``."""
    hint_ids: tuple[str, ...]
    table: dict
    force: bool = False           # smoke plumbing only: gate every episode that has a clean hint

    @classmethod
    def load(cls, path: Path, hint_set: str, force: bool = False) -> "OpsdContext":
        P.refuse_holdout_path(path)
        data = json.loads(Path(path).read_text())
        if data.get("schema") != TABLE_SCHEMA:
            raise ValueError(f"teacher table schema {data.get('schema')!r} != {TABLE_SCHEMA} (rebuild the table)")
        if tuple(data["meta"]["hint_ids"]) != HINT_SETS[hint_set]:
            raise ValueError(f"teacher table was built for hints {data['meta']['hint_ids']}, not {hint_set}")
        if data["meta"].get("reward_v5_sha256") != R5.SOURCE_SHA256:
            logging.getLogger(__name__).warning("teacher table %s was scored with a different reward_v5 (sha %s); "
                                                "its teacher_ok flags may be stale", path,
                                                data["meta"].get("reward_v5_sha256"))
        return cls(HINT_SETS[hint_set], data["entries"], force)

    def teacher_context(self, item: Any) -> TeacherCtx:
        block, _ = build_hint(item, self.hint_ids)
        if block is None:
            raise AssertionError(f"{item.row_id}: gated without a clean hint")
        return teacher_context(item, block)

    def gate(self, item: Any, ledger: dict) -> tuple[bool, list[str]]:
        if self.force:
            return build_hint(item, self.hint_ids)[0] is not None, ["forced_smoke"] + failure_reasons(ledger)
        entry = self.table.get(table_key(item))
        if not entry or not entry.get("teacher_ok"):
            return False, []
        block, _ = build_hint(item, self.hint_ids)
        if block is None or P.sha256_text(block) != entry.get("hint_sha256"):
            return False, ["hint_changed_since_table"]
        if item.arm == "memory_call" and ledger["decision"] != entry["stage1_decision"]:
            return False, ["branch_not_validated"]       # Stage 1 is left to RLVR
        why = failure_reasons(ledger, stage2_only=True)
        return bool(why), why


def truncated_forward_kl(teacher_topk_logps: Any, student_logps_at_ids: Any, mask: Optional[Any] = None) -> Any:
    """KL(teacher || student) over the teacher's top-k (teacher renormalised), token mean over ``mask`` (T,)."""
    import torch
    p_t = torch.softmax(teacher_topk_logps.float(), dim=-1)
    kl_tok = (p_t * (torch.log(p_t.clamp_min(1e-12)) - student_logps_at_ids.float())).sum(-1)
    m = torch.ones_like(kl_tok) if mask is None else mask.float()
    return (kl_tok * m).sum() / m.sum().clamp_min(1.0)
