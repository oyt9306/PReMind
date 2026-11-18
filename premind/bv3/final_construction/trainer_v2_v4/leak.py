"""Cross-format leak and format-validity monitors for batch-level CoViP mixing (owner 2026-10-13), CPU only, logging only
(never part of the reward). Counted over every training rollout of a step (incl. DAPO refills) and over dev texts.

  leak_boxed     PReMind rollouts whose Stage-2 / passive answer contains "\\boxed" or is a bare index list such as
                 "[0, 2]" (whole <answer>, or an "Answer: [..]" line) - citation lists inside <context_use> do NOT count;
  leak_protocol  CoViP rollouts containing a PReMind protocol marker: <memory_call>, call_pdb / direct_answer, any of the
                 Stage-2 tags (<decision>, <visual_evidence>, <reasoning>, <context_use>, <risk>, <action>,
                 <confidence>, <answer>);
  stage1_valid   PReMind memory-call rollouts whose Stage 1 parses to call_pdb / direct_answer (adapter.parse_stage1);
  stage2_valid   PReMind rollouts whose Stage-2 / passive output passes the lineage strict format (rewards_v3 parse).
"""
from __future__ import annotations

import re
from typing import Optional

BOXED = re.compile(r"\\boxed\s*\{")
INDEX_LIST = re.compile(r"^\s*\[\s*\d+\s*(?:,\s*\d+\s*)*\]\s*\.?\s*$")
ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*\[\s*\d+\s*(?:,\s*\d+\s*)*\]\s*$")
ANSWER_TAG = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S | re.I)
PROTOCOL = re.compile(r"<\s*/?\s*(?:memory_call|decision|visual_evidence|reasoning|context_use|risk|action|confidence|"
                      r"answer)\s*>|\bcall_pdb\b|\bdirect_answer\b", re.I)


def leak_boxed(stage2: str) -> bool:
    text = stage2 or ""
    if BOXED.search(text) or ANSWER_LINE.search(text):
        return True
    m = ANSWER_TAG.search(text)
    return bool(INDEX_LIST.match(m.group(1))) if m else bool(INDEX_LIST.match(text))


def leak_protocol(text: str) -> bool:
    return bool(PROTOCOL.search(text or ""))


def stage1_valid(stage1: Optional[str]) -> Optional[bool]:
    if stage1 is None:
        return None
    import adapter as AD  # noqa: PLC0415  (train_interface_v1, on sys.path via trainer_v2.paths_v2)
    return AD.parse_stage1(stage1) in ("call_pdb", "direct_answer")


def stage2_valid(stage2: str) -> bool:
    from trainer_v2 import reward_v5 as R5
    return bool(R5.R2.parse(stage2 or "")["format_strict"])


class Counter:
    """Per-step counts (summed over ranks by the caller); ``rates`` gives the fractions."""
    KEYS = ("pm_n", "pm_boxed", "pm_s1_n", "pm_s1_ok", "pm_s2_ok", "aux_n", "aux_protocol")

    def __init__(self) -> None:
        self.c = dict.fromkeys(self.KEYS, 0)

    def add(self, *, aux: bool, stage1: Optional[str], stage2: str) -> None:
        if aux:
            self.c["aux_n"] += 1
            self.c["aux_protocol"] += int(leak_protocol((stage1 or "") + "\n" + (stage2 or "")))
            return
        self.c["pm_n"] += 1
        self.c["pm_boxed"] += int(leak_boxed(stage2))
        self.c["pm_s2_ok"] += int(stage2_valid(stage2))
        v = stage1_valid(stage1)
        if v is not None:
            self.c["pm_s1_n"] += 1
            self.c["pm_s1_ok"] += int(v)

    def values(self) -> list[int]:
        return [self.c[k] for k in self.KEYS]

    @classmethod
    def rates(cls, vals: list) -> dict:
        c = dict(zip(cls.KEYS, vals))
        d = lambda a, b: (c[a] / c[b]) if c[b] else None  # noqa: E731
        return {"leak_boxed": d("pm_boxed", "pm_n"), "leak_protocol": d("aux_protocol", "aux_n"),
                "stage1_valid": d("pm_s1_ok", "pm_s1_n"), "stage2_valid": d("pm_s2_ok", "pm_n"),
                "n_premind_rollouts": c["pm_n"], "n_covip_rollouts": c["aux_n"]}


def dev_summary(texts: list[dict]) -> dict:
    """The same rates on a dev eval's texts.jsonl (greedy rows; dev holds PReMind rows only)."""
    k = Counter()
    for t in texts:
        if t.get("kind", "greedy") == "greedy":
            k.add(aux=str(t.get("row_id", "")).startswith("covip:"), stage1=t.get("stage1"), stage2=t.get("stage2") or "")
    return Counter.rates(k.values())
