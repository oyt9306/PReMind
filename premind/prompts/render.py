"""Exact-five PDB context renderer (PROVISIONAL, CPU interface only).

Owner decisions D-3/OC-1/OC-2 and Addendum S3-F: every rendered PDB context has
exactly five *real* records, the header states the true count, and no filler is
ever rendered. Legacy behaviours deliberately NOT reproduced here:
  * realized K = min(5, store + 2) while the header always said "top-5";
  * RAG addendum "returned up to 5 items".
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

from premind.rewards.legacy.baseline_eval.prompts import RAG_ADDENDUM  # noqa: E402  (read-only legacy import)
from premind.prompts.final_prompts import STAGE1_SYSTEM_SECTION, SYSTEM_CORE_FINAL  # noqa: E402

logger = logging.getLogger(__name__)

RENDER_K = 5
assert RENDER_K == 5  # training must stay at K = 5 (PREREG L154); render_context has no k parameter
RENDERER_VERSION = "train_interface_v1.exact5_render_v1"  # context-render contract; pinned by WP-G contract_v1 GB2
PROMPT_VERSION = "train_interface_v1.final_prompt_v1"  # system prompts (owner PRE-TRAINING 2026-10-02)
# Owner S3-F: fillers unnecessary; DF-08 gates only if a padding path is re-opened by owner decision.
FILLER_PATH_ENABLED = False
# PREREG §7.1 "Evaluation-only K analysis": the grid is frozen now, before sealing, and is never
# selected from holdout results. Training is unaffected (training stays at K = 5 only, L154).
EVAL_K_GRID = (1, 3, 5)

_LEGACY_RAG_PHRASE = "returned up to 5 items below the query"
_EXACT_RAG_PHRASE = "returned exactly 5 items below the query"
if _LEGACY_RAG_PHRASE not in RAG_ADDENDUM:
    raise RuntimeError("baseline_eval RAG_ADDENDUM changed; re-audit the exact-five wording")
RAG_NOTE_EXACT5 = RAG_ADDENDUM.replace(_LEGACY_RAG_PHRASE, _EXACT_RAG_PHRASE)

# Memory-call section, verbatim from MEMORY_CALL_DECISION_PREREG §2.2 (iii).
MEMORY_CALL_SECTION = (
    "\n\nPersonal memory: a retriever can look up this user's saved personal memories (household members, "
    "health facts, preferences, belongings, routines). You will first decide whether this request needs them. "
    "Call memory only if the correct response depends on a personal fact about this user or a household member "
    "that is not visible in the image or stated in the request.")
STAGE1_INSTRUCTION = ("Step 1: decide whether you need this person's personal memory. You may reason briefly, "
                      "then END your reply with exactly one tag: <memory_call>call_pdb</memory_call> or "
                      "<memory_call>direct_answer</memory_call>.")  # owner 2026-10-02: reasoning allowed, 512-token budget
NO_MEMORY_NOTICE = "No personal memory was retrieved for this request."
FINAL_INSTRUCTION = ("Now give your final response in the required output format "
                     "(<decision>…</decision> then <answer>…</answer>), nothing before or after.")

# System prompts. Owner PRE-TRAINING decision 2026-10-02: the legacy V1_CORE (retrieve|crop, uncertain risk)
# is replaced by SYSTEM_CORE_FINAL (final_prompts.py); baseline_eval/prompts.py itself is unchanged.
SYSTEM_PASSIVE = SYSTEM_CORE_FINAL + RAG_NOTE_EXACT5
SYSTEM_MEMORY_CALL = SYSTEM_CORE_FINAL + MEMORY_CALL_SECTION + STAGE1_SYSTEM_SECTION

STATIC_TEMPLATES = (SYSTEM_PASSIVE, SYSTEM_MEMORY_CALL, RAG_NOTE_EXACT5, STAGE1_INSTRUCTION,
                    NO_MEMORY_NOTICE, FINAL_INSTRUCTION)


class RenderError(ValueError):
    """Raised when a context cannot be rendered under the exact-five contract."""


def header(n_entries: int) -> str:
    """Header text derived from the actual number of rendered entries."""
    return f"Retrieved personal context ({n_entries} entries):"


def _render_lines(entries: Sequence[Mapping]) -> list[str]:
    """Shared validation core for both the training and evaluation-only renderers."""
    lines = []
    for expected_rank, entry in enumerate(entries, 1):
        if entry.get("is_filler") is not False:
            raise RenderError("entry is a filler or has no explicit real-record status")
        rank, text = entry.get("rank"), entry.get("text")
        if type(rank) is not int or rank != expected_rank:
            raise RenderError(f"non-contiguous rank {rank!r} at position {expected_rank}")
        if not isinstance(text, str) or not text.strip() or "\n" in text:
            raise RenderError(f"invalid entry text at rank {expected_rank}")
        lines.append(f"[{rank}] {text}")
    return lines


def render_context(entries: Sequence[Mapping], *, allow_filler: bool = False) -> str:
    """Render exactly five real PDB entries as ``[rank] text`` lines under a truthful header.

    Training-only entry point: this signature intentionally has no ``k`` parameter, so a training
    caller cannot request K != 5 (PREREG L154) -- passing ``k`` raises ``TypeError``. The
    evaluation-only K sweep (PREREG §7.1) uses ``render_eval_context`` instead.

    Args:
        entries: Ordered entries, each with ``rank`` (int), ``text`` (str) and the evaluator-only
            ``is_filler`` flag (bool). Only rank and text reach the output.
        allow_filler: Explicit opt-in for a filler path. Refused while ``FILLER_PATH_ENABLED`` is False.

    Returns:
        Model-visible context block.

    Raises:
        RenderError: On any count, rank, text or filler violation.
    """
    if allow_filler and not FILLER_PATH_ENABLED:
        raise RenderError("filler path is closed (owner S3-F); no padding allowed")
    if len(entries) != RENDER_K:
        raise RenderError(f"exact-five violated: {len(entries)} entries")
    lines = _render_lines(entries)
    block = header(len(lines)) + "\n" + "\n".join(lines)
    assert block.count("\n[") == RENDER_K and header(RENDER_K) in block
    return block


def render_eval_context(entries: Sequence[Mapping], *, k: int = RENDER_K) -> str:
    """Evaluation-only top-K renderer (PREREG §7.1, owner 2026-10-01; secondary, descriptive).

    Renders exactly ``k`` real records as the top-``k`` of the same ranked ``entries`` list (the
    first ``k`` items, which must already carry contiguous ranks 1..k), under a truthful
    ``"(k entries)"`` header. No training code calls this function; it never changes training
    behaviour or ``RENDER_K``.

    Args:
        entries: Same shape as ``render_context``'s ``entries``, ranked 1..N with N >= k. When the
            entries carry the evaluator-only ``retriever_rank`` (adapter display order, owner
            2026-10-02), the top-``k`` are chosen by ``retriever_rank`` and shown in their display
            order (the row's label-independent permutation restricted to those ``k``).
        k: Evaluation-only K. Must be one of ``EVAL_K_GRID`` ``{1, 3, 5}`` -- the grid PREREG §7.1
            freezes before sealing; any other value raises. Never selected from holdout results.

    Returns:
        Model-visible context block with exactly ``k`` entries.

    Raises:
        RenderError: ``k`` not in ``EVAL_K_GRID``; the store/ranked list has fewer than ``k`` real
            records (fail-closed exclusion, not truncation-with-defaults); any rank/text/filler
            violation (same contract as ``render_context``).
    """
    if type(k) is not int or k not in EVAL_K_GRID:
        raise RenderError(f"evaluation-only k must be one of {EVAL_K_GRID}, got {k!r}")
    if len(entries) < k:
        raise RenderError(f"store has fewer than k={k} real records: {len(entries)} available")
    has_retriever_rank = ["retriever_rank" in e for e in entries]
    if any(has_retriever_rank):   # display-ordered entries (adapter.context_entries): top-k by retriever rank,
        if not all(has_retriever_rank) or sorted(e["retriever_rank"] for e in entries) != list(range(1, len(entries) + 1)):
            raise RenderError("retriever_rank must be present on every entry and form 1..N")
        top = [e for e in entries if e["retriever_rank"] <= k]   # kept in display order, display ranks 1..k
        entries = [{**e, "rank": pos} for pos, e in enumerate(top, 1)]
    lines = _render_lines(entries[:k])
    block = header(len(lines)) + "\n" + "\n".join(lines)
    assert block.count("\n[") == k and header(k) in block
    return block


def count_rendered_entries(text: str) -> int:
    """Count ``[n] `` entry lines in a rendered block (used by tests and the leak check)."""
    return sum(1 for line in text.splitlines() if line[:1] == "[" and "] " in line[:5])


def passive_user_text(query: str, context_block: str) -> str:
    """Passive-RAG user text: query then the exact-five block."""
    return f"User query: {query}\n\n{context_block}"


def stage1_user_text(query: str) -> str:
    """Memory-call Stage-1 user text: query and the decision instruction only."""
    return f"User query: {query}\n\n{STAGE1_INSTRUCTION}"


def stage2_user_text(context_block: str | None) -> str:
    """Memory-call Stage-2 user text: exact-five block iff called, else the fixed notice."""
    if context_block is None:
        return f"{NO_MEMORY_NOTICE}\n\n{FINAL_INSTRUCTION}"
    return f"{RAG_NOTE_EXACT5.strip()}\n\n{context_block}\n\n{FINAL_INSTRUCTION}"
