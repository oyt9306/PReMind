"""Micro-batched policy / teacher forwards for grpo_v2.train_step (``LoopConfig.train_micro_batch`` > 1; default 1 =
the per-episode loop in grpo_v2, unchanged).

One update over the step's N = P x G episodes:

1. OPSD teacher top-k (no grad; update 0, or every update for a self teacher) for the gated episodes whose top-k
   is not cached yet: teacher sequences
   grouped by length, ``train_micro_batch`` rows per ``backend.logits_at_batch(..., teacher=True)`` call.
2. Policy: episodes grouped by sequence length (prompt + completion, longest first, ties by episode index), one
   ``logits_at_batch`` forward per group; each row's logits go through ``grpo_v2.loss_from_logits`` (the same loss
   code as the per-episode loop); the group's backward is that of sum_k loss_k / N, so the gradient of the update is
   sum_k grad(loss_k) / N exactly as before (only the floating-point summation order differs).
3. ``old`` (log pi_old, update 0 when mu > 1) and the teacher top-k are stored per episode index, as in grpo_v2.

Padding (HFBackend.logits_at_batch): LEFT padding through ``HFBackend.batch_inputs``, the layout the batched rollout
already uses: Qwen3.5 computes M-RoPE positions over attention-mask-1 tokens only (get_rope_index), so each row gets
the positions of its unpadded forward; its linear-attention (Gated DeltaNet) layers zero the pad states
(apply_mask_to_padding_states, "left-padding is used for linear attention mask"), which equals the zero initial
conv / recurrent state of the unpadded forward; full-attention layers mask pad keys. Right padding would also leave
causal rows untouched, but HF documents left padding for the linear-attention mask and the rollout path is already
validated on it, so one layout serves both. Logits are taken per row at exactly the unpadded model-token positions
(shifted by the row's pad width) from the final hidden states: no logits for pad, env or prompt positions.
Lengths are sorted so a group pads to its longest row (step-level padding waste is typically a few percent).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from . import grpo_v2 as G

logger = logging.getLogger(__name__)


def length_groups(lengths: Sequence[int], size: int) -> list[list[int]]:
    """Indices grouped ``size`` at a time after sorting by length (descending, ties by index): similar-length rows
    share a padded forward. Every index appears exactly once."""
    if size < 1:
        raise ValueError("micro-batch size must be >= 1")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[j:j + size] for j in range(0, len(order), size)]


def padding_waste(lengths: Sequence[int], groups: list[list[int]]) -> float:
    """Share of padded token slots over all forwarded slots (logging / smoke)."""
    total = sum(max(lengths[i] for i in g) * len(g) for g in groups)
    return 1.0 - sum(lengths) / total if total else 0.0


def _free_cuda() -> None:
    """Return the micro-batch's cached activations / logits blocks to the allocator between groups (GPU only)."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def teacher_topk_batched(eps: list[G.Episode], need: list[int], ctxs: dict[int, G.TeacherCtx], images: list[G.Images],
                         backend: G.Backend, k: int, size: int, adapter_off: bool = True
                         ) -> dict[int, tuple[Any, Any, list[int]]]:
    """``grpo_v2.teacher_topk`` for the episodes ``need`` in padded groups of ``size``; {episode index: top-k}.
    ``adapter_off`` False = self teacher (current LoRA model, still no grad)."""
    import torch
    ins = {i: G.teacher_inputs(eps[i], ctxs[i], backend, images[i]) for i in need}
    out: dict[int, tuple[Any, Any, list[int]]] = {}
    for grp in length_groups([len(ins[i][0].ids) for i in need], size):
        idx = [need[j] for j in grp]
        with torch.no_grad():
            rows = backend.logits_at_batch([ins[i][0] for i in idx], [ins[i][1] for i in idx],
                                           [images[i] for i in idx], teacher=adapter_off)
            for i, lg in zip(idx, rows, strict=True):
                v, t = torch.log_softmax(lg.float(), -1).topk(k, dim=-1)
                out[i] = (t, v, ins[i][2])
        del rows
        _free_cuda()
    return out


def accumulate(eps: list[G.Episode], advs: list[float], gates: list[tuple[bool, list]], images: list[G.Images],
               backend: G.Backend, cfg: G.LoopConfig, opsd: Optional[Any], teachers: list[Any], olds: list[Any],
               update: int, rl_weights: Optional[list[float]] = None) -> tuple[list[dict], float]:
    """Backward of sum_k loss_k / N over the episodes in length-grouped micro-batches; fills ``teachers`` (missing
    entries) and ``olds`` (update 0, mu > 1) in place like the per-episode loop; ``rl_weights`` = DAPO token-level
    weights (algo_v2.rl_weights). Returns (loss stats in episode order, teacher forward seconds)."""
    import time
    if not hasattr(backend, "logits_at_batch"):
        raise TypeError(f"{type(backend).__name__} has no logits_at_batch: use train_micro_batch=1")
    n, size = len(eps), cfg.train_micro_batch
    need = [k for k in range(n) if gates[k][0] and teachers[k] is None]
    t0 = time.time()
    if need:
        ctxs = {k: opsd.teacher_context(eps[k].item) for k in need}
        for k, top in teacher_topk_batched(eps, need, ctxs, images, backend, cfg.opsd_topk, size,
                                           adapter_off=cfg.opsd_teacher != "self").items():
            teachers[k] = top
    t_teacher = time.time() - t0
    stats: list[Optional[dict]] = [None] * n
    lengths = [len(ep.prompt.ids) + len(ep.completion) for ep in eps]
    groups = length_groups(lengths, size)
    for grp in groups:
        ins = [G.policy_inputs(eps[k]) for k in grp]
        rows = backend.logits_at_batch([s for s, _ in ins], [at for _, at in ins], [images[k] for k in grp],
                                       teacher=False)
        total = None
        for k, lg in zip(grp, rows, strict=True):
            loss, st, old = G.loss_from_logits(eps[k], advs[k], lg.float(), cfg, teachers[k], olds[k],
                                               None if rl_weights is None else rl_weights[k])
            olds[k] = old if update == 0 and cfg.updates_per_batch > 1 else olds[k]
            stats[k] = st
            total = loss / n if total is None else total + loss / n
        total.backward()
        del rows, total, loss
        _free_cuda()
    logger.debug("train micro-batch %d: %d groups, padding waste %.3f", size, len(groups),
                 padding_waste(lengths, groups))
    return stats, t_teacher  # type: ignore[return-value]
