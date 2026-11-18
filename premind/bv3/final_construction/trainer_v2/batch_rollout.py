"""Batched rollout for grpo_v2 (``LoopConfig.batch_rollout``; default off).

The sequential ``grpo_v2.rollout`` calls ``backend.generate`` once per prompt (passive, Stage 1 with
num_return_sequences = G) and once per memory-call EPISODE for Stage 2. Here a step makes at most three kinds of
batched calls, each split into micro-batches of at most ``cfg.gen_batch`` output rows (``chunk_spans``):

1. passive prompts of all P items (G returns each);
2. Stage 1 of all memory-call prompts (G returns each), unless ``forced_stage1``;
3. Stage 2 of all P x G memory-call episodes (1 return each), each with its own prompt + Stage 1 + env turn and its
   own image list (prompt image + the K retrieved memory images).

Episodes come back in exactly the sequential order (item by item, G contiguous per item), with the same fields:
completion = Stage 1 + env + Stage 2 ids, env_mask 0 on env tokens, sampler log-probs (NaN when the backend has none)
and 0.0 on env tokens. Only the generation calls are regrouped; teacher / OPSD forwards and the loss are untouched.

Sampling: every output row is drawn from the same per-sequence distribution as in the sequential path (rows of a
left-padded batch are independent given their attention mask; HF keeps finished rows inert). What differs is the
order in which the sampler's RNG is consumed (all Stage-1 draws before any Stage-2 draw, rows sampled in parallel),
so a sampled rollout is NOT bit-identical to the sequential one at the same seed; with greedy decoding on a
deterministic backend it is. On GPU, bf16 kernels on padded batches may also differ from unpadded ones in the last
bits, so greedy parity there is checked statistically (eval_v1/batch_smoke.sh), not bit-for-bit.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from . import grpo_v2 as G
from .vcm_data import VcmItem

logger = logging.getLogger(__name__)


def chunk_spans(n_seqs: int, per_seq: int, gen_batch: int) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) spans over ``n_seqs`` input sequences so that each call yields at most ``gen_batch``
    output rows (``per_seq`` = num_return_sequences); a sequence is never split, so a span holds >= 1 sequence."""
    if n_seqs < 0 or per_seq < 1 or gen_batch < 1:
        raise ValueError("chunk_spans needs n_seqs >= 0, per_seq >= 1, gen_batch >= 1")
    k = max(1, gen_batch // per_seq)
    return [(lo, min(lo + k, n_seqs)) for lo in range(0, n_seqs, k)]


def _passive_episodes(it: VcmItem, p: G.Enc, comps: list, lps: list, backend: G.Backend) -> list[G.Episode]:
    eos = backend.eos_token_id
    out = []
    for c, lp in zip(comps, lps, strict=True):
        body = G._strip_eos(c, eos)
        out.append(G.Episode(it, p, tuple(c), (1,) * len(c), G._lp(lp, c), p.images, None, backend.decode(body),
                             len(body)))
    return out


def rollout_batched(items: list[VcmItem], images: list[G.Images], backend: G.Backend, cfg: G.LoopConfig, *,
                    greedy: bool = False, num_generations: Optional[int] = None, teacher: bool = False,
                    ctx_fn: Optional[Callable[[VcmItem], G.TeacherCtx]] = None,
                    forced_stage1: Optional[Callable[[VcmItem], str]] = None,
                    temperature: Optional[float] = None) -> list[G.Episode]:
    """``grpo_v2.rollout`` semantics (same arguments, same episode order and fields) with batched generation."""
    g = num_generations or cfg.num_generations
    gen = dict(greedy=greedy, teacher=teacher, temperature=temperature, gen_batch=cfg.gen_batch)
    eos = backend.eos_token_id
    if len(items) != len(images):
        raise ValueError("items and images must align")
    ctxs = [ctx_fn(it) if ctx_fn else G.TeacherCtx(it.messages, it.env_call, it.env_no_call) for it in items]
    prompts = [backend.encode_prompt(c.messages, im) for c, im in zip(ctxs, images)]
    per_item: dict[int, list[G.Episode]] = {i: [] for i in range(len(items))}
    passive = [i for i, it in enumerate(items) if it.arm == "passive"]
    call = [i for i, it in enumerate(items) if it.arm != "passive"]
    if passive:
        comps, lps = backend.generate([prompts[i] for i in passive], [images[i] for i in passive], num_generations=g,
                                      max_tokens=cfg.max_completion_length, **gen)
        for j, i in enumerate(passive):
            per_item[i] = _passive_episodes(items[i], prompts[i], comps[j * g:(j + 1) * g], lps[j * g:(j + 1) * g],
                                            backend)
    if not call:
        return [ep for i in range(len(items)) for ep in per_item[i]]
    if forced_stage1 is not None:
        s1 = {i: ([list(backend.encode_text(forced_stage1(items[i]))) + [eos]] * g, [None] * g) for i in call}
    else:
        c1s, l1s = backend.generate([prompts[i] for i in call], [images[i] for i in call], num_generations=g,
                                    max_tokens=cfg.stage1_cap, **gen)
        s1 = {i: (c1s[j * g:(j + 1) * g], l1s[j * g:(j + 1) * g]) for j, i in enumerate(call)}
    pending = []                     # (item index, c1, l1, Stage-1 text, decision, env turn, Stage-2 input)
    for i in call:
        p, ctx = prompts[i], ctxs[i]
        for c1, l1 in zip(*s1[i], strict=True):
            text1 = backend.decode(G._strip_eos(c1, eos))
            decision = G.AD.parse_stage1(text1)
            content = ctx.env_call if decision == "call_pdb" else ctx.env_no_call
            env = backend.encode_env_turn(content, images[i], stage1_ended_with_eos=bool(c1) and c1[-1] == eos)
            pending.append((i, c1, l1, text1, decision, env, G.Enc(p.ids + tuple(c1) + env.ids, p.images + env.images)))
    c2s, l2s = backend.generate([x[6] for x in pending], [images[x[0]] for x in pending], num_generations=1,
                                max_tokens=cfg.max_completion_length, **gen)
    for (i, c1, l1, text1, decision, env, seq2), c2, l2 in zip(pending, c2s, l2s, strict=True):
        body = G._strip_eos(c2, eos)
        per_item[i].append(G.Episode(items[i], prompts[i], tuple(c1) + env.ids + tuple(c2),
                                     (1,) * len(c1) + (0,) * len(env.ids) + (1,) * len(c2),
                                     G._lp(l1, c1) + (0.0,) * len(env.ids) + G._lp(l2, c2), seq2.images, text1,
                                     backend.decode(body), len(body), decision, n_stage1=len(c1), n_env=len(env.ids)))
    logger.debug("batched rollout: %d passive + %d memory-call prompts, %d Stage-2 rows, gen_batch=%d",
                 len(passive), len(call), len(pending), cfg.gen_batch)
    return [ep for i in range(len(items)) for ep in per_item[i]]
