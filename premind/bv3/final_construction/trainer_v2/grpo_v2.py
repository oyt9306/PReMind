"""Single-process GRPO (+ privileged OPSD) loop for multi-image VCM episodes.

Why not TRL GRPOTrainer here: TRL 1.9.2 rebuilds ``pixel_values`` from the dataset prompt only (trainer_v1 trl_hook
notes L2373/L2444-2459), so the memory-call Stage-2 environment turn, which injects K memory IMAGES mid-completion,
cannot be scored by it. This loop owns the forward: every episode is (prompt ids + completion ids, all images in
sequence order), so passive (9 images in the prompt) and memory call (1 + 8 images) share one code path.

Math mirrors trainer_v1.train (TRL 1.9.2 loss_type "grpo"): grouped advantages (G contiguous episodes per prompt),
on-policy ratio with IS token_truncate (cap) against the sampler log-probs, per-sequence mean over model tokens
(env tokens masked), mean over the step's sequences. OPSD (rl_sweep/train_rl.py port): + lambda * truncated forward
KL(teacher || student) over the teacher's top-k at the student-visited prefixes, on gated episodes only; teacher =
same model with LoRA disabled, prompt = student prompt + privileged hint block. Micro-batch = 1 episode per
forward/backward; gradients accumulate over the whole step (P prompts x G episodes), then one optimizer step,
repeated ``updates_per_batch`` times on the same rollout (log pi_old fixed at update 0). ``algo`` = "grpo" (token
ratio, default) or "gspo" (sequence ratio); the surrogates live in policy_loss.py. ``batch_rollout`` (default off =
the sequential rollout below, unchanged) generates a training step's episodes in batched calls (batch_rollout.py).
``train_micro_batch`` (default 1 = the per-episode loop below, unchanged) groups episodes of similar length into
left-padded policy / teacher forwards with one backward per group (train_mb.py); the loss weighting is unchanged.
``algo`` "dapo" (clip-higher, dynamic sampling, token-level loss, overlong shaping) and the self-teacher OPSD modes
(``opsd_teacher`` / ``opsd_gate``) are in algo_v2.py; their defaults leave this loop unchanged. ``train_step(dist=...)``
(default None = single process, unchanged) is the torchrun data-parallel hook of ddp_v2.py. ``train_step(vef=...)``
(VEF reward presets only, trainer_v2/vef.py) judges each episode's <visual_evidence> before scoring.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from . import metrics_v9 as MV
from . import opsd as OP
from . import policy_loss as PL
from . import algo_v2 as AL
from . import reward_v5 as R5
from .records_v2 import component_means, episode_record, token_diagnostics, truncated  # noqa: F401 (re-exported)
from .vcm_data import VcmItem

import adapter as AD  # noqa: E402  (train_interface_v1 Stage-1 parser; import only)

logger = logging.getLogger(__name__)
Images = dict[str, Any]


@dataclass(frozen=True)
class Enc:
    """Token ids of a segment and the image paths whose placeholders it contains, in order."""
    ids: tuple[int, ...]
    images: tuple[str, ...] = ()


class Backend(Protocol):
    eos_token_id: int

    def encode_prompt(self, messages: list[dict], images: Images) -> Enc: ...

    def encode_env_turn(self, content: list[dict], images: Images, *, stage1_ended_with_eos: bool) -> Enc: ...

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:
        """``num_generations`` outputs per sequence, grouped by sequence in input order. ``gen_batch`` 0 = one call
        per sequence; > 0 = left-padded batched calls of at most ``gen_batch`` output rows (``images`` may then be
        one dict per sequence)."""
        ...

    def encode_text(self, text: str) -> list[int]: ...

    def logits_at(self, seq: Enc, positions: Sequence[int], images: Images, *, teacher: bool) -> Any: ...

    def logits_at_batch(self, seqs: Sequence[Enc], positions: Sequence[Sequence[int]], images: Sequence[Images], *,
                        teacher: bool) -> list[Any]:
        """``logits_at`` of several sequences in one padded forward (row b: positions[b] of seqs[b]); only used
        when ``LoopConfig.train_micro_batch`` > 1."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class LoopConfig:
    num_generations: int = 8
    max_completion_length: int = 512
    stage1_cap: int = 512
    learning_rate: float = 1e-5
    epsilon: float = 0.2
    is_cap: float = 3.0
    scale_rewards: str = "group"
    max_grad_norm: float = 1.0
    r: Optional[float] = None               # memory-call decision magnitude (DF-01; required for that arm)
    reward_preset: str = "main"             # reward_v5.PRESETS key ("P-3" = citation terms off)
    opsd_lambda: float = 0.0
    opsd_only: bool = False
    opsd_topk: int = OP.TOPK
    algo: str = "grpo"                      # policy_loss.ALGOS: "grpo" (token ratio) | "gspo" (seq.) | "dapo"
    updates_per_batch: int = 1              # mu: optimizer updates per rollout batch (log pi_old fixed at update 0)
    clip_eps_low: Optional[float] = None    # None -> grpo: epsilon; gspo: 3e-4 (policy_loss.clip_bounds)
    clip_eps_high: Optional[float] = None   # None -> grpo: epsilon; gspo: 4e-4
    batch_rollout: bool = False             # train_step: batched Stage 1 / Stage 2 / passive generation
    gen_batch: int = 16                     # batched rollout: max output rows per generate call (memory bound)
    train_micro_batch: int = 1              # episodes per policy / teacher forward (1 = per-episode loop, unchanged)
    dapo_dynamic_sampling: bool = True      # algo=dapo: drop zero-variance groups, refill (algo_v2.dynamic_rollout)
    dapo_max_prompts_factor: float = 3.0    # algo=dapo: at most ceil(factor * P) prompts drawn per step
    overlong_buffer: int = 128              # algo=dapo: soft overlong shaping over the last N tokens before the cap
    overlong_penalty: float = 1.0           # algo=dapo: shaping at the cap (0 = off)
    opsd_teacher: str = "frozen"            # "frozen" (base, adapter off) | "self" (current model, no grad)
    opsd_gate: Optional[str] = None         # "table" | "hint"; None -> table (frozen) / hint (self)
    opsd_only_failed: bool = False          # hint gate: only episodes failing a Stage-2 check

    def __post_init__(self) -> None:
        if self.algo not in PL.ALGOS or self.updates_per_batch < 1:
            raise ValueError(f"algo must be in {PL.ALGOS} and updates_per_batch >= 1")
        if self.gen_batch < 1 or self.train_micro_batch < 1:
            raise ValueError("gen_batch and train_micro_batch must be >= 1")
        AL.validate(self)


@dataclass
class Episode:
    item: VcmItem
    prompt: Enc
    completion: tuple[int, ...]
    env_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    images: tuple[str, ...]                 # prompt images + env images, in sequence order
    stage1_text: Optional[str]
    stage2_text: str
    n_stage2: int                           # Stage-2 model tokens excluding a final EOS (degenerate gate)
    decision: Optional[str] = None
    ledger: dict = field(default_factory=dict)
    n_stage1: int = 0                       # memory call: Stage-1 tokens at the start of the completion
    n_env: int = 0                          # memory call: env tokens after them (masked)
    vef: Optional[dict] = None              # VEF presets: prefetched judge input (vef.VefContext.prefetch)


TeacherCtx = OP.TeacherCtx


def _strip_eos(ids: Sequence[int], eos: int) -> list[int]:
    return list(ids[:-1]) if ids and ids[-1] == eos else list(ids)


def _lp(lp: Optional[list[float]], ids: Sequence[int]) -> tuple[float, ...]:
    return tuple(lp) if lp is not None else (math.nan,) * len(ids)


def rollout(items: list[VcmItem], images: list[Images], backend: Backend, cfg: LoopConfig, *,
            greedy: bool = False, num_generations: Optional[int] = None, teacher: bool = False,
            ctx_fn: Optional[Callable[[VcmItem], TeacherCtx]] = None,
            forced_stage1: Optional[Callable[[VcmItem], str]] = None,
            temperature: Optional[float] = None, batched: bool = False) -> list[Episode]:
    """G episodes per item (contiguous). Teacher table: ``ctx_fn`` swaps in the hinted context, ``teacher`` disables
    LoRA, ``forced_stage1`` replaces Stage-1 sampling by a fixed decision text (the Stage-1 turn is never hinted).
    Sampler log-probs a backend does not return are NaN (the loss then uses the on-policy log-probs: IS ratio 1).
    ``batched``: same episodes, same order, generated in batched calls (batch_rollout.rollout_batched)."""
    if batched:
        from .batch_rollout import rollout_batched
        return rollout_batched(items, images, backend, cfg, greedy=greedy, num_generations=num_generations,
                               teacher=teacher, ctx_fn=ctx_fn, forced_stage1=forced_stage1, temperature=temperature)
    g = num_generations or cfg.num_generations
    gen = dict(greedy=greedy, teacher=teacher, temperature=temperature)
    eos = backend.eos_token_id
    out: list[Episode] = []
    for it, imgs in zip(items, images, strict=True):
        ctx = ctx_fn(it) if ctx_fn else TeacherCtx(it.messages, it.env_call, it.env_no_call)
        p = backend.encode_prompt(ctx.messages, imgs)
        if it.arm == "passive":
            comps, lps = backend.generate([p], imgs, num_generations=g, max_tokens=cfg.max_completion_length, **gen)
            for c, lp in zip(comps, lps, strict=True):
                body = _strip_eos(c, eos)
                out.append(Episode(it, p, tuple(c), (1,) * len(c), _lp(lp, c), p.images, None, backend.decode(body),
                                   len(body)))
            continue
        if forced_stage1 is not None:
            s1, lp1 = [list(backend.encode_text(forced_stage1(it))) + [eos]] * g, [None] * g
        else:
            s1, lp1 = backend.generate([p], imgs, num_generations=g, max_tokens=cfg.stage1_cap, **gen)
        for c1, l1 in zip(s1, lp1, strict=True):
            text1 = backend.decode(_strip_eos(c1, eos))
            decision = AD.parse_stage1(text1)
            content = ctx.env_call if decision == "call_pdb" else ctx.env_no_call
            env = backend.encode_env_turn(content, imgs, stage1_ended_with_eos=bool(c1) and c1[-1] == eos)
            seq2 = Enc(p.ids + tuple(c1) + env.ids, p.images + env.images)
            (c2,), (l2,) = backend.generate([seq2], imgs, num_generations=1, max_tokens=cfg.max_completion_length,
                                            **gen)
            body = _strip_eos(c2, eos)
            out.append(Episode(it, p, tuple(c1) + env.ids + tuple(c2),
                               (1,) * len(c1) + (0,) * len(env.ids) + (1,) * len(c2),
                               _lp(l1, c1) + (0.0,) * len(env.ids) + _lp(l2, c2), seq2.images, text1,
                               backend.decode(body), len(body), decision, n_stage1=len(c1), n_env=len(env.ids)))
    return out


def score(ep: Episode, cfg: LoopConfig) -> dict:
    flags = R5.PRESETS[cfg.reward_preset]
    kw = {"vef": ep.vef} if flags.vef else {}
    if flags.withhold_match:                     # v52ws_ag: the parsed Match part of <visual_evidence> (no judge)
        from .withhold_ag import episode_match
        kw["match"] = episode_match(ep.item, ep.stage2_text)
    if ep.item.arm == "passive":
        return R5.reward_passive(ep.item.target, ep.stage2_text, ep.n_stage2, flags=flags, **kw)
    return R5.reward_call(ep.item.target, ep.stage1_text, ep.stage2_text, ep.n_stage2, r=cfg.r, flags=flags, **kw)


def grouped_advantages(rewards: Sequence[float], g: int, scale_rewards: str = "group"):
    """TRL grpo_trainer L2685-2707: mean-centred per group of ``g`` contiguous episodes, optionally / (std + 1e-4)."""
    import torch
    r = torch.as_tensor(list(rewards), dtype=torch.float32)
    if r.numel() % g:
        raise ValueError("rewards must hold whole groups")
    grp = r.view(-1, g)
    adv = (grp - grp.mean(1, keepdim=True)).view(-1)
    if scale_rewards == "group":
        adv = adv / (grp.std(1, keepdim=True).expand_as(grp).reshape(-1) + 1e-4)
    elif scale_rewards != "none":
        raise ValueError("scale_rewards must be 'group' or 'none'")
    return adv


def model_positions(ep: Episode) -> list[int]:
    return [i for i, m in enumerate(ep.env_mask) if m]


def distill_positions(ep: Episode) -> list[int]:
    """Indices into ``model_positions(ep)`` that are distilled: every token (passive), Stage-2 tokens (memory call)."""
    pos = model_positions(ep)
    if ep.item.arm == "passive":
        return list(range(len(pos)))
    return [j for j, i in enumerate(pos) if i >= ep.n_stage1 + ep.n_env]


def episode_loss(ep: Episode, adv: float, backend: Backend, images: Images, cfg: LoopConfig,
                 teacher: Optional[tuple[Any, Any, list[int]]] = None) -> tuple[Any, dict]:
    """Per-episode loss (before / N): GRPO token mean (or GSPO sequence term) over model tokens [+ lambda * truncated
    KL on the distilled positions if gated]. ``teacher`` = (top-k ids, top-k log-probs, indices into model positions)."""
    loss, stats, _ = episode_loss_old(ep, adv, backend, images, cfg, teacher)
    return loss, stats


def episode_loss_old(ep: Episode, adv: float, backend: Backend, images: Images, cfg: LoopConfig,
                     teacher: Optional[tuple[Any, Any, list[int]]] = None, old_logp: Any = None,
                     rl_weight: Optional[float] = None) -> tuple[Any, dict, Any]:
    """``episode_loss`` with log pi_old made explicit: ``old_logp`` None = this forward's detached log-probs (on-policy
    update 0); otherwise the tensor returned by update 0 (updates 1..mu-1 reuse the same rollout batch)."""
    seq, at = policy_inputs(ep)
    logits = backend.logits_at(seq, at, images, teacher=False).float()
    return loss_from_logits(ep, adv, logits, cfg, teacher, old_logp, rl_weight)


def policy_inputs(ep: Episode) -> tuple[Enc, list[int]]:
    """Training sequence (prompt + completion, all images) and the positions of its model tokens."""
    return Enc(ep.prompt.ids + ep.completion, ep.images), [len(ep.prompt.ids) + i for i in model_positions(ep)]


def loss_from_logits(ep: Episode, adv: float, logits: Any, cfg: LoopConfig,
                     teacher: Optional[tuple[Any, Any, list[int]]] = None, old_logp: Any = None,
                     rl_weight: Optional[float] = None) -> tuple[Any, dict, Any]:
    """``episode_loss_old`` after the forward: ``logits`` (float) at ``policy_inputs(ep)[1]``, one row per model
    token. Shared by the per-episode loop and the micro-batched one (train_mb), so both use the same loss."""
    import torch
    pos = model_positions(ep)
    logp_all = torch.log_softmax(logits, -1)
    tgt = torch.tensor([ep.completion[i] for i in pos], device=logits.device)
    logp = logp_all.gather(-1, tgt[:, None])[:, 0]
    old = logp.detach() if old_logp is None else old_logp.to(logp.device)
    samp = torch.tensor([ep.logprobs[i] for i in pos], dtype=torch.float32, device=logits.device)
    samp = torch.where(torch.isnan(samp), old, samp)
    is_ratio = torch.exp(old - samp).clamp(max=cfg.is_cap)
    a = torch.tensor(float(adv), device=logits.device)
    seq_diag: dict = {}
    if cfg.algo == "gspo" and pos:
        rl, s, is_seq = PL.gspo_term(logp, old, samp, a, cfg)
        coef1, seq_diag = torch.exp(logp - old), PL.sequence_diagnostics(s, is_seq, cfg)
    else:
        rl, coef1 = PL.grpo_term(logp, old, is_ratio, a, cfg)
        rl = rl if pos else logits.sum() * 0.0
    stats = {"n_model_tokens": len(pos), "rl": float(rl.detach()), "is_ratio_mean": float(is_ratio.mean())
             if pos else None, "kl": None, "n_distill_tokens": 0, **token_diagnostics(logp_all, coef1, is_ratio, cfg),
             "ratio_mean": float(coef1.detach().mean()) if pos else None, **seq_diag}
    loss = rl * 0.0 if cfg.opsd_only else rl
    loss = loss if rl_weight is None else loss * rl_weight          # DAPO token-level aggregation (algo_v2)
    if teacher is not None and teacher[2]:
        t_ids, t_lps, sel = teacher
        idx = torch.tensor(sel, dtype=torch.long, device=logits.device)
        s_lp = logp_all[idx].gather(-1, t_ids.to(logits.device))
        kl = OP.truncated_forward_kl(t_lps.to(logits.device), s_lp)
        loss = loss + cfg.opsd_lambda * kl
        stats.update(kl=float(kl.detach()), n_distill_tokens=len(sel))
    return loss, stats, old


def teacher_topk(ep: Episode, ctx: TeacherCtx, backend: Backend, images: Images, k: int, adapter_off: bool = True):
    """Teacher (LoRA disabled; ``adapter_off`` False = self teacher, adapter on, still no grad) top-k ids / log-probs at the student's distilled prefixes. Passive: hinted prompt +
    the student completion. Memory call: student prompt + student Stage 1 + HINTED env turn + student Stage 2."""
    import torch
    seq, at, sel = teacher_inputs(ep, ctx, backend, images)
    with torch.no_grad():
        logits = backend.logits_at(seq, at, images, teacher=adapter_off).float()
        v, i = torch.log_softmax(logits, -1).topk(k, dim=-1)
    return i, v, sel


def teacher_inputs(ep: Episode, ctx: TeacherCtx, backend: Backend, images: Images) -> tuple[Enc, list[int], list[int]]:
    """Teacher sequence, positions of the distilled tokens in it, and their indices into model positions."""
    pos, sel = model_positions(ep), distill_positions(ep)
    if ep.item.arm == "passive":
        tp = backend.encode_prompt(ctx.messages, images)
        seq = Enc(tp.ids + ep.completion, tp.images + ep.images[len(ep.prompt.images):])
        at = [len(tp.ids) + pos[j] for j in sel]
    else:
        c1 = ep.completion[:ep.n_stage1]
        content = ctx.env_call if ep.decision == "call_pdb" else ctx.env_no_call
        env = backend.encode_env_turn(content, images,
                                      stage1_ended_with_eos=bool(c1) and c1[-1] == backend.eos_token_id)
        c2 = ep.completion[ep.n_stage1 + ep.n_env:]
        seq = Enc(ep.prompt.ids + c1 + env.ids + c2, ep.prompt.images + env.images)
        base = len(ep.prompt.ids) + len(c1) + len(env.ids)
        at = [base + pos[j] - ep.n_stage1 - ep.n_env for j in sel]
    return seq, at, sel


def train_step(items: list[VcmItem], images: list[Images], backend: Backend, optimizer: Any, params: list,
               cfg: LoopConfig, *, opsd: Optional["OP.OpsdContext"] = None,
               on_episode: Optional[Callable[[Episode, dict], None]] = None,
               refill: Optional[AL.Refill] = None, dist: Optional[Any] = None, step: Optional[int] = None,
               vef: Optional[Any] = None) -> dict:
    """Rollout -> reward v5 -> advantages -> [OPSD gate + teacher top-k] -> accumulated loss -> optimizer step,
    repeated ``cfg.updates_per_batch`` (mu) times on the same rollout batch. Update 0 is the on-policy step (ratio 1);
    its detached log-probs are log pi_old for updates 1..mu-1, and the teacher top-k (frozen base) is reused. Episode
    records carry the LAST update's loss stats (clip_frac / ratio_mean of the moved policy); ``grad_norm`` is update
    0's (comparable across mu), ``grad_norms`` lists every update when mu > 1. DAPO dynamic sampling draws extra
    prompts through ``refill(k) -> (items, images)``; a self teacher is recomputed at every update (algo_v2).
    ``dist`` (ddp_v2.DistCtx): this rank's episodes only; LoRA grads are all-reduced to the global-batch gradient
    before clipping, DAPO token weights use global totals, and the returned record is merged over ranks.
    ``vef`` (vef.VefContext; required by and only allowed with a VEF preset): every rollout round is judged
    (``prefetch(eps, step)``, QAs drawn for the GLOBAL ``step``) before it is scored; ``vef_raw`` joins the record."""
    import time

    import torch
    t0 = time.time()
    extra: dict = {}
    if bool(R5.PRESETS[cfg.reward_preset].vef) != (vef is not None):
        raise ValueError(f"reward preset {cfg.reward_preset!r} and the VEF context must come together")
    prefetch = None if vef is None else (lambda round_eps: vef.prefetch(round_eps, step))
    if cfg.algo == "dapo" and cfg.dapo_dynamic_sampling:
        eps, items, images, extra = AL.dynamic_rollout(items, images, backend, cfg, refill, rollout, score,
                                                       **({} if prefetch is None else {"prefetch": prefetch}))
        extra["used_items"] = items
    else:
        eps = rollout(items, images, backend, cfg, batched=cfg.batch_rollout)
        if prefetch is not None:
            prefetch(eps)
    t1 = time.time()
    for ep in eps:
        ep.ledger = score(ep, cfg) if not extra else ep.ledger
    rewards = [ep.ledger["total"] for ep in eps]
    adv = grouped_advantages(AL.shaped_rewards(eps, cfg), cfg.num_generations, cfg.scale_rewards)
    if cfg.algo == "dapo":
        extra["overlong_penalty_mean"] = (sum(AL.shaped_rewards(eps, cfg)) - sum(rewards)) / len(eps)
    by_item = {id(it): im for it, im in zip(items, images)}
    gates = [AL.opsd_gate(opsd, ep, cfg, distill_positions) for ep in eps]
    totals = None if dist is None else dist.step_totals(eps)              # (global episodes, global model tokens)
    n, mu, rlw = len(eps), cfg.updates_per_batch, AL.rl_weights(eps, cfg, totals)
    olds: list[Any] = [None] * n
    teachers: list[Any] = [None] * n
    grad_norms: list[float] = []
    self_t, t_teacher = cfg.opsd_teacher == "self", 0.0
    for u in range(mu):
        optimizer.zero_grad(set_to_none=True)
        teachers = [None] * n if self_t else teachers          # self teacher: recomputed with the current model
        if cfg.train_micro_batch > 1:                    # grouped forwards, one backward per group (train_mb.py)
            from .train_mb import accumulate
            stats, dt = accumulate(eps, adv.tolist(), gates, [by_item[id(ep.item)] for ep in eps], backend, cfg,
                                   opsd, teachers, olds, u, rlw)
            t_teacher += dt
        else:
            stats = []
            for k, (ep, a, (gated, _)) in enumerate(zip(eps, adv.tolist(), gates, strict=True)):
                imgs = by_item[id(ep.item)]
                if gated and teachers[k] is None:
                    tt = time.time()
                    teachers[k] = teacher_topk(ep, opsd.teacher_context(ep.item), backend, imgs, cfg.opsd_topk,
                                               adapter_off=not self_t)
                    t_teacher += time.time() - tt
                loss, st, old = episode_loss_old(ep, a, backend, imgs, cfg, teachers[k], olds[k],
                                                 **({} if rlw is None else {"rl_weight": rlw[k]}))
                olds[k] = old if u == 0 and mu > 1 else olds[k]
                (loss / n).backward()
                stats.append(st)
        per_ep = [episode_record(ep, a, gated, why, st, cfg, backend.eos_token_id)
                  for ep, a, (gated, why), st in zip(eps, adv.tolist(), gates, stats, strict=True)]
        if on_episode is not None and u == mu - 1:
            for ep, rec in zip(eps, per_ep):
                on_episode(ep, rec)
        if dist is not None:
            dist.reduce_grads(params, n, totals[0])                            # = sum_all grad(L_k) / N_global
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    out = {"n_episodes": n, "reward_mean": sum(rewards) / n, "rewards": rewards, "t_rollout_s": round(t1 - t0, 2),
            "records": [MV.episode_record(ep.item, ep.ledger, decision=ep.decision) for ep in eps],
            "components": component_means([ep.ledger for ep in eps]),
            "t_train_s": round(time.time() - t1, 2),
            "grad_norm": grad_norms[0], "opsd_gate_rate": sum(g for g, _ in gates) / n, "episodes": per_ep,
            **({"grad_norms": grad_norms, "updates_per_batch": mu} if mu > 1 else {}), **extra,
            **({"opsd_self_teacher": 1, "t_teacher_s": round(t_teacher, 2)} if self_t and opsd else {}),
            **({} if vef is None else {"vef_raw": vef.raw(eps)})}
    return out if dist is None else dist.merge_step(out, eps, items)
