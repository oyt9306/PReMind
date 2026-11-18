"""Data-parallel (torchrun) support for grpo_v2.train_step: rank context, LoRA-gradient all-reduce, step-record merge
and the rank-partitioned prompt stream. The launcher is eval_v1/ddp_main.py (rlvr_main_ddp.sh); with ``dist=None``
(every single-GPU entry point) none of this runs.

Gradient sync: manual all-reduce rather than ``torch.nn.parallel.DistributedDataParallel``. The custom loop runs a
rank-dependent number of backward calls per optimizer update (one per episode, or per length group with
train_micro_batch > 1; DAPO dynamic sampling keeps a rank-dependent number of groups), interleaves no-grad teacher
forwards and adapter-off generation through the same module, and repeats mu updates on one rollout. DDP's
per-backward bucket hooks would need ``no_sync()`` on all but the last backward of every update, identical backward
counts on every rank (else a hang), and ``find_unused_parameters`` + static-graph care with non-reentrant gradient
checkpointing; wrapping also changes the PEFT module paths used by save_pretrained / set_peft_model_state_dict.
Instead each rank accumulates exactly as the single-process loop (backward of loss_k / n_local), then, once per
update and before gradient clipping, ``reduce_grads`` scales by n_local / N_global and SUM-all-reduces one flat
buffer of the LoRA grads: every rank then holds sum over ALL episodes of grad(loss_k) / N_global, the gradient of
the single-process loop on the global batch (up to fp summation order), and clips / steps identically, so the
replicas stay equal (checked by ``sync_check`` at checkpoints; rank 0's weights are re-broadcast on a mismatch).
DAPO token-level weights use the global (episodes, model tokens) totals (``step_totals``) so the RL term is the
global token mean; the OPSD KL keeps its 1 / N_global weight.

Group advantages are rank-local (each prompt's G episodes are generated on one rank). Records / episodes / rewards
are gathered so every rank's step record is the global one (rank order = global prompt order), so the over-abstain
monitor sees identical records on every rank.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional, Sequence

from .records_v2 import component_means

logger = logging.getLogger(__name__)
RANK_SEED_STRIDE = 1_000_003          # rank r > 0 samples with seed + r * stride (rank 0 keeps the run seed)
SUM_KEYS = ("dapo_filtered_groups", "dapo_prompts_drawn", "dapo_kept_groups")
MAX_KEYS = ("t_rollout_s", "t_train_s", "dapo_resample_rounds", "t_teacher_s")


@dataclass(frozen=True)
class DistCtx:
    """One process of an initialised default process group; ``device`` holds the collective tensors."""
    rank: int
    world: int
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _reduce(self, vals: Sequence[float], op: str) -> list[float]:
        import torch
        import torch.distributed as td
        t = torch.tensor([float(v) for v in vals], dtype=torch.float64, device=self.device)
        td.all_reduce(t, op=getattr(td.ReduceOp, op))
        return t.tolist()

    def all_sum(self, vals: Sequence[float]) -> list[float]:
        return self._reduce(vals, "SUM")

    def all_max(self, vals: Sequence[float]) -> list[float]:
        return self._reduce(vals, "MAX")

    def gather(self, obj: Any) -> list:
        import torch.distributed as td
        out: list = [None] * self.world
        td.all_gather_object(out, obj)
        return out

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        import torch.distributed as td
        box = [obj]
        td.broadcast_object_list(box, src=src)
        return box[0]

    def barrier(self) -> None:
        import torch.distributed as td
        td.barrier()

    # ------------------------------------------------------------------ training hooks (grpo_v2.train_step)
    def step_totals(self, eps: list) -> tuple[int, int]:
        """(global episodes, global model tokens) of the step."""
        n, t = self.all_sum([len(eps), sum(sum(ep.env_mask) for ep in eps)])
        return int(n), int(t)

    def reduce_grads(self, params: list, n_local: int, n_global: int) -> None:
        """grad <- SUM_ranks(grad * n_local / n_global) over one flat fp32 buffer (missing grads count as 0)."""
        import torch
        import torch.distributed as td
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch.cat([g.reshape(-1).to(torch.float32) for g in grads])
        flat.mul_(n_local / n_global)
        td.all_reduce(flat, op=td.ReduceOp.SUM)
        off = 0
        for p, g in zip(params, grads):
            new = flat[off:off + g.numel()].view_as(g).to(g.dtype)
            if p.grad is None:
                p.grad = new.clone()
            else:
                p.grad.copy_(new)
            off += g.numel()

    def merge_step(self, out: dict, eps: list, items: list) -> dict:
        """The global step record from every rank's ``train_step`` result (rank order = global prompt order).
        ``used_items`` = all ranks' trained prompts (run_steps then logs global rows)."""
        parts = self.gather({"out": {k: v for k, v in out.items() if k != "used_items"},
                             "ledgers": [ep.ledger for ep in eps], "items": list(items)})
        outs = [p["out"] for p in parts]
        n = sum(o["n_episodes"] for o in outs)
        rewards = [r for o in outs for r in o["rewards"]]
        merged = dict(outs[0])                    # grad_norm(s) / updates_per_batch: identical after reduce_grads
        merged.update(n_episodes=n, rewards=rewards, reward_mean=sum(rewards) / n,
                      records=[r for o in outs for r in o["records"]],
                      episodes=[e for o in outs for e in o["episodes"]],
                      components=component_means([led for p in parts for led in p["ledgers"]]),
                      opsd_gate_rate=sum(o["opsd_gate_rate"] * o["n_episodes"] for o in outs) / n,
                      used_items=[it for p in parts for it in p["items"]], ddp_world_size=self.world,
                      ddp_episodes_per_rank=[o["n_episodes"] for o in outs])
        for k in MAX_KEYS:
            if k in merged:
                merged[k] = max(o[k] for o in outs if k in o)
        for k in SUM_KEYS:
            if k in merged:
                merged[k] = sum(o.get(k, 0) for o in outs)
        if "dapo_no_informative" in merged:
            merged["dapo_no_informative"] = any(o.get("dapo_no_informative") for o in outs)
        if "overlong_penalty_mean" in merged:
            merged["overlong_penalty_mean"] = sum(o["overlong_penalty_mean"] * o["n_episodes"] for o in outs) / n
        if "vef_raw" in merged:                                            # VEF presets: every rank's judge stats
            from .vef import merge_raw
            merged["vef_raw"] = merge_raw([o["vef_raw"] for o in outs])
        return merged

    # ------------------------------------------------------------------ replicas
    def broadcast_params(self, params: list) -> None:
        import torch.distributed as td
        for p in params:
            td.broadcast(p.data, src=0)

    def sync_check(self, params: list) -> bool:
        """True if every rank holds bit-identical LoRA weights; else warn and re-broadcast rank 0's."""
        import hashlib
        h = hashlib.sha256()
        for p in params:
            h.update(p.detach().float().cpu().numpy().tobytes())
        digests = self.gather(h.hexdigest())
        if len(set(digests)) == 1:
            return True
        logger.warning("LoRA replicas diverged across ranks (%s): re-broadcasting rank 0", digests)
        self.broadcast_params(params)
        return False


def init_from_env(timeout_min: Optional[int] = None) -> Optional[DistCtx]:
    """torchrun environment -> initialised process group (nccl on GPU, gloo on CPU); None when WORLD_SIZE <= 1.
    The timeout covers rank 0's dev eval while the others wait at a barrier (env DDP_TIMEOUT_MIN, default 180)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return None
    import torch
    import torch.distributed as td
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    minutes = timeout_min if timeout_min is not None else int(os.environ.get("DDP_TIMEOUT_MIN", "180"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend, device = "nccl", f"cuda:{local}"
    else:
        backend, device = "gloo", "cpu"
    td.init_process_group(backend, timeout=timedelta(minutes=minutes))
    logger.info("ddp: rank %d / %d (local %d) backend %s", rank, world, local, backend)
    return DistCtx(rank, world, device)


def shutdown() -> None:
    import torch.distributed as td
    if td.is_initialized():
        td.destroy_process_group()


def rank_slice(batch: list, dist: DistCtx, per_rank: int) -> list:
    """Rank r's share of a global batch of world * per_rank prompts: positions [r * per_rank, (r + 1) * per_rank)."""
    if len(batch) != dist.world * per_rank:
        raise ValueError(f"global batch of {len(batch)} != world {dist.world} x {per_rank}")
    return batch[dist.rank * per_rank:(dist.rank + 1) * per_rank]


class StridedStream:
    """Rank r's view of a shared stream of extra prompts (DAPO refills): global positions r, r + W, r + 2W, ... so
    ranks never draw the same position without communicating. ``next_fn(k)`` returns the next k stream elements
    (every rank computes the same stream). After the step the stream consumed W * max_r(taken_r) positions
    (``consumed``); positions no rank took are skipped."""

    def __init__(self, next_fn: Callable[[int], list], rank: int, world: int) -> None:
        self.next_fn, self.rank, self.world = next_fn, rank, world
        self.buf: list = []
        self.taken = 0

    def take(self, k: int) -> list:
        if k <= 0:
            return []
        need = self.rank + self.world * (self.taken + k - 1) + 1
        if need > len(self.buf):
            self.buf.extend(self.next_fn(need - len(self.buf)))
        got = [self.buf[self.rank + self.world * (self.taken + j)] for j in range(k)]
        self.taken += k
        return got

    @staticmethod
    def consumed(max_taken: int, world: int) -> int:
        return world * max_taken

    def taken_items(self, taken: Sequence[int]) -> list:
        """Every rank's refills (rank order) given all ranks' ``taken`` counts (extends this rank's buffer)."""
        need = self.consumed(max(taken), self.world)
        if need > len(self.buf):
            self.buf.extend(self.next_fn(need - len(self.buf)))
        return [self.buf[r + self.world * j] for r, n in enumerate(taken) for j in range(n)]


def rank_seed(seed: int, rank: int) -> int:
    return seed + RANK_SEED_STRIDE * rank


def rng_state() -> dict:
    import torch
    return {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    import torch
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])


def reseed_rank(seed: int, rank: int) -> None:
    """Fresh run: rank 0 keeps the run's seeded RNG; rank r > 0 samples from ``rank_seed`` (distinct rollouts)."""
    if rank == 0:
        return
    import torch
    s = rank_seed(seed, rank)
    random.seed(s)
    torch.manual_seed(s)                     # also seeds every CUDA device
