"""vLLM inference backend for Qwen3.5 evaluation (owner 2026-10-06: evals on vLLM instead of HF generate).

Encoding is inherited from ``trainer_v2.hf_backend.HFBackend`` (same chat template, same Stage-1 / env-turn token
layout); only ``generate`` runs on vLLM. HF-encoded sequences carry EXPANDED image placeholders (one <|image_pad|> per
vision token); vLLM expands placeholders itself, so every run of <|image_pad|> is collapsed back to one token before
the request (vLLM re-expands it with the same HF image processor). LoRA: vLLM 0.18 LoRA fails on Qwen3.5 (packed
linear-attention projections, IndexError in set_lora at warm-up), so the adapter is merged into the base weights once
(PEFT merge_and_unload, cached under eval_v1/vllm_merged/<sha12 of the adapter>) and the merged model is served. Greedy = temperature 0; sampling = temperature T, top_k / top_p disabled.
Outputs follow HFBackend.generate: grouped per sequence in input order, cut after the first EOS (EOS kept).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from trainer_v2.grpo_v2 import Enc, Images
from trainer_v2.hf_backend import HFBackend

logger = logging.getLogger(__name__)
MAX_IMAGES_PER_PROMPT = 32


def collapse_image_runs(ids: Sequence[int], image_token_id: int) -> list[int]:
    """Each maximal run of ``image_token_id`` -> a single token (vLLM re-expands it)."""
    out: list[int] = []
    for t in ids:
        if t == image_token_id and out and out[-1] == image_token_id:
            continue
        out.append(t)
    return out


class VLLMBackend(HFBackend):
    """grpo_v2.Backend: HFBackend encoders + vLLM ``LLM.generate``."""

    def __init__(self, llm: Any, processor: Any, *, lora: Optional[str] = None, temperature: float = 1.0,
                 enable_thinking: bool = False) -> None:
        super().__init__(None, processor, temperature=temperature, device="cuda", enable_thinking=enable_thinking)
        self.llm = llm
        self.lora_request = None
        if lora:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("eval_lora", 1, lora)

    def _request(self, seq: Enc, images: Images) -> dict:
        ids = collapse_image_runs(seq.ids, self.image_token_id)
        n_runs = sum(1 for a, b in zip([None] + ids[:-1], ids) if b == self.image_token_id)
        if n_runs != len(seq.images):
            raise AssertionError(f"{n_runs} image placeholder runs vs {len(seq.images)} images")
        req: dict = {"prompt_token_ids": ids}
        if seq.images:
            req["multi_modal_data"] = {"image": [images[p] for p in seq.images]}
        return req

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:
        """All sequences in ONE vLLM call (vLLM batches internally; ``gen_batch`` is ignored)."""
        from vllm import SamplingParams
        ims = list(images) if isinstance(images, (list, tuple)) else [images] * len(seqs)
        if len(ims) != len(seqs):
            raise ValueError("one images dict per sequence (or one shared dict)")
        sp = SamplingParams(n=num_generations, max_tokens=max_tokens, stop_token_ids=[self.eos_token_id],
                            temperature=0.0 if greedy else (temperature or self.temperature), top_k=-1, top_p=1.0,
                            detokenize=False, seed=None if greedy else 0)
        reqs = [self._request(s, im) for s, im in zip(seqs, ims, strict=True)]
        res = self.llm.generate(reqs, sp, lora_request=None if teacher else self.lora_request, use_tqdm=False)
        outs: list[list[int]] = []
        for r in res:
            for o in sorted(r.outputs, key=lambda o: o.index):
                row = list(o.token_ids)
                if o.finish_reason == "stop" and self.eos_token_id not in row:
                    row.append(self.eos_token_id)
                cut = row.index(self.eos_token_id) + 1 if self.eos_token_id in row else len(row)
                outs.append(row[:cut])
        return outs, [None] * len(outs)

    def logits_at(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - eval-only backend
        raise NotImplementedError("VLLMBackend is generation-only")

    logits_at_batch = logits_at


def fit_memory_utilization(target: float = 0.85, margin: float = 0.04) -> float:  # pragma: no cover - GPU
    """vLLM fraction of TOTAL memory, capped by what is free now (shared GPUs may hold another user's process)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return round(max(0.3, min(target, free / total - margin)), 3)


def merged_dir(size: str, lora: str) -> "Path":  # pragma: no cover - GPU
    """Base + LoRA merged once into a cache dir (complete marker); returns the dir vLLM loads."""
    import gc
    import hashlib
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trainer_v2 import paths_v2 as P
    mid, rev = P.MODELS[size]
    sha = hashlib.sha256((Path(lora) / "adapter_model.safetensors").read_bytes()).hexdigest()[:12]
    out = Path(__file__).resolve().parent / "vllm_merged" / f"{size}_{sha}"
    if (out / "MERGED_COMPLETE").is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(mid, revision=rev, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, lora).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(mid, revision=rev).save_pretrained(out)
    (out / "MERGED_COMPLETE").write_text(f"{mid}@{rev} + {lora} (adapter sha {sha})\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("merged LoRA %s -> %s", lora, out)
    return out


def load(size: str, *, lora: Optional[str], temperature: float, max_model_len: int = 32768,
         gpu_memory_utilization: float = 0.85) -> VLLMBackend:  # pragma: no cover - GPU
    from transformers import AutoProcessor
    from vllm import LLM
    from trainer_v2 import paths_v2 as P
    mid, rev = P.MODELS[size]
    proc = AutoProcessor.from_pretrained(mid, revision=rev)
    src = dict(model=str(merged_dir(size, lora))) if lora else dict(model=mid, revision=rev)
    llm = LLM(**src, dtype="bfloat16", seed=0, max_model_len=max_model_len,
              gpu_memory_utilization=fit_memory_utilization(gpu_memory_utilization), enable_prefix_caching=True,
              limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT})
    logger.info("vLLM backend: %s@%s lora=%s (merged)", mid, rev[:12], lora)
    return VLLMBackend(llm, proc, lora=None, temperature=temperature)
