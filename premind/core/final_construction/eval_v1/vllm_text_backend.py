"""vLLM generation for the external baselines (owner 2026-10-06: all queued models on vLLM where possible).

The baseline encoders (baseline_backend.BaselineBackend for Qwen-VL families, generic_backend.GenericBackend for
InternVL / GLM) are reused unchanged except for ``_ids``: sequences are kept UNEXPANDED (tokenizer only, the chat
template's image markup with one placeholder per image). ``generate`` decodes each sequence back to its template text
and sends text + images to vLLM, whose HF processor expands the placeholders exactly as the HF path did
(``proc(text=..., images=...)``); ``add_special_tokens`` is off because the template text already carries BOS etc.
Stop ids / EOS cutting follow the encoder (``stop_ids`` + ``cut`` for generic backends, <|im_end|> for Qwen-VL).
Families whose architecture vLLM 0.18 lacks (Gemma-4) and Phi-4 (remote code; served by local_vllm_eval.sh) keep HF.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from eval_v1.vllm_backend import fit_memory_utilization
from trainer_v2.grpo_v2 import Enc, Images

logger = logging.getLogger(__name__)
VLLM_FAMILIES = ("qwen3_vl", "qwen2_5_vl", "internvl", "glm4v")
MAX_IMAGES_PER_PROMPT = 32


class VLLMTextBackend:
    """grpo_v2.Backend: baseline encoder (unexpanded ids) + vLLM text-prompt generation."""

    def __init__(self, enc: Any, llm: Any, *, temperature: float = 1.0) -> None:
        self.enc, self.llm, self.temperature = enc, llm, temperature
        tok = enc.tok
        enc._ids = lambda text, pils: tok(text, add_special_tokens=False)["input_ids"]   # unexpanded placeholders
        self.stop_ids = list(getattr(enc, "stop_ids", [enc.eos_token_id]))
        self.stats = getattr(enc, "stats", {})

    def __getattr__(self, name: str) -> Any:   # encode_prompt / encode_env_turn / encode_text / decode / eos_token_id
        return getattr(self.enc, name)

    def _cut(self, row: list[int]) -> list[int]:
        if hasattr(self.enc, "cut"):
            return self.enc.cut(row)
        eos = self.enc.eos_token_id
        return row[:row.index(eos) + 1] if eos in row else row

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:
        from vllm import SamplingParams
        if teacher:
            raise ValueError("baselines have no teacher")
        ims = list(images) if isinstance(images, (list, tuple)) else [images] * len(seqs)
        sp = SamplingParams(n=num_generations, max_tokens=max_tokens, stop_token_ids=self.stop_ids,
                            temperature=0.0 if greedy else (temperature or self.temperature), top_k=-1, top_p=1.0,
                            detokenize=False, seed=None if greedy else 0)
        reqs = []
        for s, im in zip(seqs, ims, strict=True):
            req: dict = {"prompt": self.enc.tok.decode(list(s.ids), skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)}
            if s.images:
                req["multi_modal_data"] = {"image": [im[p] for p in s.images]}
            reqs.append(req)
        res = self.llm.generate(reqs, sp, use_tqdm=False, tokenization_kwargs={"add_special_tokens": False})
        outs: list[list[int]] = []
        for r in res:
            for o in sorted(r.outputs, key=lambda o: o.index):
                row = list(o.token_ids)
                if o.finish_reason == "stop" and not any(t in self.stop_ids for t in row):
                    row.append(self.enc.eos_token_id)
                outs.append(self._cut(row))
        return outs, [None] * len(outs)


def load(key: str, *, temperature: float, max_model_len: int = 32768,
         gpu_memory_utilization: float = 0.85) -> tuple[VLLMTextBackend, dict]:  # pragma: no cover - GPU
    from vllm import LLM
    from eval_v1 import baselines as B
    meta = B.resolve(key)
    if meta["family"] not in VLLM_FAMILIES:
        raise ValueError(f"{key}: family {meta['family']} not served by vLLM here (use --backend hf)")
    proc = B.load_processor(meta)
    if meta.get("backend") == "generic":
        from eval_v1.generic_backend import from_meta
        enc = from_meta(None, proc, meta, temperature=temperature)
    else:
        from eval_v1.baseline_backend import BaselineBackend
        enc = BaselineBackend(None, proc, temperature=temperature)
    llm = LLM(model=meta["snapshot"], dtype=meta["dtype"], seed=0, max_model_len=max_model_len,
              gpu_memory_utilization=fit_memory_utilization(gpu_memory_utilization), enable_prefix_caching=True,
              limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT})
    logger.info("vLLM baseline %s: %s@%s", key, meta["model_id"], meta["revision"][:12])
    return VLLMTextBackend(enc, llm, temperature=temperature), {**meta, "backend_impl": "vllm_text"}
