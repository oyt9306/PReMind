"""HF transformers backend (GPU jobs only; encode helpers are CPU-testable with the real processor).

Rollouts use ``model.generate`` on the training GPU (no vLLM server): sampling at temperature T with top_k=0,
top_p=1.0 (HF defaults would otherwise apply top_k=50). Sampler log-probs are not materialised (the sampler is the
same HF model, so the trainer's on-policy log-probs are used: IS ratio 1; vLLM-mismatch IS is moot here).
Every forward gets ``input_ids`` + ``pixel_values`` / ``image_grid_thw`` for ALL images in sequence order and
``mm_token_type_ids`` (1 on <|image_pad|>) so Qwen3.5 M-RoPE is computed over the whole multi-image sequence.
Batched generation (``gen_batch`` > 0): different sequences are LEFT-padded into one call (pad id, attention 0,
mm type 0 on the left); ``pixel_values`` / ``image_grid_thw`` are the per-row tensors concatenated in row order, each
row's images in sequence order - the layout the Qwen3.5 processor returns for a batch and that ``get_rope_index`` /
``_expand_inputs_for_generation`` walk (images of row b, then row b + 1). Left padding is the layout Qwen3.5's
linear-attention layers expect (``apply_mask_to_padding_states`` zeroes the pad states; M-RoPE ignores masked
positions and stores per-row rope deltas). The micro-batched training forward (``logits_at_batch``, only when
``LoopConfig.train_micro_batch`` > 1) uses the same left-padded ``batch_inputs``.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional, Sequence

from .batch_rollout import chunk_spans
from .grpo_v2 import Enc, Images

logger = logging.getLogger(__name__)
SENTINEL = "⁣TRAINER_V2_SENTINEL⁣"
IM_END = "<|im_end|>"


def template_messages(messages: list[dict]) -> tuple[list[dict], tuple[str, ...]]:
    """Chat-template view of rendered messages (image parts -> {"type": "image"}) + image paths in order."""
    out, paths = [], []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for p in content:
            if p["type"] == "image":
                parts.append({"type": "image"})
                paths.append(p["image"])
            else:
                parts.append({"type": "text", "text": p["text"]})
        out.append({"role": m["role"], "content": parts})
    return out, tuple(paths)


class HFBackend:
    """grpo_v2.Backend over a (PEFT-wrapped) Qwen3.5 image-text model and its processor."""

    def __init__(self, model: Any, processor: Any, *, temperature: float = 1.0, device: str = "cuda",
                 enable_thinking: bool = False) -> None:
        self.model, self.proc, self.tok = model, processor, processor.tokenizer
        self.temperature, self.device = temperature, device
        self.ctk = {"enable_thinking": enable_thinking}
        self.eos_token_id = self.tok.convert_tokens_to_ids(IM_END)
        self.image_token_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_token_id
        self.merge = getattr(processor.image_processor, "merge_size", 2)

    # ------------------------------------------------------------------ encoding (CPU-safe)
    def _ids(self, text: str, pils: list) -> list[int]:
        enc = self.proc(text=[text], images=pils or None, return_tensors="pt")
        return enc["input_ids"][0].tolist()

    def encode_prompt(self, messages: list[dict], images: Images) -> Enc:
        tm, paths = template_messages(messages)
        text = self.proc.apply_chat_template(tm, tokenize=False, add_generation_prompt=True, **self.ctk)
        return Enc(tuple(self._ids(text, [images[p] for p in paths])), paths)

    def encode_env_turn(self, content: list[dict], images: Images, *, stage1_ended_with_eos: bool) -> Enc:
        """Tokens that close the Stage-1 assistant turn (if the model did not), the env user turn, and the next
        assistant header - exactly as the chat template renders them after an assistant message."""
        tm, paths = template_messages([{"role": "user", "content": content}])
        conv = [{"role": "user", "content": [{"type": "text", "text": "x"}]},
                {"role": "assistant", "content": [{"type": "text", "text": SENTINEL}]}, tm[0]]
        text = self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **self.ctk)
        suffix = text.split(SENTINEL, 1)[1]
        if not suffix.startswith(IM_END):
            raise AssertionError("chat template: env turn must start by closing the assistant turn")
        head = self.proc.apply_chat_template(conv[:1], tokenize=False, add_generation_prompt=True, **self.ctk)
        if not suffix.endswith(head[head.rindex("<|im_start|>assistant"):]):
            raise AssertionError("Stage-2 generation header differs from the Stage-1 one")
        if stage1_ended_with_eos:
            suffix = suffix[len(IM_END):]
        return Enc(tuple(self._ids(suffix, [images[p] for p in paths])), paths)

    def model_inputs(self, seq: Enc, images: Images) -> dict:
        """input_ids / attention_mask / mm_token_type_ids / pixel_values / image_grid_thw for one sequence."""
        import torch
        ids = torch.tensor([list(seq.ids)], dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
               "mm_token_type_ids": (ids == self.image_token_id).long()}
        if seq.images:
            iv = self.proc.image_processor(images=[images[p] for p in seq.images], return_tensors="pt")
            n_pad = int(out["mm_token_type_ids"].sum())
            need = int((iv["image_grid_thw"].prod(-1) // self.merge ** 2).sum())
            if n_pad != need:
                raise AssertionError(f"{n_pad} image placeholders vs {need} vision tokens for {len(seq.images)} images")
            out.update(pixel_values=iv["pixel_values"], image_grid_thw=iv["image_grid_thw"])
        elif int(out["mm_token_type_ids"].sum()):
            raise AssertionError("image placeholders without images")
        return out

    def batch_inputs(self, seqs: Sequence[Enc], images: Sequence[Images]) -> dict:
        """``model_inputs`` of several sequences as one left-padded batch (see module docstring)."""
        import torch
        rows = [self.model_inputs(s, im) for s, im in zip(seqs, images, strict=True)]
        width = max(r["input_ids"].shape[1] for r in rows)
        fill = {"input_ids": self.pad_token_id, "attention_mask": 0, "mm_token_type_ids": 0}
        out = {k: torch.cat([torch.nn.functional.pad(r[k], (width - r[k].shape[1], 0), value=v) for r in rows])
               for k, v in fill.items()}
        vis = [r for r in rows if "pixel_values" in r]
        if vis:
            out.update(pixel_values=torch.cat([r["pixel_values"] for r in vis]),
                       image_grid_thw=torch.cat([r["image_grid_thw"] for r in vis]))
        return out

    # ------------------------------------------------------------------ GPU
    def _adapter_off(self, teacher: bool):
        if teacher and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def encode_text(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:
        """``gen_batch`` 0: one ``model.generate`` per sequence (the pre-batching behaviour). > 0: left-padded calls
        over ``chunk_spans(len(seqs), num_generations, gen_batch)``. Outputs are grouped by sequence in input order
        (num_return_sequences rows each) and cut after the first EOS either way."""
        import torch
        outs: list[list[int]] = []
        ims = list(images) if isinstance(images, (list, tuple)) else [images] * len(seqs)
        if len(ims) != len(seqs):
            raise ValueError("one images dict per sequence (or one shared dict)")
        spans = (chunk_spans(len(seqs), num_generations, gen_batch) if gen_batch
                 else [(j, j + 1) for j in range(len(seqs))])
        was_training = self.model.training
        self.model.eval()
        try:
            for lo, hi in spans:
                if not gen_batch:                         # pre-batching path: unpadded single-sequence inputs
                    inp = {k: v.to(self.device) for k, v in self.model_inputs(seqs[lo], ims[lo]).items()}
                else:
                    inp = {k: v.to(self.device) for k, v in self.batch_inputs(seqs[lo:hi], ims[lo:hi]).items()}
                kw = dict(max_new_tokens=max_tokens, num_return_sequences=num_generations, use_cache=True,
                          eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
                          return_dict_in_generate=True)
                kw.update(do_sample=False) if greedy else kw.update(do_sample=True,
                                                                     temperature=temperature or self.temperature,
                                                                     top_k=0, top_p=1.0)
                with torch.no_grad(), self._adapter_off(teacher):
                    res = self.model.generate(**inp, **kw)
                for row in res.sequences[:, inp["input_ids"].shape[1]:].tolist():
                    cut = row.index(self.eos_token_id) + 1 if self.eos_token_id in row else len(row)
                    outs.append(row[:cut])
        finally:
            self.model.train(was_training)
        return outs, [None] * len(outs)

    def logits_at(self, seq: Enc, positions: Sequence[int], images: Images, *, teacher: bool) -> Any:
        """Logits predicting ``seq.ids[p]`` for each p (computed at p - 1); grad unless ``teacher``."""
        import torch
        inp = {k: v.to(self.device) for k, v in self.model_inputs(seq, images).items()}
        keep = torch.tensor([p - 1 for p in positions], dtype=torch.long, device=self.device)
        with self._adapter_off(teacher):
            out = self.model(**inp, use_cache=False, logits_to_keep=keep)
        return out.logits[0]

    def logits_at_batch(self, seqs: Sequence[Enc], positions: Sequence[Sequence[int]], images: Sequence[Images], *,
                        teacher: bool) -> list[Any]:
        """``logits_at`` of several sequences in ONE left-padded forward (``batch_inputs``; train_mb.py explains why
        left padding). Row b: logits predicting ``seqs[b].ids[p]`` for p in ``positions[b]``, taken from the final
        hidden states at the row's pad offset + p - 1 and passed through the LM head (the decoder body + head split
        of the model's forward, so no logits are computed at pad / prompt / env positions); grad unless ``teacher``.
        Text-only batches get explicit mask-cumsum positions (HF would otherwise use arange over the padded width)."""
        import torch
        inp = {k: v.to(self.device) for k, v in self.batch_inputs(seqs, images).items()}
        width = inp["input_ids"].shape[1]
        if "image_grid_thw" not in inp:
            inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        body, head = base.model, base.get_output_embeddings()
        with self._adapter_off(teacher):
            hidden = body(**inp, use_cache=False).last_hidden_state
        out = []
        for b, (seq, pos) in enumerate(zip(seqs, positions, strict=True)):
            off = width - len(seq.ids)
            idx = torch.tensor([off + p - 1 for p in pos], dtype=torch.long, device=hidden.device)
            out.append(head(hidden[b].index_select(0, idx)))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)
