"""grpo_v2.Backend for non-Qwen open baselines (EVAL ONLY): InternVL3.5, Gemma 4, GLM-4.6V (eval_v1/baselines.py).

Same protocol as trainer_v2.hf_backend.HFBackend (prompt, Stage 1, env turn continuation, Stage 2, passive arm) but
through each processor's OWN chat template:

  * Text parity: every text part of every message is rendered as a unique sentinel and substituted back after the
    template ran, so system / user / env text is byte-identical to what the Qwen3.5 models see (templates that
    ``| trim`` text parts, e.g. Gemma 4, cannot alter it). Only template markup differs (role headers, turn tokens,
    the per-image markup, Gemma's space after the system text, GLM's ``/nothink`` thinking-off switch).
  * Images: one template image placeholder per image part, in message order; the processor expands each (InternVL
    <img><IMG_CONTEXT>x256xtiles</img>, Gemma <|image><|image|>xN<image|>, GLM grid / merge**2 <|image|>).
  * Stage-2 continuation (``encode_env_turn``): the template rendering of [user, assistant SENTINEL, env user] after
    the sentinel = turn-end token + env user turn + assistant header; the turn-end token is dropped when Stage 1
    already ended with it. Generation stops at ANY shipped EOS id or the turn-end token; the stop token is returned
    as the turn-end id (``eos_token_id``), i.e. the history is continued exactly as the template renders an
    assistant turn (count of normalised stop tokens in ``stats``).
  * ``model_inputs``: the vision tensors come from one processor call over the sequence's images (markup x N), and
    its expanded image-token count must equal the sequence's; ``mm_token_type_ids`` (when the processor emits it) is
    rebuilt per token id from that call.
  * Decoding: greedy (do_sample False) / sampling top_k 0, top_p 1, like HFBackend; the max-token limits come from
    grpo_v2.LoopConfig. GLM: ``<|begin_of_box|>`` / ``<|end_of_box|>`` and an EMPTY leading ``<think></think>`` are
    stripped from decoded text (wrapper only, counted in ``stats``); a non-empty think block is left for the parser.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from trainer_v2.grpo_v2 import Enc, Images
from trainer_v2.hf_backend import template_messages

logger = logging.getLogger(__name__)
SENT = "⁣S{}⁣"
NON_VISION = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")
BOX = ("<|begin_of_box|>", "<|end_of_box|>")


class GenericBackend:
    """Backend over an HF image-text model whose chat template is not Qwen's."""

    def __init__(self, model: Any, processor: Any, *, turn_end: str, eos_ids: Sequence[int] = (),
                 template_kwargs: Optional[dict] = None, temperature: float = 1.0, device: str = "cuda",
                 strip_glm_wrappers: bool = False) -> None:
        self.model, self.proc, self.tok = model, processor, processor.tokenizer
        self.temperature, self.device = temperature, device
        self.ctk = dict(template_kwargs or {})
        self.turn_end = turn_end
        self.eos_token_id = self._special_id(turn_end)
        self.stop_ids = sorted({self.eos_token_id, *(int(i) for i in eos_ids)})
        self.image_token_id = self._special_id(processor.image_token)
        self.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_token_id
        self.strip_glm = strip_glm_wrappers
        self.markup = self._image_markup()
        self.stats = {"stop_token_normalised": 0, "glm_box_stripped": 0, "glm_empty_think_stripped": 0,
                      "glm_nonempty_think_left": 0}

    def _special_id(self, token: str) -> int:
        tid = self.tok.convert_tokens_to_ids(token)
        if tid is None or (self.tok.unk_token_id is not None and tid == self.tok.unk_token_id):
            raise ValueError(f"tokenizer has no {token!r} token")
        return int(tid)

    # ------------------------------------------------------------------ rendering (CPU-safe)
    def render(self, messages: list[dict], *, add_generation_prompt: bool) -> tuple[str, tuple[str, ...]]:
        """Chat-template text of rendered messages with every text part byte-identical + image paths in order."""
        tm, paths = template_messages(messages)
        texts: list[str] = []
        for m in tm:
            for part in m["content"]:
                if part["type"] == "text":
                    texts.append(part["text"])
                    part["text"] = SENT.format(len(texts) - 1)
        out = self.proc.apply_chat_template(tm, tokenize=False, add_generation_prompt=add_generation_prompt,
                                            **self.ctk)
        for i, t in enumerate(texts):
            if out.count(SENT.format(i)) != 1:
                raise AssertionError(f"chat template dropped or duplicated text part {i}")
            out = out.replace(SENT.format(i), t)
        return out, paths

    def _image_markup(self) -> str:
        """The template's rendering of one image part (text between two sentinel text parts)."""
        msg = [{"role": "user", "content": [{"type": "text", "text": SENT.format("a")}, {"type": "image"},
                                            {"type": "text", "text": SENT.format("b")}]}]
        out = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, **self.ctk)
        a, b = SENT.format("a"), SENT.format("b")
        return out[out.index(a) + len(a):out.index(b)]

    def _ids(self, text: str, pils: list) -> list[int]:
        enc = self.proc(text=[text], images=pils or None, return_tensors="pt")
        return enc["input_ids"][0].tolist()

    def encode_prompt(self, messages: list[dict], images: Images) -> Enc:
        text, paths = self.render(messages, add_generation_prompt=True)
        return Enc(tuple(self._ids(text, [images[p] for p in paths])), paths)

    def env_turn_text(self, content: list[dict], *, stage1_ended_with_eos: bool) -> tuple[str, tuple[str, ...]]:
        x = SENT.format("x")
        conv = [{"role": "user", "content": [{"type": "text", "text": x}]},
                {"role": "assistant", "content": [{"type": "text", "text": SENT.format("asst")}]},
                {"role": "user", "content": content}]
        text, paths = self.render(conv, add_generation_prompt=True)
        suffix = text.split(SENT.format("asst"), 1)[1]
        if not suffix.startswith(self.turn_end):
            raise AssertionError(f"chat template: env turn must start with the turn end {self.turn_end!r}")
        full, _ = self.render(conv[:1], add_generation_prompt=True)
        bare, _ = self.render(conv[:1], add_generation_prompt=False)
        header = full[len(bare):] if full.startswith(bare) else full[full.index(x) + len(x):]
        if not header or not suffix.endswith(header):
            raise AssertionError("Stage-2 generation header differs from the Stage-1 one")
        return (suffix[len(self.turn_end):] if stage1_ended_with_eos else suffix), paths

    def encode_env_turn(self, content: list[dict], images: Images, *, stage1_ended_with_eos: bool) -> Enc:
        text, paths = self.env_turn_text(content, stage1_ended_with_eos=stage1_ended_with_eos)
        return Enc(tuple(self._ids(text, [images[p] for p in paths])), paths)

    def encode_text(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def model_inputs(self, seq: Enc, images: Images) -> dict:
        """input_ids / attention_mask (+ mm_token_type_ids) + the processor's vision tensors for all images."""
        import torch
        ids = torch.tensor([list(seq.ids)], dtype=torch.long)
        out: dict = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        n_img = int((ids == self.image_token_id).sum())
        if not seq.images:
            if n_img:
                raise AssertionError("image placeholders without images")
            return out
        ref = self.proc(text=[self.markup * len(seq.images)], images=[images[p] for p in seq.images],
                        return_tensors="pt")
        need = int((ref["input_ids"] == self.image_token_id).sum())
        if n_img != need:
            raise AssertionError(f"{n_img} image tokens vs {need} expected for {len(seq.images)} images")
        runs = int(((ids[0, 1:] == self.image_token_id) & (ids[0, :-1] != self.image_token_id)).sum()
                   + int(ids[0, 0] == self.image_token_id))
        ref_runs = int(((ref["input_ids"][0, 1:] == self.image_token_id)
                        & (ref["input_ids"][0, :-1] != self.image_token_id)).sum())
        if runs != ref_runs:
            raise AssertionError(f"{runs} image-token runs vs {ref_runs} in the processor reference")
        if "mm_token_type_ids" in ref:
            kinds = {int(i): int(t) for i, t in zip(ref["input_ids"][0].tolist(), ref["mm_token_type_ids"][0].tolist())
                     if t}
            out["mm_token_type_ids"] = torch.tensor([[kinds.get(i, 0) for i in seq.ids]], dtype=torch.long)
        out.update({k: v for k, v in ref.items() if k not in NON_VISION})
        return out

    def decode(self, ids: Sequence[int]) -> str:
        text = self.tok.decode(list(ids), skip_special_tokens=True)
        if not self.strip_glm:
            return text
        if any(b in text for b in BOX):
            self.stats["glm_box_stripped"] += 1
            for b in BOX:
                text = text.replace(b, "")
        if EMPTY_THINK.match(text):
            self.stats["glm_empty_think_stripped"] += 1
            text = EMPTY_THINK.sub("", text, count=1)
        elif "<think>" in text:
            self.stats["glm_nonempty_think_left"] += 1
        return text

    def cut(self, row: list[int]) -> list[int]:
        """Cut after the first stop id; the stop token is returned as the turn-end id."""
        for j, t in enumerate(row):
            if t in self.stop_ids:
                if t != self.eos_token_id:
                    self.stats["stop_token_normalised"] += 1
                return row[:j] + [self.eos_token_id]
        return row

    # ------------------------------------------------------------------ GPU
    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[Optional[list[float]]]]:  # pragma: no cover - GPU job
        import torch
        if teacher or gen_batch:
            raise ValueError("GenericBackend: eval only (no teacher / batched generation)")
        ims = list(images) if isinstance(images, (list, tuple)) else [images] * len(seqs)
        outs: list[list[int]] = []
        dtype = getattr(self.model, "dtype", None)
        for seq, im in zip(seqs, ims, strict=True):
            inp = {k: (v.to(self.device, dtype=dtype) if v.is_floating_point() and dtype is not None
                       else v.to(self.device)) for k, v in self.model_inputs(seq, im).items()}
            kw = dict(max_new_tokens=max_tokens, num_return_sequences=num_generations, use_cache=True,
                      eos_token_id=self.stop_ids, pad_token_id=self.pad_token_id, return_dict_in_generate=True)
            kw.update(do_sample=False) if greedy else kw.update(do_sample=True,
                                                                 temperature=temperature or self.temperature,
                                                                 top_k=0, top_p=1.0)
            with torch.no_grad():
                res = self.model.generate(**inp, **kw)
            outs += [self.cut(row) for row in res.sequences[:, inp["input_ids"].shape[1]:].tolist()]
        return outs, [None] * len(outs)

    def logits_at(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - eval only
        raise NotImplementedError("GenericBackend is eval only")

    def logits_at_batch(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - eval only
        raise NotImplementedError("GenericBackend is eval only")


def from_meta(model: Any, processor: Any, meta: dict, *, temperature: float) -> GenericBackend:
    """GenericBackend configured from a resolved baseline meta (eval_v1/baselines.resolve)."""
    shipped = (meta.get("shipped_generation_config") or {}).get("eos_token_id")
    eos = shipped if isinstance(shipped, list) else ([shipped] if shipped is not None else [])
    return GenericBackend(model, processor, turn_end=meta["turn_end"], eos_ids=eos,
                          template_kwargs=meta.get("template_kwargs"), temperature=temperature,
                          strip_glm_wrappers=meta["family"] == "glm4v")
