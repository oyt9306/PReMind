"""grpo_v2.Backend for the external baselines (eval only): HFBackend with the family's chat-template kwargs.

Everything that touches the prompt or the protocol is inherited unchanged from trainer_v2.hf_backend.HFBackend:
``template_messages`` (same system / user / env-turn text), ``encode_prompt``, ``encode_env_turn`` (the Stage-2
continuation: close the Stage-1 assistant turn, env user turn with the memory images, next assistant header),
``model_inputs`` / ``generate`` (greedy = do_sample False; sampling top_k 0 / top_p 1; EOS = <|im_end|>; the
max-token limits come from grpo_v2.LoopConfig exactly as for Qwen3.5).

Family differences (transformers 5.12):
  * chat template: Qwen3-VL Instruct and Qwen2.5-VL templates have no thinking switch, so no ``enable_thinking``
    kwarg is passed and no ``<think>\\n\\n</think>\\n\\n`` block is added to the assistant header (the Qwen3.5 template
    adds that block when ``enable_thinking=False``; that block is the only rendered-text difference).
  * ``mm_token_type_ids`` IS kept: Qwen3-VL raises without it when images are passed, and Qwen2.5-VL silently falls
    back to 1D positions without it (wrong M-RoPE); both processors return it, 1 on <|image_pad|> (same id 151655).
  * image placeholders: the template emits one ``<|vision_start|><|image_pad|><|vision_end|>`` per image for both
    families; the processor expands each to grid / merge**2 pads (patch 16 for Qwen3-VL, 14 for Qwen2.5-VL), which
    ``model_inputs`` checks against the image processor's grid for every sequence.
"""
from __future__ import annotations

import logging
from typing import Any

from trainer_v2.hf_backend import IM_END, HFBackend

logger = logging.getLogger(__name__)
IMAGE_PAD = "<|image_pad|>"


class BaselineBackend(HFBackend):
    """HFBackend without chat-template kwargs (no thinking switch)."""

    def __init__(self, model: Any, processor: Any, *, temperature: float = 1.0, device: str = "cuda") -> None:
        super().__init__(model, processor, temperature=temperature, device=device)
        self.ctk = {}
        unk = self.tok.unk_token_id
        for tok, tid in ((IM_END, self.eos_token_id), (IMAGE_PAD, self.image_token_id)):
            if tid is None or (unk is not None and tid == unk):
                raise ValueError(f"baseline tokenizer has no {tok} token")
