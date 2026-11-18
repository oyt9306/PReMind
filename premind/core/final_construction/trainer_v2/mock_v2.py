"""CPU mock backend for trainer_v2 tests and the --dry-run: trainer_v1's lossless piece tokenizer + tiny GRU policy,
multi-image placeholders, scripted samplers. The "teacher" is a frozen copy of the initial policy (LoRA disabled)."""
from __future__ import annotations

import copy
import random
from typing import Any, Optional, Sequence

import torch

from .batch_rollout import chunk_spans
from .grpo_v2 import Enc, Images
from trainer_v1.mock import IM_START, N_IMG_TOKENS, MockPolicy, MockTokenizer

STAGE1_CHOICES = ("<memory_call>call_pdb</memory_call>", "<memory_call>direct_answer</memory_call>",
                  "I am not sure what to do here")


def completion(risk: str, action: str, answer: str, cite: str = "none") -> str:
    """A strict-format Stage-2 / passive completion (final_prompts.SYSTEM_CORE_FINAL schema)."""
    return (f"<decision>\n<visual_evidence>v</visual_evidence>\n<reasoning>r</reasoning>\n<context_use>{cite}"
            f"</context_use>\n<risk>{risk}</risk>\n<action>{action}</action>\n<confidence>high</confidence>\n"
            f"</decision>\n<answer>{answer}</answer>")


class MockBackend:
    """Implements grpo_v2.Backend. Each image part becomes N_IMG_TOKENS placeholders; images are recorded per call."""

    def __init__(self, stage2_texts: Sequence[str], seed: int = 0, vocab_size: int = 16384) -> None:
        self.tok = MockTokenizer(vocab_size)
        self.policy = MockPolicy(vocab_size)
        self.base = copy.deepcopy(self.policy).requires_grad_(False)   # "adapter disabled"
        self.rng = random.Random(seed)
        self.stage2_texts = list(stage2_texts)
        self.eos_token_id = self.tok.eos_token_id
        self.calls: list[dict] = []
        self.batches: list[int] = []          # output rows per backend generate call (sequential: G per sequence)
        self.train_batches: list[int] = []    # rows per logits_at_batch call (train_micro_batch > 1)

    def _render(self, messages: list[dict], images: Images, *, gen: bool) -> Enc:
        paths = tuple(p["image"] for m in messages if isinstance(m["content"], list)
                      for p in m["content"] if p["type"] == "image")
        missing = [p for p in paths if p not in images]
        if missing:
            raise KeyError(f"images not loaded for this batch: {missing[:2]}")
        return Enc(tuple(self.tok.encode(self.tok.render(messages, add_generation_prompt=gen))), paths)

    def encode_prompt(self, messages: list[dict], images: Images) -> Enc:
        return self._render(messages, images, gen=True)

    def encode_env_turn(self, content: list[dict], images: Images, *, stage1_ended_with_eos: bool) -> Enc:
        enc = self._render([{"role": "user", "content": content}], images, gen=True)
        ids = ([] if stage1_ended_with_eos else [self.eos_token_id]) + self.tok.encode("\n") + list(enc.ids)
        return Enc(tuple(ids), enc.images)

    def generate(self, seqs: list[Enc], images: Images | Sequence[Images], *, num_generations: int, max_tokens: int,
                 greedy: bool, teacher: bool = False, temperature: Optional[float] = None, gen_batch: int = 0
                 ) -> tuple[list[list[int]], list[list[float]]]:
        """Batched calls (``gen_batch`` > 0) are split like HFBackend (``chunk_spans``) and recorded in ``batches``;
        per-sequence images (a list) must hold every image of their sequence."""
        outs, lps = [], []
        model = self.base if teacher else self.policy
        ims = list(images) if isinstance(images, (list, tuple)) else [images] * len(seqs)
        if len(ims) != len(seqs):
            raise ValueError("one images dict per sequence (or one shared dict)")
        spans = (chunk_spans(len(seqs), num_generations, gen_batch) if gen_batch
                 else [(j, j + 1) for j in range(len(seqs))])
        self.batches += [(hi - lo) * num_generations for lo, hi in spans]
        for s, im in zip(seqs, ims):
            if s.ids.count(self.tok.image_token_id) != N_IMG_TOKENS * len(s.images):
                raise AssertionError("image placeholders and images are out of sync")
            if any(p not in im for p in s.images):
                raise KeyError("sequence images missing from its images dict")
            last_user = self.tok.decode(s.ids).rsplit(f"{IM_START}user\n", 1)[-1]
            stage1 = "Step 1:" in last_user
            self.calls.append({"n_images": len(s.images), "stage1": stage1, "max_tokens": max_tokens,
                               "greedy": greedy, "teacher": teacher, "prompt_len": len(s.ids)})
            for k in range(num_generations):
                pool = STAGE1_CHOICES if stage1 else self.stage2_texts
                text = self._choose(pool, s, k, greedy)
                ids = (self.tok.encode(text) + [self.eos_token_id])[:max_tokens]
                outs.append(ids)
                lps.append(self._logps(model, list(s.ids), ids))
        return outs, lps

    def _choose(self, pool: Sequence[str], seq: Enc, k: int, greedy: bool) -> str:
        """Sampler stand-in (tests may override it with an order-independent choice)."""
        return pool[0] if greedy else self.rng.choice(pool)

    @staticmethod
    def _logps(policy: Any, prompt: list[int], comp: list[int]) -> list[float]:
        with torch.no_grad():
            ids = torch.tensor([prompt + comp])
            lp = torch.log_softmax(policy(ids)[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0]
        return lp[len(prompt) - 1:].tolist()

    def logits_at(self, seq: Enc, positions: Sequence[int], images: Images, *, teacher: bool) -> Any:
        if seq.ids.count(self.tok.image_token_id) != N_IMG_TOKENS * len(seq.images):
            raise AssertionError("training sequence: image placeholders and images are out of sync")
        model = self.base if teacher else self.policy
        logits = model(torch.tensor([list(seq.ids)]))[0]
        return logits[torch.tensor([p - 1 for p in positions], dtype=torch.long)]

    def logits_at_batch(self, seqs: Sequence[Enc], positions: Sequence[Sequence[int]], images: Sequence[Images], *,
                        teacher: bool) -> list[Any]:
        """Padded batch forward of the GRU policy (rows recorded in ``train_batches``). The GRU has no attention
        mask, so rows are RIGHT-padded with EOS: it is causal, so pads after a row never reach its real positions
        (HFBackend left-pads with an attention mask instead)."""
        for s in seqs:
            if s.ids.count(self.tok.image_token_id) != N_IMG_TOKENS * len(s.images):
                raise AssertionError("training sequence: image placeholders and images are out of sync")
        if len(seqs) != len(positions) or len(seqs) != len(images):
            raise ValueError("seqs, positions and images must align")
        self.train_batches.append(len(seqs))
        width = max(len(s.ids) for s in seqs)
        ids = torch.tensor([list(s.ids) + [self.eos_token_id] * (width - len(s.ids)) for s in seqs])
        logits = (self.base if teacher else self.policy)(ids)
        return [logits[b, torch.tensor([p - 1 for p in pos], dtype=torch.long)] for b, pos in enumerate(positions)]

    def encode_text(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids, skip_special=True)


def fake_images(item: Any) -> dict[str, str]:
    """Loader stand-in for tests: path -> path (no file I/O)."""
    return {p: p for p in item.image_paths}

