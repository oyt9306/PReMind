"""CPU mocks for the dry run and tests: a lossless piece tokenizer, a tiny causal policy and a scripted sampler.

Nothing here downloads a model. The scripted sampler picks Stage-1 / Stage-2 texts with a seeded RNG (the reward
needs realistic outputs to vary within a group); the per-token logprobs it returns are computed under the mock
policy, as vLLM would return sampling logprobs.
"""
from __future__ import annotations

import random
import re
from typing import Any, Sequence

import torch
from torch import nn

from .rollout import RD  # train_interface_v1/render.py (frozen, import only)

PAD, EOS, IMG, IM_START = "<|pad|>", "<|im_end|>", "<|image_pad|>", "<|im_start|>"
SPECIALS = (PAD, EOS, IMG, IM_START, "<memory_call>", "</memory_call>")
N_IMG_TOKENS = 4  # mock "expanded" image placeholders
_PIECE = re.compile("|".join(re.escape(s) for s in SPECIALS) + r"|<[^<>\s]{1,24}>|\w+|\s|[^\w\s]")


class MockTokenizer:
    """Lossless regex-piece tokenizer with a growing vocabulary (bounded by ``vocab_size``)."""

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.id2piece: list[str] = list(SPECIALS)
        self.piece2id = {p: i for i, p in enumerate(self.id2piece)}
        self.pad_token_id, self.eos_token_id, self.image_token_id = 0, 1, 2

    def encode(self, text: str) -> list[int]:
        pieces = _PIECE.findall(text)
        if "".join(pieces) != text:
            raise ValueError("mock tokenizer is not lossless on this text")
        out = []
        for p in pieces:
            if p not in self.piece2id:
                if len(self.id2piece) >= self.vocab_size:
                    raise ValueError("mock vocabulary exhausted")
                self.piece2id[p] = len(self.id2piece)
                self.id2piece.append(p)
            out.append(self.piece2id[p])
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return "".join(self.id2piece[i] for i in ids if not (skip_special and self.id2piece[i] in (PAD, EOS, IMG)))

    def render(self, messages: list[dict], *, add_generation_prompt: bool) -> str:
        """ChatML-like template; an image part becomes ``N_IMG_TOKENS`` placeholders."""
        out = []
        for m in messages:
            content = m["content"]
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            body = "".join(IMG * N_IMG_TOKENS if p["type"] == "image" else p["text"] for p in parts)
            out.append(f"{IM_START}{m['role']}\n{body}{EOS}\n")
        if add_generation_prompt:
            out.append(f"{IM_START}assistant\n")
        return "".join(out)


class MockPolicy(nn.Module):
    """Embedding -> GRU -> linear: causal (position t sees tokens <= t), CPU-fast."""

    def __init__(self, vocab_size: int, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.emb(ids))
        return self.head(h)


def token_logprobs(policy: MockPolicy, prompt: Sequence[int], completion: Sequence[int]) -> list[float]:
    """Log-probability of each completion token under ``policy`` (no grad)."""
    if not completion:
        return []
    with torch.no_grad():
        ids = torch.tensor([list(prompt) + list(completion)])
        logp = torch.log_softmax(policy(ids)[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        lp = logp.gather(-1, tgt[:, None])[:, 0]
    return lp[len(prompt) - 1:].tolist()


STAGE1_CHOICES = ("<memory_call>call_pdb</memory_call>", "<memory_call>direct_answer</memory_call>",
                  "This may depend on the person's allergies, so I will check memory. <memory_call>call_pdb</memory_call>",
                  "I think I should call the memory tool first please")  # last one is malformed; padded over the cap


class ScriptedBackend:
    """Mock rollout backend implementing ``rollout.Backend`` (TRL VLLMGeneration call semantics)."""

    def __init__(self, tokenizer: MockTokenizer, policy: MockPolicy, seed: int, stage2_texts: Sequence[str],
                 stage1_cap: int) -> None:
        self.tok, self.policy, self.rng, self.stage1_cap = tokenizer, policy, random.Random(seed), stage1_cap
        self.stage2_texts = list(stage2_texts)
        self.eos_token_id = tokenizer.eos_token_id
        self.calls: list[dict] = []

    def encode_prompt(self, messages: list[dict], image: Any) -> list[int]:
        return self.tok.encode(self.tok.render(messages, add_generation_prompt=True))

    def encode_env_turn(self, env_user_message: dict, *, stage1_ended_with_eos: bool) -> list[int]:
        sentinel = "\x00SENTINEL\x00"
        text = self.tok.render([{"role": "assistant", "content": sentinel}, env_user_message],
                               add_generation_prompt=True).split(sentinel, 1)[1]
        if not text.startswith(EOS):
            raise AssertionError("env turn must start by closing the assistant turn")
        return self.tok.encode(text[len(EOS):] if stage1_ended_with_eos else text)

    def generate(self, prompts: list[list[int]], images: list, *, num_generations: int,
                 max_tokens: int) -> tuple[list[list[int]], list[list[float]]]:
        if len(prompts) % num_generations:
            raise ValueError("prompts must hold num_generations contiguous copies")
        self.calls.append({"n_prompts": len(prompts), "num_generations": num_generations, "max_tokens": max_tokens,
                           "prompt_lens": [len(p) for p in prompts]})
        outs, lps = [], []
        for p in prompts:
            stage1 = self._is_stage1(p)
            if stage1 and max_tokens != self.stage1_cap:
                raise AssertionError("Stage-1 prompt sampled without the Stage-1 cap")
            text = self.rng.choice(STAGE1_CHOICES if stage1 else self.stage2_texts)
            if stage1 and text == STAGE1_CHOICES[-1]:
                text += " please" * self.stage1_cap  # tag-less reasoning that runs past any cap
            ids = self.tok.encode(text) + [self.eos_token_id]
            ids = ids[:max_tokens]  # generation stops at the cap, as in vLLM (no EOS then)
            outs.append(ids)
            lps.append(token_logprobs(self.policy, p, ids))
        return outs, lps

    def _is_stage1(self, prompt: Sequence[int]) -> bool:
        """Stage 1 iff the last user turn holds the Stage-1 instruction (``max_tokens`` cannot tell: cap == 512)."""
        last_user = self.tok.decode(prompt).rsplit(f"{IM_START}user\n", 1)[-1]
        return RD.STAGE1_INSTRUCTION in last_user

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids, skip_special=True)
