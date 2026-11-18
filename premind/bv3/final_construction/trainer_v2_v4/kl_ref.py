"""KL-to-reference term for the PReMind v4 recipe (WP-B item 1).

Reference policy: the frozen base model = the same PEFT model with the LoRA adapters DISABLED
(``backend.logits_at(..., teacher=True)`` -> ``HFBackend._adapter_off`` -> ``model.disable_adapter()``; the mock
backend's ``base`` copy in CPU tests). For a warm-start run (INIT_LORA) the reference is still the base model, not the
warm-start policy: the term is there to retain general ability, i.e. to stay near the zero-shot model.

Estimator: Schulman's k3, per model token t (Stage-1 + Stage-2 / passive tokens; env tokens excluded, as the RL term)
    k3_t = exp(ref_t - logp_t) - (ref_t - logp_t) - 1   >= 0,
the estimator of TRL GRPOTrainer and of CoViP's own trainer (CoViP-main/VLM-R1-Qwen3/src/open-r1-multimodal/src/
open_r1/trainer/grpo_trainer.py L783-787, default beta 0.04 in trainer/grpo_config.py L291-292). It is added to the
token loss inside the same aggregation as the RL term: per-sequence token mean (GRPO / GSPO) or token-level weights
(DAPO: the episode's rl_weight multiplies RL + beta * KL). Gradient: d k3 / d logp = 1 - exp(ref - logp), zero at
logp = ref. With mu = 2 the second update reuses the rollout of pi_old (as TRL).
Cost: ref log-probs are computed ONCE per episode, right after its rollout (no grad, one forward of the frozen model
over prompt + completion), and reused by every update; that is one extra no-grad forward per episode = about a third
of the policy forward+backward, i.e. +10-15 % of the training phase and ~ +4-6 % of a 4B step (rollout dominates:
354 s/step). Memory: logits at the model positions only (logits_to_keep), freed after the gather.
"""
from __future__ import annotations

from typing import Any, Optional

ESTIMATORS = ("k3",)


def k3(logp: Any, ref: Any) -> Any:
    """Per-token k3 (tensor), logp with grad, ref detached."""
    import torch
    d = ref - logp
    return torch.exp(d) - d - 1.0


def token_logp(logits: Any, targets: Any) -> Any:
    """log softmax(logits)[targets] without materialising a second [T, V] tensor in the forward."""
    import torch
    return logits.gather(-1, targets[:, None])[:, 0] - torch.logsumexp(logits, -1)


def ref_logp(ep: Any, backend: Any, images: Any) -> Any:
    """Reference (adapter-off) log-probs of the episode's model tokens, detached, on CPU float32."""
    import torch
    from trainer_v2 import grpo_v2 as G
    seq, at = G.policy_inputs(ep)
    pos = G.model_positions(ep)
    if not pos:
        return torch.zeros(0)
    with torch.no_grad():
        logits = backend.logits_at(seq, at, images, teacher=True).float()
        tgt = torch.tensor([ep.completion[i] for i in pos], device=logits.device)
        out = token_logp(logits, tgt).detach().cpu()
    del logits
    return out


def kl_term(ep: Any, logits: Any, ref: Any) -> tuple[Any, float]:
    """(mean k3 over the episode's model tokens (with grad), its float value)."""
    import torch
    from trainer_v2 import grpo_v2 as G
    pos = G.model_positions(ep)
    if not pos:
        z = logits.sum() * 0.0
        return z, 0.0
    tgt = torch.tensor([ep.completion[i] for i in pos], device=logits.device)
    logp = token_logp(logits, tgt)
    kl = k3(logp, ref.to(logits.device)).mean()
    return kl, float(kl.detach())


def validate(beta: float, estimator: str = "k3") -> None:
    if not (beta >= 0.0) or estimator not in ESTIMATORS:
        raise ValueError(f"kl beta must be >= 0 and estimator in {ESTIMATORS} (got {beta}, {estimator})")


def weighted(rl_weight: Optional[float]) -> float:
    return 1.0 if rl_weight is None else float(rl_weight)
