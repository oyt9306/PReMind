"""wandb metric layout for the release-based RLVR runner (pure functions, CPU, no wandb import).

Namespaces (x axis ``train/step`` = optimizer steps done, so curves align across runs and resumes):
  train/reward/*    reward level, within-group spread, zero-variance groups, accumulated mean + EMA, success rate,
                    reward_v5 component means (``train/reward/comp/<name>``);
  train/policy/*    token entropy, advantage stats, grad norm, lr, PPO clip / IS-cap fractions, lengths, truncation;
  train/behavior/*  memory-call decisions, withhold / clarify / escalation, format / degenerate, action / risk / cite;
  train/opsd/*      gate rate, distilled tokens, distillation KL (+ lambda-weighted loss), gate reasons per hint type;
                    self teacher: train/opsd/self_teacher = 1 and teacher forward seconds;
  train/dapo/*      DAPO only: filtered zero-variance groups, resample rounds, prompts drawn, kept groups, overlong;
  val/*             periodic dev eval (metrics_v9 + analysis extras), capability summary val/cap/*,
                    K-retention val/kret/*;
  sys/*             rollout / train / step time, peak GPU memory, tokens per second.
Metrics whose value is undefined at a step (empty denominator) are omitted rather than logged as null.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

EMA_ALPHA = 0.1
SUCCESS_DEF = "episode passes every reward check (trainer_v2.opsd.failure_reasons == [])"
CAP_DEF = {"C1": "cite_f1 (identification rows)", "C2": "memory-call decision accuracy (memory_call arm)",
           "C3": "balanced withhold = (correct_withhold + 1 - false_withhold) / 2",
           "C4": "mean of risk_acc and hazard_first_rate (whichever are defined)"}
BEHAVIOR_FROM_METRICS = {"call_recall_required": "call_recall", "false_call_rate": "g0_call_rate",
                         "call_precision": "call_precision", "correct_withhold": "correct_withhold",
                         "false_withhold": "false_withhold", "withhold_precision": "withhold_precision",
                         "format_ok_rate": "format_rate", "degenerate_rate": "degenerate_rate",
                         "action_acc": "action_acc", "risk_acc": "risk_acc", "cite_f1": "cite_f1",
                         "cite_exact": "cite_exact", "wrong_concept_rate": "wrong_concept_rate",
                         "distractor_rate": "distractor_rate", "spurious_cite_rate": "spurious_cite_rate"}
VAL_FROM_SUMMARY = {"reward": "reward_mean", "action_acc": "action_acc", "risk_acc": "risk_acc", "cite_f1": "cite_f1",
                    "cite_exact": "cite_exact", "call_precision": "call_precision", "call_recall": "call_recall",
                    "false_call_rate": "g0_call_rate", "correct_withhold": "correct_withhold",
                    "false_withhold": "false_withhold", "withhold_precision": "withhold_precision",
                    "withhold_recall": "withhold_recall", "hazard_first_rate": "hazard_first_rate",
                    "wrong_concept_rate": "wrong_concept_rate", "pair_accuracy": "pair_accuracy",
                    "format_rate": "format_rate", "degenerate_rate": "degenerate_rate", "n": "n"}
KRET_FROM_SUMMARY = {"memory_follow_lenient": "memory_follow_lenient_rate", "memory_follow": "memory_follow_rate",
                     "action_acc": "action_acc", "decisive_fact_hit": "decisive_fact_hit_rate", "n": "n"}
BEST_KEYS = ("val/reward", "val/action_acc", "val/risk_acc")


def _mean(xs: Iterable[Any]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _put(out: dict, key: str, value: Any) -> None:
    """Numeric values only (bool -> 0/1); None is dropped."""
    if isinstance(value, bool):
        out[key] = int(value)
    elif isinstance(value, (int, float)):
        out[key] = value


def group_stats(rewards: list[float], g: int) -> tuple[Optional[float], Optional[float]]:
    """(mean within-group sample std, fraction of groups with zero reward variance) over contiguous groups of ``g``
    (the grouping of grpo_v2.grouped_advantages). A zero-variance group has all-zero advantages: no learning signal."""
    if g < 2 or not rewards or len(rewards) % g:
        return None, None
    groups = [rewards[i:i + g] for i in range(0, len(rewards), g)]
    stds = [statistics.stdev(x) for x in groups]
    return sum(stds) / len(stds), sum(max(x) - min(x) < 1e-9 for x in groups) / len(groups)


@dataclass
class Accumulator:
    """Running mean over all steps and an EMA (``alpha`` weight on the newest value; first value seeds the EMA)."""
    alpha: float = EMA_ALPHA
    n: int = 0
    total: float = 0.0
    ema: Optional[float] = None

    def update(self, x: float) -> tuple[float, float]:
        self.n += 1
        self.total += x
        self.ema = x if self.ema is None else self.alpha * x + (1 - self.alpha) * self.ema
        return self.total / self.n, self.ema


def reward_metrics(rec: dict, g: int, acc: Accumulator) -> dict:
    out: dict = {}
    eps = rec.get("episodes") or []
    _put(out, "train/reward/mean", rec.get("reward_mean"))
    std, zero = group_stats(list(rec.get("rewards") or []), g)
    _put(out, "train/reward/std_in_group", std)
    _put(out, "train/reward/zero_var_group_frac", zero)
    if rec.get("reward_mean") is not None:
        running, ema = acc.update(float(rec["reward_mean"]))
        _put(out, "train/reward/acc_mean", running)
        _put(out, "train/reward/ema", ema)
    _put(out, "train/reward/success_rate", _mean(e.get("success") for e in eps))
    for name, v in (rec.get("components") or {}).items():
        _put(out, f"train/reward/comp/{name}", v)
    return out


def policy_metrics(rec: dict, lr: Optional[float]) -> dict:
    out: dict = {}
    eps = rec.get("episodes") or []
    toks = [e for e in eps if e.get("entropy") is not None and e.get("n_model_tokens")]
    if toks:      # token-weighted mean over every model-generated token of the step
        _put(out, "train/policy/entropy", sum(e["entropy"] * e["n_model_tokens"] for e in toks)
             / sum(e["n_model_tokens"] for e in toks))
    adv = [float(e["advantage"]) for e in eps if e.get("advantage") is not None]
    if adv:
        _put(out, "train/policy/adv_mean", sum(adv) / len(adv))
        _put(out, "train/policy/adv_std", statistics.pstdev(adv))
        _put(out, "train/policy/adv_abs_mean", sum(map(abs, adv)) / len(adv))
    _put(out, "train/policy/grad_norm", rec.get("grad_norm"))
    _put(out, "train/policy/lr", lr)
    # GRPO: token-level means per episode; GSPO: per-sequence values (clip_frac = share of clipped sequences);
    # ratio_mean = policy ratio vs log pi_old (1 on the first update), is_ratio_mean = sampler-mismatch IS weight
    for key in ("clip_frac", "is_cap_frac", "is_ratio_mean", "ratio_mean"):
        _put(out, f"train/policy/{key}", _mean(e.get(key) for e in eps))
    _put(out, "train/policy/updates_per_batch", rec.get("updates_per_batch"))
    for u, gn in enumerate(rec.get("grad_norms") or []):
        _put(out, f"train/policy/grad_norm_u{u}", gn)
    lens = [e["n_model_tokens"] for e in eps if e.get("n_model_tokens") is not None]
    if lens:
        _put(out, "train/policy/completion_len_mean", sum(lens) / len(lens))
        _put(out, "train/policy/completion_len_max", max(lens))
    _put(out, "train/policy/truncation_rate", _mean(e.get("truncated") for e in eps))
    return out


def behavior_metrics(rec: dict) -> dict:
    out: dict = {}
    m, eps = rec.get("metrics") or {}, rec.get("episodes") or []
    if rec.get("arm") == "memory_call":
        _put(out, "train/behavior/memory_call_rate", _mean(d == "call_pdb" for d in rec.get("decisions") or []))
    for dst, src in BEHAVIOR_FROM_METRICS.items():
        _put(out, f"train/behavior/{dst}", m.get(src))
    if m.get("n"):
        _put(out, "train/behavior/withhold_rate", m.get("n_withhold_precision", 0) / m["n"])
    acts = [e.get("action_pred") for e in eps]
    if acts:
        for name, label in (("clarify_rate", "clarify"), ("abstain_rate", "abstain"),
                            ("escalation_rate", "safety_escalation")):
            _put(out, f"train/behavior/{name}", sum(a == label for a in acts) / len(acts))
    return out


def opsd_metrics(rec: dict, opsd_lambda: float, g: int, hints_by_row: Optional[dict[str, list[str]]] = None) -> dict:
    out: dict = {}
    _put(out, "train/opsd/gate_rate", rec.get("opsd_gate_rate"))
    eps = rec.get("episodes") or []
    gated = [e for e in eps if e.get("opsd_gate")]
    _put(out, "train/opsd/distilled_tokens", sum(e.get("n_distill_tokens") or 0 for e in eps))
    _put(out, "train/opsd/distill_kl", rec.get("kl_mean"))
    if rec.get("kl_mean") is not None:
        _put(out, "train/opsd/distill_loss", opsd_lambda * rec["kl_mean"])
    _put(out, "train/opsd/n_gated", len(gated))
    _put(out, "train/opsd/self_teacher", rec.get("opsd_self_teacher"))       # only logged for the self teacher
    _put(out, "train/opsd/teacher_s", rec.get("t_teacher_s"))
    if hints_by_row is None or not eps:
        return out
    rows, reasons = rec.get("rows") or [], rec.get("gate_reasons") or []
    for i, e in enumerate(eps):
        if not e.get("opsd_gate") or i // g >= len(rows):
            continue
        for h in hints_by_row.get(rows[i // g], []):
            out[f"train/opsd/{h}/gated"] = out.get(f"train/opsd/{h}/gated", 0) + 1
            for why in (reasons[i] if i < len(reasons) else []):
                out[f"train/opsd/{h}/reason_{why}"] = out.get(f"train/opsd/{h}/reason_{why}", 0) + 1
    return out


def dapo_metrics(rec: dict) -> dict:
    """train/dapo/*: dynamic-sampling counts and overlong shaping (only present for algo=dapo)."""
    out: dict = {}
    for key in ("filtered_groups", "resample_rounds", "prompts_drawn", "kept_groups"):
        _put(out, f"train/dapo/{key}", rec.get(f"dapo_{key}"))
    if rec.get("dapo_no_informative") is not None:
        out["train/dapo/no_informative"] = int(bool(rec["dapo_no_informative"]))
    _put(out, "train/dapo/overlong_penalty_mean", rec.get("overlong_penalty_mean"))
    return out


def sys_metrics(rec: dict, step_s: Optional[float] = None) -> dict:
    out: dict = {}
    eps = rec.get("episodes") or []
    t_r, t_t = rec.get("t_rollout_s"), rec.get("t_train_s")
    _put(out, "sys/rollout_s", t_r)
    _put(out, "sys/train_s", t_t)
    _put(out, "sys/step_s", step_s if step_s is not None else (t_r or 0) + (t_t or 0))
    _put(out, "sys/peak_mem_alloc_gb", rec.get("peak_mem_alloc_gb"))
    _put(out, "sys/peak_mem_reserved_gb", rec.get("peak_mem_reserved_gb"))
    gen = sum(e.get("n_model_tokens") or 0 for e in eps)
    seq = sum((e.get("n_prompt") or 0) + (e.get("n_completion") or 0) for e in eps)
    if t_r:
        _put(out, "sys/gen_tok_per_s", gen / t_r)
    if t_t:
        _put(out, "sys/train_tok_per_s", seq / t_t)
    return out


def train_payload(rec: dict, *, step: int, g: int, acc: Accumulator, lr: Optional[float], opsd_lambda: float,
                  hints_by_row: Optional[dict[str, list[str]]] = None, step_s: Optional[float] = None) -> dict:
    """Every train/* and sys/* metric of one step record (trainer_v2.run_v2.run_steps) at x = ``step``."""
    out = {"train/step": step}
    for part in (reward_metrics(rec, g, acc), policy_metrics(rec, lr), behavior_metrics(rec),
                 opsd_metrics(rec, opsd_lambda, g, hints_by_row), dapo_metrics(rec), sys_metrics(rec, step_s)):
        out.update(part)
    return out


def capability_summary(summary: dict, recs: list[dict]) -> dict:
    """val/cap/* headline per capability (definitions in CAP_DEF) plus their components."""
    out: dict = {}
    _put(out, "val/cap/C1", summary.get("cite_f1"))
    _put(out, "val/cap/C1_wrong_concept_rate", summary.get("wrong_concept_rate"))
    call = [r for r in recs if r.get("arm") == "memory_call"]
    _put(out, "val/cap/C2", _mean(r.get("decision") == ("call_pdb" if r.get("memory_required") else "direct_answer")
                                  for r in call))
    cw, fw = summary.get("correct_withhold"), summary.get("false_withhold")
    parts = [x for x in (cw, None if fw is None else 1 - fw) if x is not None]
    _put(out, "val/cap/C3", sum(parts) / len(parts) if parts else None)
    _put(out, "val/cap/C3_pair_accuracy", summary.get("pair_accuracy"))
    _put(out, "val/cap/C4", _mean([summary.get("risk_acc"), summary.get("hazard_first_rate")]))
    return out


def val_payload(summary: dict, recs: list[dict], *, step: int, kret: Optional[dict] = None) -> dict:
    """``summary`` = eval_v1.analysis.summary(dev records); ``kret`` = kret_stats.summary(K-retention records)."""
    out = {"train/step": step}
    for dst, src in VAL_FROM_SUMMARY.items():
        _put(out, f"val/{dst}", summary.get(src))
    out.update(capability_summary(summary, recs))
    for dst, src in KRET_FROM_SUMMARY.items():
        _put(out, f"val/kret/{dst}", (kret or {}).get(src))
    return out


@dataclass
class BestTracker:
    """Best value (max) of each BEST_KEYS metric and the step it occurred at (first occurrence wins ties)."""
    best: dict = field(default_factory=dict)

    def update(self, payload: dict) -> dict:
        for k in BEST_KEYS:
            v = payload.get(k)
            if v is not None and (k not in self.best or v > self.best[k][0]):
                self.best[k] = (v, payload["train/step"])
        return self.summary()

    def summary(self) -> dict:
        out: dict = {}
        for k, (v, s) in self.best.items():
            name = k.replace("/", "_")
            out[f"best/{name}"], out[f"best/{name}_step"] = v, s
        return out
