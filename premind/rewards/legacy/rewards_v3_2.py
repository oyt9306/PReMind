"""Additive v3.2: v3.1 M1 application reward, without twin in the core.

No trainer import hook. `rewards_v3.py` and historical selections stay untouched.
The inherited r_ownership twin penalty remains unresolved and train-blocking.
"""
from __future__ import annotations


from premind.rewards.legacy import rewards_v3 as R3  # noqa: E402
from premind.rewards.legacy import rewards_v3_1 as M1  # noqa: E402

VERSION = "v3.2-no-twin"
W = {k: v for k, v in M1.W.items() if k != "twin"}
PEN = dict(R3.PEN)
POSITIVE_WEIGHT_SUM = round(sum(W.values()), 10)
RLVR_FUNCS = [f for f in M1.RLVR_FUNCS if f is not R3.r_twin]
CORE_COMPONENT_NAMES = tuple(f.__name__ for f in RLVR_FUNCS)
TWIN_CORE_CONTRIBUTION = 0.0

assert POSITIVE_WEIGHT_SUM == 0.88
assert len(RLVR_FUNCS) == len(M1.RLVR_FUNCS) - 1
assert R3.r_twin in M1.RLVR_FUNCS and R3.r_twin not in RLVR_FUNCS
assert len(set(CORE_COMPONENT_NAMES)) == len(CORE_COMPONENT_NAMES)
assert W.keys() == M1.W.keys() - {"twin"}
assert all(W[k] == M1.W[k] for k in W) and PEN == R3.PEN


def diagnostic_r_twin(completions, *, enable_diagnostic: bool = False, **kwargs) -> list[float]:
    """Legacy pair-aware score for isolated diagnostics; never in RLVR_FUNCS."""
    if enable_diagnostic is not True:
        raise RuntimeError("legacy twin reward is diagnostic-only and disabled by default")
    return R3.r_twin(completions, **kwargs)


def core_accounting(completions, **kwargs) -> dict[str, list[float]]:
    """Evaluate deterministic core components on CPU; excludes judge network call."""
    out = {f.__name__: f(completions, **kwargs) for f in RLVR_FUNCS}
    assert R3.r_twin.__name__ not in out
    assert all(len(v) == len(completions) for v in out.values())
    return out
