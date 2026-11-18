"""Fast causal-conv1d path for Qwen3.5 Gated DeltaNet layers, implemented with flash-linear-attention's triton kernels.

transformers 5.12 ``modeling_qwen3_5`` imports ``causal_conv1d_fn`` / ``causal_conv1d_update`` from the
``causal-conv1d`` package (not installed here, no wheel). Without them every Gated DeltaNet layer falls back to
torch: prefill ``F.silu(conv1d(x)[..., :T])``, decode ``torch_causal_conv1d_update``. fla 0.5.2 ships triton versions
with a different API; this module exposes the exact ``causal_conv1d`` package API that transformers calls and maps it
onto fla:

* ``causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, return_final_states=False,
  final_states_out=None, activation=None)``: x (batch, dim, seqlen), weight (dim, width) -> (batch, dim, seqlen).
  transformers calls it with keywords ``x, weight, bias, activation="silu", seq_idx`` (Qwen3_5GatedDeltaNet.forward,
  prefill / chunked continuation). fla ``causal_conv1d`` takes x as [B, T, D] and returns (y, final_state): the
  adapter transposes in and out (the transformers input is the transpose of a contiguous [B, T, D] projection, so
  ``.transpose(1, 2).contiguous()`` is free) and keeps autograd (fla's CausalConv1dFunction has a backward, needed
  because LoRA layers below the conv receive gradients through it). ``seq_idx`` (packed sequences, batch 1) maps to
  fla ``cu_seqlens``; initial / final states are not used by transformers and are refused.
* ``causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None,
  conv_state_indices=None)``: x (batch, dim) or (batch, dim, seqlen); conv_state (batch, dim, state_len) updated IN
  PLACE; returns the same shape as x. transformers calls it positionally ``(x, conv_state, weight, bias, activation)``
  with seqlen 1 for single-token cached decode, conv_state of width = kernel size (DynamicCache ``F.pad`` to
  ``conv_kernel_size``). fla ``causal_conv1d_update(x, cache, residual, weight, bias, activation)`` takes x (N, D) and
  a cache [N, D, W]: shift left by one, write x last, y = sum(cache * w) (+ bias, silu): the same new state and output
  as ``torch_causal_conv1d_update`` (cat(state, x), keep the last W, depthwise conv over them). fla writes the cache
  through a raw pointer after ``input_guard`` makes it contiguous, so a non-contiguous conv_state is updated on a
  contiguous copy and copied back (in-place semantics preserved either way).

``install(model_module, model)`` patches ``modeling_qwen3_5.causal_conv1d_fn`` / ``causal_conv1d_update`` (and its
``is_fast_path_available``) for layers built later, and the ``causal_conv1d_fn`` / ``causal_conv1d_update``
attributes of every already-constructed Gated DeltaNet layer of ``model``. Off by default (``fast_conv=False`` =
unchanged behaviour); the GPU runners turn it on with FAST_CONV=1 / ``--fast-conv``. fla's triton kernels need a CUDA
device: on CPU the CPU tests substitute ``_fla`` with reference implementations of fla's documented semantics; the
real-kernel parity runs only in eval_v1/fast_conv_smoke (scheduler GPU job).
"""
from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
ACTIVATIONS = (None, "silu", "swish")
MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"


def _fla() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """fla's (causal_conv1d, causal_conv1d_update); a separate function so CPU tests can substitute references."""
    from fla.modules.convolution import causal_conv1d, causal_conv1d_update
    return causal_conv1d, causal_conv1d_update


def _check(x: Any, weight: Any, activation: Optional[str], ndims: tuple[int, ...]) -> None:
    if activation not in ACTIVATIONS:
        raise NotImplementedError(f"activation must be one of {ACTIVATIONS} (got {activation!r})")
    if x.dim() not in ndims or weight.dim() != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(f"shapes: x {tuple(x.shape)} must be (batch, dim{', seqlen' if 3 in ndims else ''}) and "
                         f"weight {tuple(weight.shape)} (dim, width)")


def seq_idx_to_cu_seqlens(seq_idx: Any) -> Any:
    """(1, T) segment ids (non-decreasing, contiguous segments) -> fla cu_seqlens [0, ..., T] (int64)."""
    import torch
    if seq_idx.dim() != 2 or seq_idx.shape[0] != 1:
        raise NotImplementedError("seq_idx is only supported for batch 1 (packed sequences)")
    s = seq_idx[0]
    cuts = (torch.nonzero(s[1:] != s[:-1]).flatten() + 1).tolist()
    return torch.tensor([0] + cuts + [s.numel()], dtype=torch.long, device=seq_idx.device)


def causal_conv1d_fn(x: Any, weight: Any, bias: Any = None, seq_idx: Any = None, initial_states: Any = None,
                     return_final_states: bool = False, final_states_out: Any = None,
                     activation: Optional[str] = None) -> Any:
    """causal-conv1d package API on fla's triton ``causal_conv1d`` (see module docstring)."""
    _check(x, weight, activation, (3,))
    if initial_states is not None or return_final_states or final_states_out is not None:
        raise NotImplementedError("initial / final conv states are not used by transformers' Qwen3.5 path")
    conv, _ = _fla()
    cu = None
    if seq_idx is not None:
        cu = seq_idx_to_cu_seqlens(seq_idx)
        if len(cu) == 2:
            cu = None                                     # one segment = the plain causal conv
    y, _ = conv(x.transpose(1, 2).contiguous(), weight=weight, bias=bias, activation=activation, cu_seqlens=cu)
    return y.transpose(1, 2)


def causal_conv1d_update(x: Any, conv_state: Any, weight: Any, bias: Any = None, activation: Optional[str] = None,
                         cache_seqlens: Any = None, conv_state_indices: Any = None) -> Any:
    """causal-conv1d package API on fla's triton ``causal_conv1d_update``; conv_state is updated in place."""
    import torch
    _check(x, weight, activation, (2, 3))
    if cache_seqlens is not None or conv_state_indices is not None:
        raise NotImplementedError("cache_seqlens / conv_state_indices are not used by transformers' Qwen3.5 path")
    if conv_state.dim() != 3 or tuple(conv_state.shape[:2]) != tuple(x.shape[:2]):
        raise ValueError(f"conv_state {tuple(conv_state.shape)} must be (batch, dim, state_len) for x {tuple(x.shape)}")
    if conv_state.shape[-1] != weight.shape[1]:
        raise NotImplementedError(f"fla's cache holds exactly width={weight.shape[1]} columns "
                                  f"(got state_len={conv_state.shape[-1]})")
    _, upd = _fla()
    state = conv_state if conv_state.is_contiguous() else conv_state.contiguous()
    steps = [x] if x.dim() == 2 else [x[:, :, t] for t in range(x.shape[-1])]
    ys = []
    for xt in steps:
        y, _ = upd(xt, state, weight=weight, bias=bias, activation=activation)
        ys.append(y)
    if state is not conv_state:
        conv_state.copy_(state)
    return ys[0] if x.dim() == 2 else torch.stack(ys, dim=-1)


def _gdn_layers(model: Any) -> list:
    """Every module carrying the Gated DeltaNet conv hooks (Qwen3_5GatedDeltaNet)."""
    return [m for m in model.modules() if hasattr(m, "causal_conv1d_fn") and hasattr(m, "causal_conv1d_update")]


def install(model_module: Optional[ModuleType] = None, model: Any = None) -> int:
    """Patch the module-level imports and the already-built layers of ``model``; returns the number of layers
    patched. Idempotent. Logs which delta-rule kernels the layers use (fla vs torch fallback)."""
    if model_module is None:
        import importlib
        model_module = importlib.import_module(MODULE)
    if not hasattr(model_module, "_fast_conv_orig"):
        model_module._fast_conv_orig = (model_module.causal_conv1d_fn, model_module.causal_conv1d_update,
                                        model_module.is_fast_path_available)
    model_module.causal_conv1d_fn = causal_conv1d_fn
    model_module.causal_conv1d_update = causal_conv1d_update
    model_module.is_fast_path_available = all(getattr(model_module, n, None) is not None for n in (
        "causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"))
    layers = _gdn_layers(model) if model is not None else []
    for m in layers:
        m.causal_conv1d_fn = causal_conv1d_fn
        m.causal_conv1d_update = causal_conv1d_update
    rule = {str(getattr(getattr(m, "chunk_gated_delta_rule", None), "__module__", None)) for m in layers}
    logger.info("fast conv installed: %d Gated DeltaNet layers patched (fla triton causal_conv1d); module fast path=%s;"
                " delta-rule kernels=%s", len(layers), model_module.is_fast_path_available, sorted(rule))
    return len(layers)


def installed(model: Any) -> bool:
    """True iff every Gated DeltaNet layer of ``model`` uses this adapter (and there is at least one)."""
    layers = _gdn_layers(model)
    return bool(layers) and all(m.causal_conv1d_fn is causal_conv1d_fn and
                                m.causal_conv1d_update is causal_conv1d_update for m in layers)


def uninstall(model_module: Optional[ModuleType] = None, model: Any = None) -> int:
    """Back to transformers' own path (module imports as before ``install``; layers: package functions if they were
    importable, else the torch fallback). Used by eval_v1/fast_conv_smoke to compare both on one loaded model."""
    if model_module is None:
        import importlib
        model_module = importlib.import_module(MODULE)
    fn, upd, fast = getattr(model_module, "_fast_conv_orig", (None, None, False))
    model_module.causal_conv1d_fn, model_module.causal_conv1d_update = fn, upd
    model_module.is_fast_path_available = fast
    layers = _gdn_layers(model) if model is not None else []
    for m in layers:
        m.causal_conv1d_fn = fn
        m.causal_conv1d_update = upd or model_module.torch_causal_conv1d_update
    return len(layers)


def kernel_parity(device: str = "cuda", dim: int = 1024, seqlen: int = 37, steps: int = 8, seed: int = 0) -> dict:
    """Real-kernel parity of the adapter vs transformers' torch fallbacks on random tensors (GPU only: fla's triton
    kernels do not run on CPU). Per dtype: prefill output / input-gradient max abs diff and max |reference| (for a
    relative reading), decode output and conv-state max abs diff over ``steps`` cached steps, and whether the state
    was updated in place."""
    import torch
    import torch.nn.functional as F
    model_module = __import__(MODULE, fromlist=["torch_causal_conv1d_update"])
    out: dict = {}
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator(device=device).manual_seed(seed)
        w = (torch.randn(dim, 4, device=device, generator=g) * 0.5).to(dtype)
        x0 = torch.randn(3, seqlen, dim, device=device, generator=g).to(dtype).requires_grad_(True)
        x = x0.transpose(1, 2)                                     # (batch, dim, seqlen), as transformers passes
        ref = F.silu(F.conv1d(x, w.unsqueeze(1), None, padding=3, groups=dim)[..., :seqlen])
        got = causal_conv1d_fn(x=x, weight=w, bias=None, activation="silu", seq_idx=None)
        g_ref = torch.autograd.grad(ref.float().square().sum(), x0)[0]
        g_got = torch.autograd.grad(got.float().square().sum(), x0)[0]
        s_ref = torch.randn(3, dim, 4, device=device, generator=g).to(dtype)
        s_new, dec, st = s_ref.clone(), 0.0, 0.0
        ptr = s_new.data_ptr()
        for _ in range(steps):
            x1 = torch.randn(3, 1, dim, device=device, generator=g).to(dtype).transpose(1, 2)
            a = model_module.torch_causal_conv1d_update(x1, s_ref, w, None, "silu")
            b = causal_conv1d_update(x1, s_new, w, None, "silu")
            dec = max(dec, float((a.float() - b.float()).abs().max()))
            st = max(st, float((s_ref.float() - s_new.float()).abs().max()))
        out[str(dtype).removeprefix("torch.")] = {
            "prefill_max_abs": float((ref.float() - got.float()).abs().max()),
            "prefill_ref_max": float(ref.detach().abs().max()),
            "grad_max_abs": float((g_ref - g_got).abs().max()), "grad_ref_max": float(g_ref.abs().max()),
            "decode_out_max_abs": dec, "decode_state_max_abs": st, "state_in_place": s_new.data_ptr() == ptr}
    return out


def parity_ok(rep: dict) -> bool:
    """fp32 tight; bf16 within 2 % of the output scale (bf16 rounding of conv outputs); conv states copied exactly."""
    f, b = rep["float32"], rep["bfloat16"]
    return (f["prefill_max_abs"] <= 1e-4 and f["decode_out_max_abs"] <= 1e-4 and f["grad_max_abs"] <= 1e-3 * max(
        1.0, f["grad_ref_max"]) and b["prefill_max_abs"] <= 0.02 * max(1.0, b["prefill_ref_max"])
        and b["grad_max_abs"] <= 0.02 * max(1.0, b["grad_ref_max"]) and b["decode_out_max_abs"] <= 0.05
        and f["decode_state_max_abs"] == 0.0 == b["decode_state_max_abs"] and f["state_in_place"]
        and b["state_in_place"])
