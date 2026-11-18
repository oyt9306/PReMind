"""Closed-source provider clients for the API evaluation path (EVAL ONLY): request bodies, pricing, token estimates.

A request is provider-neutral (``ApiRequest``): the system text and the turns of the SAME conversation the Qwen
models see (eval_v1/api_backend.py builds it), each turn a list of ("text", str) / ("image", path) parts in order
(adjacent text parts merged, so the text is byte-identical). Images are sent as the release files' own bytes
(PNG / JPEG, inline base64), never re-encoded.

Decoding: temperature 0 where the model accepts it, thinking OFF where it can be switched off (MODEL_PROFILES);
models that cannot run without thinking (thinking tokens would share the max-output budget) are not registered and
refuse. OpenAI reasoning models (no none/minimal effort) are the exception: reasoning effort is a profile param
(selectable per run as ``<model>@<effort>``), max_output_tokens = visible cap + a reasoning allowance, and the
VISIBLE text is then truncated to the protocol cap counted with the Qwen3.5 tokenizer (truncate_visible), so the
visible budget equals the open models'. Real calls need: the provider's key in the environment (ANTHROPIC_API_KEY /
GEMINI_API_KEY / OPENAI_API_KEY; never read
from files or other credential sources), ``--allow-provider <name>`` (default: none allowed; owner 2026-10-05: no
Gemini spending), a budget cap and the prereg acknowledgement (api_eval.py). Retries: 429 / 5xx / connection
errors only, bounded exponential backoff.

PRICES / image-token rules are a snapshot - verify current prices before launch.
"""
from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "gemini", "openai", "local")   # local = free OpenAI-compatible server (local_provider.py)
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ANTHROPIC_IMAGE_MAX_BYTES = 5 * 2**20     # per image (base64 payload limit, documented API limit)
GEMINI_INLINE_MAX_BYTES = 20 * 2**20      # whole inline request
CHARS_PER_TOKEN = 3.0                     # conservative text estimate (English ~4 chars / token; newer Claude
                                          # tokenizers use up to ~1.35x more tokens than older ones)


class ApiUnavailableError(RuntimeError):
    """Provider not usable (unknown model, key absent, provider not allowed, refused request)."""


class RetryableError(RuntimeError):
    """429 / 5xx / connection error (retried with backoff)."""


@dataclass(frozen=True)
class Profile:
    provider: str
    usd_in_per_mtok: float     # verify current prices before launch
    usd_out_per_mtok: float
    image_rule: str            # "claude_1568" | "claude_2576" | "gemini_tiles" | "openai" | "phi4mm"
    params: tuple[tuple[str, Any], ...] = ()   # provider request knobs (temperature / thinking / reasoning)
    note: str = ""
    efforts: tuple[str, ...] = ()              # openai: reasoning efforts selectable as "<model>@<effort>"


# verify current prices before launch (snapshot 2026-10-05: Claude from the bundled claude-api skill table; Gemini
# from public list prices, <=200k-token prompts, standard tier - NOT re-checked online here).
MODEL_PROFILES: dict[str, Profile] = {
    "claude-sonnet-4-6": Profile("anthropic", 3.00, 15.00, "claude_1568", (("temperature", 0.0),),
                                 "temperature 0; thinking omitted = off on 4.6"),
    "claude-sonnet-5": Profile("anthropic", 2.00, 10.00, "claude_2576", (("thinking", {"type": "disabled"}),),
                               "sampling params rejected (default sampling); thinking disabled"),
    "claude-sonnet-5-5": Profile("anthropic", 2.00, 10.00, "claude_2576", (("thinking", {"type": "between_tools"}),),
                                 "sampling params rejected (default sampling); thinking off via between_tools"),
    "claude-haiku-4-5": Profile("anthropic", 1.00, 5.00, "claude_1568", (("temperature", 0.0),),
                                "temperature 0; no thinking"),
    "gemini-2.5-flash": Profile("gemini", 0.30, 2.50, "gemini_tiles",
                                (("temperature", 0.0), ("thinkingConfig", {"thinkingBudget": 0})),
                                "temperature 0; thinkingBudget 0 (off)"),
    "gemini-2.5-flash-lite": Profile("gemini", 0.10, 0.40, "gemini_tiles",
                                     (("temperature", 0.0), ("thinkingConfig", {"thinkingBudget": 0})),
                                     "temperature 0; thinkingBudget 0 (off)"),
}
# OpenAI reasoning models (Responses API, store=False): effort default medium (no none/minimal on gpt-6.1-sol; luna
# also run at medium for parity, and at high via "gpt-6-luna@high"). reasoning_allowance = hidden reasoning tokens
# added on top of the visible cap in max_output_tokens (popped from the body, kept in params so it enters the cache
# key and the prereg line). Prices: docs/BASELINE_CANDIDATE_SURVEY_2026-10-01.md, standard tier.
OPENAI_REASONING_ALLOWANCE = 8192
MODEL_PROFILES.update({
    "gpt-6.1-sol": Profile("openai", 2.00, 10.00, "openai",
                           (("reasoning", {"effort": "medium"}), ("reasoning_allowance", OPENAI_REASONING_ALLOWANCE)),
                           "Responses API, store=False; reasoning effort per run (medium default, @high; no none/minimal); "
                           "default sampling", ("medium", "high")),   # owner 2026-10-09: sol high added
    "gpt-6-luna": Profile("openai", 0.10, 0.50, "openai",
                          (("reasoning", {"effort": "medium"}), ("reasoning_allowance", OPENAI_REASONING_ALLOWANCE)),
                          "Responses API, store=False; reasoning effort per run (medium default, @high); "
                          "default sampling", ("medium", "high")),
})
# ESTIMATE-ONLY assumption (dry-run / prereg): hidden reasoning tokens per call by effort. NOT measured.
ASSUMED_REASONING_TOKENS = {"medium": 1500, "high": 4000}
OPENAI_IMAGE_TOKENS_1536x1024 = 1844       # UNVERIFIED (survey: not in OpenAI's sizing table); scaled by pixels
NOT_SUPPORTED = {
    "claude-opus-5-5": "thinking cannot be disabled (would consume the 512-token output budget)",
    "gemini-2.5-pro": "thinking cannot be disabled (min budget 128, counted in maxOutputTokens)",
    "gemini-3-pro-preview": "thinking cannot be disabled",
}


# api:local:<served> -> model "local:<served>": free (prices 0), temperature 0; Phi-4 image-token rule for estimates
LOCAL_PROFILE = Profile("local", 0.0, 0.0, "phi4mm", (("temperature", 0.0),),
                        "local OpenAI-compatible server (vLLM); free; temperature 0 (greedy)")


def profile(model: str) -> Profile:
    if model.startswith("local:") and len(model) > len("local:"):
        return LOCAL_PROFILE
    if "@" in model:                       # "<openai model>@<effort>": same profile, reasoning effort replaced
        base, effort = model.split("@", 1)
        prof = profile(base)
        if effort not in prof.efforts:
            raise ApiUnavailableError(f"model {base!r}: reasoning effort {effort!r} not allowed; allowed {prof.efforts}")
        prm = dict(prof.params, reasoning={"effort": effort})
        return dataclasses.replace(prof, params=tuple(prm.items()))
    if model in NOT_SUPPORTED:
        raise ApiUnavailableError(f"model {model!r} refused: {NOT_SUPPORTED[model]}")
    if model not in MODEL_PROFILES:
        raise ApiUnavailableError(f"model {model!r} has no registered profile/price; known {sorted(MODEL_PROFILES)}")
    return MODEL_PROFILES[model]


# ----------------------------------------------------------------------------- token / cost estimates
def image_tokens(rule: str, width: int, height: int) -> int:
    """Approximate input tokens of one image.

    claude_*: downscaled to fit the long-edge limit (1568 px; 2576 px / 3.75 MP on high-res models) and ~1.15 MP
    (older models), then ceil(w * h / 750), capped at 1600 / 4784 (documented caps).
    gemini_tiles: both sides <= 384 px -> 258; else 768x768 tiles x 258 (ceil per side after scaling the short side
    to 768); Gemini 3 media_resolution may differ, so a floor of 1120 per image is added as conservative bound."""
    if rule.startswith("claude"):
        edge, mp, cap = (1568, 1.15e6, 1600) if rule == "claude_1568" else (2576, 3.75e6, 4784)
        s = min(1.0, edge / max(width, height), math.sqrt(mp / (width * height)))
        return min(cap, math.ceil(width * s * height * s / 750))
    if rule == "gemini_tiles":
        if width <= 384 and height <= 384:
            return 1120
        s = 768 / min(width, height)
        tiles = math.ceil(width * s / 768) * math.ceil(height * s / 768)
        return max(1120, 258 * tiles)
    if rule == "phi4mm":
        return phi4mm_image_tokens(width, height)
    if rule == "openai":   # UNVERIFIED ~1,844 per 1536x1024 (detail high), scaled by pixel count
        return math.ceil(OPENAI_IMAGE_TOKENS_1536x1024 * width * height / (1536 * 1024))
    raise ValueError(rule)


def phi4mm_image_tokens(width: int, height: int, *, crop: int = 448, patch: int = 14, max_crops: int = 36) -> int:
    """Phi-4-multimodal image tokens (vLLM 0.16 Phi4MMProcessingInfo._compute_num_image_tokens, dynamic_hd=36):
    global 16x16 + 16 newlines + 1 separator, plus the non-padding HD crop grid (2x2-compressed) + 1 newline per row.
    448x448 -> 545; 1536x1024 -> 3068. Only the in-limit case (w_crops * h_crops <= 36) is exact."""
    wc, hc = math.ceil(width / crop), math.ceil(height / crop)
    if wc * hc > max_crops:
        raise ValueError(f"phi4mm estimate: {width}x{height} needs > {max_crops} crops (aspect-ratio search not modelled)")
    tw, th = crop * wc, crop * hc
    if tw / width < th / height:
        pad_w, pad_h = 0, th - int(height * (tw / width))
    else:
        pad_w, pad_h = tw - int(width * (th / height)), 0
    fw, fh = tw // patch, th // patch
    if pad_w >= patch:
        fw -= pad_w // patch
    elif pad_h >= patch:
        fh -= pad_h // patch
    fw, fh = math.ceil(fw / 2), math.ceil(fh / 2)
    g = crop // patch // 2
    return g * g + 1 + fw * fh + fh + g


def api_model_id(model: str) -> str:
    """Provider model id ("gpt-6-luna@high" -> "gpt-6-luna")."""
    return model.split("@", 1)[0]


def reasoning_allowance(model: str) -> int:
    return int(dict(profile(model).params).get("reasoning_allowance", 0))


def reasoning_effort(model: str) -> Optional[str]:
    r = dict(profile(model).params).get("reasoning")
    return r.get("effort") if isinstance(r, dict) else None


def assumed_reasoning_tokens(model: str) -> int:
    """ASSUMPTION for estimates only (ASSUMED_REASONING_TOKENS); 0 for non-reasoning profiles."""
    eff = reasoning_effort(model)
    return ASSUMED_REASONING_TOKENS[eff] if eff and reasoning_allowance(model) else 0


def text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p = profile(model)
    return in_tok * p.usd_in_per_mtok / 1e6 + out_tok * p.usd_out_per_mtok / 1e6


# ----------------------------------------------------------------------------- request
@dataclass(frozen=True)
class ApiRequest:
    system: str
    turns: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]   # (role "user"/"assistant", parts)
    max_tokens: int

    def images(self) -> list[str]:
        return [v for _, parts in self.turns for k, v in parts if k == "image"]


@dataclass
class ApiResponse:
    text: str
    input_tokens: int
    output_tokens: int          # visible output tokens (what n_stage2 counts)
    stopped: bool               # natural end of turn (not the max-token cap)
    raw_stop: str
    extra: dict = field(default_factory=dict)


def _mime(path: str) -> str:
    head = Path(path).read_bytes()[:8]
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    raise ApiUnavailableError(f"unsupported image format (PNG/JPEG only): {path}")


def _b64(path: str) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anthropic_body(model: str, req: ApiRequest) -> dict:
    def block(k: str, v: str) -> dict:
        if k == "text":
            return {"type": "text", "text": v}
        if Path(v).stat().st_size > ANTHROPIC_IMAGE_MAX_BYTES:
            raise ApiUnavailableError(f"image over the 5 MB per-image limit: {v}")
        return {"type": "image", "source": {"type": "base64", "media_type": _mime(v), "data": _b64(v)}}
    msgs = [{"role": role, "content": [block(k, v) for k, v in parts]} for role, parts in req.turns]
    return {"model": model, "max_tokens": req.max_tokens, "system": req.system, "messages": msgs,
            **dict(profile(model).params)}


def gemini_body(model: str, req: ApiRequest) -> dict:
    def part(k: str, v: str) -> dict:
        return {"text": v} if k == "text" else {"inline_data": {"mime_type": _mime(v), "data": _b64(v)}}
    prm = dict(profile(model).params)
    gen = {"maxOutputTokens": req.max_tokens, "candidateCount": 1, **prm}
    return {"systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [{"role": "model" if role == "assistant" else "user", "parts": [part(k, v) for k, v in parts]}
                         for role, parts in req.turns],
            "generationConfig": gen}


def openai_body(model: str, req: ApiRequest) -> dict:
    """Responses API body: instructions = system, input = the same turns (text / data-URL images, detail high, in
    order), store=False, reasoning effort from the profile, max_output_tokens = visible cap + reasoning allowance."""
    def part(k: str, v: str) -> dict:
        if k == "text":
            return {"type": "input_text", "text": v}
        return {"type": "input_image", "image_url": f"data:{_mime(v)};base64,{_b64(v)}", "detail": "high"}
    msgs: list[dict] = []
    for role, parts in req.turns:
        if role == "assistant":
            if any(k != "text" for k, _ in parts):
                raise ValueError("assistant turns are text only")
            msgs.append({"role": "assistant", "content": "".join(v for _, v in parts)})
        else:
            msgs.append({"role": "user", "content": [part(k, v) for k, v in parts]})
    prm = dict(profile(model).params)
    allowance = int(prm.pop("reasoning_allowance", 0))
    return {"model": api_model_id(model), "instructions": req.system, "input": msgs,
            "max_output_tokens": req.max_tokens + allowance, "store": False, **prm}


# ----------------------------------------------------------------------------- visible-output cap (Qwen3.5 tokens)
VISIBLE_COUNTER = "Qwen3.5 tokenizer.json (Qwen/Qwen3.5-4B @ trainer_v2.paths_v2.MODEL_REVISION, offline)"


@functools.lru_cache(maxsize=1)
def _qwen_tokenizer() -> Any:
    """The open models' tokenizer, loaded OFFLINE from the local HF cache snapshot (no hub access)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tokenizers import Tokenizer
    from trainer_v2 import paths_v2 as P
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
               / "hub")
    path = hub / f"models--{P.MODEL_ID.replace('/', '--')}" / "snapshots" / P.MODEL_REVISION / "tokenizer.json"
    if not path.is_file():
        raise ApiUnavailableError(f"Qwen3.5 tokenizer not in the local HF cache: {path}")
    return Tokenizer.from_file(str(path))


def truncate_visible(text: str, cap: int, *, tok: Any = None) -> tuple[str, int, bool]:
    """(text cut to <= cap Qwen3.5 tokens on a token boundary, its token count, truncated?) - the same visible
    budget the open models get (they stop after cap generated tokens)."""
    tok = tok or _qwen_tokenizer()
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) <= cap:
        return text, len(enc.ids), False
    k = cap
    while k > 0:
        cut = text[:enc.offsets[k - 1][1]]
        n = len(tok.encode(cut, add_special_tokens=False).ids)
        if n <= cap:
            return cut, n, True
        k -= 1
    return "", 0, True


# ----------------------------------------------------------------------------- clients
def require_key(provider: str) -> str:
    key = os.environ.get(KEY_ENV[provider], "")
    if not key:
        raise ApiUnavailableError(f"{KEY_ENV[provider]} is not set: {provider} evaluation unavailable (fail closed)")
    return key


def anthropic_call(model: str, req: ApiRequest, *, client: Any = None) -> ApiResponse:  # network unless mocked
    import anthropic
    client = client or anthropic.Anthropic(api_key=require_key("anthropic"), max_retries=0)
    try:
        msg = client.messages.create(**anthropic_body(model, req))
    except anthropic.RateLimitError as e:
        raise RetryableError(str(e)) from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise RetryableError(str(e)) from e
        raise
    except anthropic.APIConnectionError as e:
        raise RetryableError(str(e)) from e
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return ApiResponse(text, int(msg.usage.input_tokens), int(msg.usage.output_tokens),
                       msg.stop_reason == "end_turn", str(msg.stop_reason), {"id": getattr(msg, "id", None)})


def gemini_call(model: str, req: ApiRequest, *, post: Optional[Callable[..., Any]] = None) -> ApiResponse:
    if post is None:  # pragma: no cover - network
        import requests
        post = requests.post
    try:
        r = post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": require_key("gemini"),
                                                          "Content-Type": "application/json"},
                 data=json.dumps(gemini_body(model, req)), timeout=300)
    except OSError as e:
        raise RetryableError(str(e)) from e
    if r.status_code == 429 or r.status_code >= 500:
        raise RetryableError(f"gemini HTTP {r.status_code}")
    if r.status_code != 200:
        raise ApiUnavailableError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
    js = r.json()
    cand = (js.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    um = js.get("usageMetadata") or {}
    fin = str(cand.get("finishReason"))
    return ApiResponse(text, int(um.get("promptTokenCount", 0)), int(um.get("candidatesTokenCount", 0)),
                       fin == "STOP", fin, {"thoughts_tokens": int(um.get("thoughtsTokenCount", 0))})


OPENAI_TIMEOUT_S = 600.0
_OPENAI_TRANSIENT = ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError")


def _transient(e: BaseException) -> bool:
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status in (408, 429)):
        return True
    return isinstance(e, (TimeoutError, ConnectionError)) or any(c.__name__ in _OPENAI_TRANSIENT
                                                                 for c in type(e).__mro__)


def openai_call(model: str, req: ApiRequest, *, client: Any = None,
                truncate: Callable[[str, int], tuple[str, int, bool]] = truncate_visible) -> ApiResponse:
    """One Responses API call (network unless mocked). Visible text is truncated to req.max_tokens Qwen3.5 tokens;
    an incomplete response (max_output_tokens hit, e.g. by reasoning) keeps whatever visible text exists (logged).
    Cost accounting uses the API's usage: input_tokens, output_tokens (INCLUDES reasoning tokens)."""
    if client is None:  # pragma: no cover - network
        from openai import OpenAI
        client = OpenAI(api_key=require_key("openai"), max_retries=0, timeout=OPENAI_TIMEOUT_S)
    try:
        r = client.responses.create(**openai_body(model, req))
    except Exception as e:  # SDK error classes are classified by status / name (openai import stays optional)
        if _transient(e):
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        # owner 2026-10-13: a prompt rejected by the provider's usage-policy filter (HTTP 400, not billed) is recorded
        # as an empty provider refusal (scored like an unparseable output) instead of aborting the whole run.
        if getattr(e, "status_code", None) == 400 and "usage policy" in str(e):
            logger.warning("openai %s prompt rejected by the usage-policy filter; recorded as an empty refusal", model)
            return ApiResponse("", 0, 0, False, "policy_refused",
                               {"id": None, "actual_model": model, "status": "policy_refused", "incomplete_reason": None,
                                "reasoning_effort": reasoning_effort(model), "billed_input_tokens": 0,
                                "billed_output_tokens": 0, "reasoning_tokens": 0, "provider_visible_tokens": 0,
                                "visible_truncated": False, "visible_raw_chars": 0, "visible_counter": VISIBLE_COUNTER,
                                "usage": {}, "policy_refusal": str(e)[:300]})
        raise
    usage = r.usage
    ud = usage.model_dump() if hasattr(usage, "model_dump") else dict(vars(usage))
    in_tok, out_tok = int(usage.input_tokens), int(usage.output_tokens)
    details = getattr(usage, "output_tokens_details", None)
    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
    status = str(getattr(r, "status", None))
    inc = getattr(r, "incomplete_details", None)
    reason = getattr(inc, "reason", None) if inc is not None else None
    raw_text = r.output_text or ""
    text, n_vis, cut = truncate(raw_text, req.max_tokens)
    if status != "completed":
        logger.warning("openai %s response %s (%s): visible chars kept=%d, reasoning=%d, output=%d",
                       model, status, reason, len(text), reasoning, out_tok)
    if cut:
        logger.info("openai %s visible text truncated to %d Qwen3.5 tokens", model, req.max_tokens)
    stopped = status == "completed" and not cut
    raw_stop = "completed" if stopped else ("visible_cap" if status == "completed" else f"{status}:{reason}")
    return ApiResponse(text, in_tok, n_vis, stopped, raw_stop,
                       {"id": getattr(r, "id", None), "actual_model": getattr(r, "model", None), "status": status,
                        "incomplete_reason": reason, "reasoning_effort": reasoning_effort(model),
                        "billed_input_tokens": in_tok, "billed_output_tokens": out_tok, "reasoning_tokens": reasoning,
                        "provider_visible_tokens": out_tok - reasoning, "visible_truncated": cut,
                        "visible_raw_chars": len(raw_text), "visible_counter": VISIBLE_COUNTER, "usage": ud})


def with_retries(fn: Callable[[], ApiResponse], *, attempts: int = 5, base: float = 2.0, cap: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep) -> ApiResponse:
    for k in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if k == attempts - 1:
                raise
            delay = min(cap, base * 2 ** k)
            logger.warning("retryable API error (%s); attempt %d/%d, sleeping %.0fs", e, k + 1, attempts, delay)
            sleep(delay)
    raise AssertionError("unreachable")
