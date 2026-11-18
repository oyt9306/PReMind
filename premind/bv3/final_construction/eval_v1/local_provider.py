"""LOCAL OpenAI-compatible provider (vLLM ``/v1/chat/completions``) for ``--model api:local:<served-name>`` (EVAL ONLY).

Free: no key, ledger cost 0, not gated by --allow-provider / --budget-usd / --prereg-ack; requires an explicit
``--api-base`` (e.g. http://127.0.0.1:8001/v1) and ``--local-model-key`` (LOCAL_MODELS: model pin + image marker).
This module never starts a server: eval_v1/local_vllm_eval.sh does that under the scheduler.

Request = the SAME conversation as the API path (eval_v1/api_backend.py): system turn, user prompt, [assistant
Stage-1 text, user env turn]. Each user turn is ONE text part (the turn's text parts concatenated byte-identically,
with the model's image marker inserted at each image position) followed by its images as base64 data URLs in the
same order. Markers are numbered across the whole conversation (Phi-4: <|image_1|> .. <|image_9|>, image k ->
k-th image item), which is the model's native interleaving; vLLM (``--chat-template-content-format string``) then
finds every marker in the text and inserts nothing else (no "\\n" joins, no images moved to the front).
temperature 0 (greedy), max_tokens = the protocol limits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from eval_v1 import api_providers as AP

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_S = 900


@dataclass(frozen=True)
class LocalModel:
    hf_id: str
    revision: str
    params_b: float
    image_marker: Optional[str]   # "{n}" = 1-based image index in the conversation; None = server default placement
    served_as: str                # recommended served name (LoRA module name for Phi-4)
    note: str = ""


LOCAL_MODELS: dict[str, LocalModel] = {
    "phi4_mm": LocalModel(
        "microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81", 5.6, "<|image_{n}|>",
        "vision",
        "vLLM 0.16, vision LoRA <snapshot>/vision-lora (r=256, alpha=512) served as 'vision'; vLLM applies the "
        "LoRA to the language model only (vision-encoder LoRA modules ignored, server warning)"),
}


def local_model(key: Optional[str]) -> LocalModel:
    if key not in LOCAL_MODELS:
        raise SystemExit(f"REFUSED: --local-model-key must be one of {sorted(LOCAL_MODELS)} for api:local (got {key!r})")
    return LOCAL_MODELS[key]


def check_api_base(api_base: Optional[str]) -> str:
    if not api_base or not api_base.startswith(("http://", "https://")):
        raise SystemExit(f"REFUSED: api:local needs an explicit --api-base http://<host>:<port>/v1 (got {api_base!r})")
    return api_base.rstrip("/")


def openai_body(served: str, req: AP.ApiRequest, marker: Optional[str]) -> dict:
    """OpenAI chat-completions body of a provider-neutral request (module docstring)."""
    msgs: list[dict] = [{"role": "system", "content": req.system}] if req.system else []
    n = 0
    for role, parts in req.turns:
        if role == "assistant":
            if any(k != "text" for k, _ in parts):
                raise ValueError("assistant turns are text only")
            msgs.append({"role": "assistant", "content": "".join(v for _, v in parts)})
            continue
        text: list[str] = []
        content: list[dict] = []
        for k, v in parts:
            if k == "text":
                if marker and marker.split("{n}")[0] in v:
                    raise ValueError("prompt text already contains an image marker")
                text.append(v)
                if not marker:
                    content.append({"type": "text", "text": v})
                continue
            n += 1
            img = {"type": "image_url", "image_url": {"url": f"data:{AP._mime(v)};base64,{AP._b64(v)}"}}
            if marker:
                text.append(marker.format(n=n))
            content.append(img)
        if marker:
            content = [{"type": "text", "text": "".join(text)}, *content]
        msgs.append({"role": "user", "content": content})
    return {"model": served, "messages": msgs, "temperature": 0.0, "max_tokens": req.max_tokens}


def local_call(api_base: str, served: str, marker: Optional[str], req: AP.ApiRequest, *,
               post: Optional[Callable[..., Any]] = None) -> AP.ApiResponse:
    """One chat completion; 429 / 5xx / connection errors are retryable, anything else fails closed."""
    if post is None:
        import requests
        post = requests.post
    try:
        r = post(f"{api_base}/chat/completions", json=openai_body(served, req, marker), timeout=REQUEST_TIMEOUT_S)
    except OSError as e:          # requests.RequestException is an OSError (connection refused / reset / timeout)
        raise AP.RetryableError(str(e)) from e
    if r.status_code == 429 or r.status_code >= 500:
        raise AP.RetryableError(f"local HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:      # e.g. 400 prompt + max_tokens > --max-model-len: never skipped silently
        raise AP.ApiUnavailableError(f"local HTTP {r.status_code}: {r.text[:500]}")
    js = r.json()
    ch = js["choices"][0]
    usage = js.get("usage") or {}
    fin = str(ch.get("finish_reason"))
    return AP.ApiResponse(ch["message"].get("content") or "", int(usage.get("prompt_tokens", 0)),
                          int(usage.get("completion_tokens", 0)), fin == "stop", fin, {"id": js.get("id")})


def server_info(api_base: str, served: str, *, get: Optional[Callable[..., Any]] = None) -> dict:
    """/v1/models (served name must be listed; LoRA root / parent recorded) + /version (vLLM version)."""
    if get is None:
        import requests
        get = requests.get
    models = get(f"{api_base}/models", timeout=30).json().get("data", [])
    entry = next((m for m in models if m.get("id") == served), None)
    if entry is None:
        raise AP.ApiUnavailableError(f"served model {served!r} not listed by {api_base}/models: "
                                     f"{[m.get('id') for m in models]}")
    root = api_base[:-3] if api_base.endswith("/v1") else api_base
    try:
        version = get(f"{root}/version", timeout=30).json().get("version")
    except (OSError, ValueError) as e:
        logger.warning("could not read %s/version: %s", root, e)
        version = None
    return {"api_base": api_base, "served_model": served, "server_version": version,
            "served_entry": {k: entry.get(k) for k in ("id", "root", "parent", "max_model_len")},
            "all_served": [m.get("id") for m in models]}


def make_call(api_base: str, served: str, marker: Optional[str]) -> Callable[[AP.ApiRequest], AP.ApiResponse]:
    return lambda req: AP.with_retries(lambda: local_call(api_base, served, marker, req))
