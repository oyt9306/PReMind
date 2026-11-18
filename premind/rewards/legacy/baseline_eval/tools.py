"""Crop/zoom tool, tool-call parsing, and the shared image preprocessing."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image

from premind.rewards.legacy.baseline_eval.config import INFER

_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNC_RE = re.compile(r"<function=([\w\-]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w\-]+)>\s*(.*?)\s*</parameter>", re.S)


def preprocess(img: Image.Image) -> Image.Image:
    """Identical resize rule for query images and crops: keep aspect, bound pixel count."""
    img = img.convert("RGB")
    w, h = img.size
    area = w * h
    scale = 1.0
    if area > INFER.max_pixels:
        scale = math.sqrt(INFER.max_pixels / area)
    elif area < INFER.min_pixels:
        scale = math.sqrt(INFER.min_pixels / area)
    if scale != 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return img


def parse_tool_call(text: str) -> tuple[dict | None, str | None]:
    """Return (call, error). call = {"name", "arguments"}; (None, None) if no tool call present."""
    m = _CALL_RE.search(text)
    if not m:
        return None, None
    body = m.group(1).strip()
    f = _FUNC_RE.search(body)
    if f:
        args = {}
        for k, v in _PARAM_RE.findall(f.group(2)):
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
        return {"name": f.group(1), "arguments": args}, None
    try:  # JSON fallback: {"name": ..., "arguments": {...}}
        obj = json.loads(body)
        return {"name": obj.get("name"), "arguments": obj.get("arguments") or {}}, None
    except json.JSONDecodeError:
        return {"name": None, "arguments": {}, "raw": body[:300]}, "unparsable_tool_call"


def validate_bbox(args: dict) -> tuple[list[float] | None, str | None]:
    b = args.get("bbox_2d")
    if isinstance(b, str):
        try:
            b = json.loads(b)
        except json.JSONDecodeError:
            return None, "bbox_not_json"
    if not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
        return None, "bbox_malformed"
    x1, y1, x2, y2 = (float(v) for v in b)
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1000.0, x2), min(1000.0, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, "bbox_empty_or_inverted"
    return [x1, y1, x2, y2], None


def to_pixels(norm: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = norm
    return [round(x1 / 1000 * w), round(y1 / 1000 * h), round(x2 / 1000 * w), round(y2 / 1000 * h)]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(src: Image.Image, px: list[int], out_path: Path) -> Image.Image:
    c = src.crop(tuple(px))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c.save(out_path)
    return preprocess(c)
