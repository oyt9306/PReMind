"""Path mapping for the research-snapshot data bundles.

The training and evaluation code in ``premind/core`` and ``premind/{training/cprime,evaluation/paper}`` is the code that
produced the paper. Its data files (releases, manifests, preregistrations, judge outputs) are shipped byte-identical, so
their integrity hashes still verify, and some of them store the absolute paths of the original machine. ``install()``
maps those original prefixes to the local bundle at file-system call time (``open``, ``os.stat``, ``os.listdir``, ...).
Path strings are left unchanged; Path.resolve() / os.path.realpath return the local path for both forms, so
resolved-path comparisons in the original code keep working.

The prefix table ships with each bundle as ``core/PATHMAP.json``: ``{"<original prefix>": "<path relative to premind/>"}``.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_MAP: list[tuple[str, str]] = []


def _load() -> list[tuple[str, str]]:
    table: dict[str, str] = {}
    for p in sorted((ROOT / 'core').glob('PATHMAP*.json')):
        table.update(json.loads(p.read_text()))
    out = []
    for src, dst in table.items():
        dst = re.sub(r'\$\{([A-Z0-9_]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), dst)
        if '${' in dst:
            continue
        out.append((src.rstrip('/'), str(dst if os.path.isabs(dst) else ROOT / dst).rstrip('/')))
    return sorted(out, key=lambda x: -len(x[0]))


def remap(path):
    """Return the local path for an original absolute path (str, bytes or PathLike); other values pass through."""
    if isinstance(path, int) or not _MAP:
        return path
    try:
        s = os.fspath(path)
    except TypeError:
        return path
    if isinstance(s, bytes):
        r = remap(s.decode('utf-8', 'surrogateescape'))
        return r.encode('utf-8', 'surrogateescape') if isinstance(r, str) else path
    for src, dst in _MAP:
        if s == src or s.startswith(src + '/'):
            return dst + s[len(src):]
    return path


def _wrap(fn):
    def inner(path, *a, **k):
        return fn(remap(path), *a, **k)
    inner.__wrapped__ = fn
    return inner


def install() -> int:
    """Install the mapping once; returns the number of mapped prefixes."""
    global _MAP
    if getattr(builtins.open, '__wrapped__', None) is not None:
        return len(_MAP)
    _MAP = _load()
    if not _MAP:
        return 0
    builtins.open = _wrap(builtins.open)
    io.open = builtins.open
    io.open_code = _wrap(io.open_code)
    os.path.realpath = _wrap(os.path.realpath)   # Path.resolve(): original and local paths compare equal
    for name in ('stat', 'lstat', 'open', 'listdir', 'scandir', 'access', 'readlink', 'mkdir', 'utime', 'chmod'):
        if hasattr(os, name):
            setattr(os, name, _wrap(getattr(os, name)))
    return len(_MAP)
