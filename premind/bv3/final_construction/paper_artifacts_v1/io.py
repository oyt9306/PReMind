"""Read-only source loading, path guards, provenance, and write-once outputs."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SourceRefusal(ValueError):
    """A sealed source was supplied without explicit authorization."""


def guard(path: Path, *, final_holdout: bool = False) -> Path:
    path = Path(path)
    if not final_holdout and ("holdout" in str(path).lower() or "holdout" in str(path.resolve()).lower()):
        raise SourceRefusal(f"refused sealed path: {path}")
    return path


def sha12(path: Path, *, final_holdout: bool = False) -> str:
    path = guard(path, final_holdout=final_holdout)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def source(path: Path, *, final_holdout: bool = False) -> str:
    path = guard(path, final_holdout=final_holdout)
    return f"% source: {path.resolve()} sha256={sha12(path, final_holdout=final_holdout)}"


def read_json(path: Path, *, final_holdout: bool = False) -> dict:
    return json.loads(guard(path, final_holdout=final_holdout).read_text(encoding="utf-8"))


def read_jsonl(path: Path, *, final_holdout: bool = False) -> list[dict]:
    rows = []
    with guard(path, final_holdout=final_holdout).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def next_version(root: Path, *, final_holdout: bool = False) -> Path:
    root = guard(root, final_holdout=final_holdout)
    root.mkdir(parents=True, exist_ok=True)
    used = {int(p.name[1:]) for p in root.iterdir() if p.is_dir() and p.name.startswith("v")
            and p.name[1:].isdigit()}
    n = max(used, default=0) + 1
    out = root / f"v{n}"
    out.mkdir(exist_ok=False)
    return out


def tex_escape(value: Any) -> str:
    text = str(value)
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in text)


def fmt(value: Any, *, rate: bool = False) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(int(value))
    return f"{float(value):.3f}" if rate else str(value)


def metric(cell: dict, name: str) -> str:
    m = cell.get("metrics", {}).get(name)
    ci = cell.get("ci95", {}).get(name)
    if m is None:
        return "--"
    return f"{fmt(m, rate=True)} [{fmt(ci[0], rate=True)}, {fmt(ci[1], rate=True)}]" if ci else fmt(m, rate=True)


def write_tex(path: Path, body: str, sources: list[Path], *, final_holdout: bool = False) -> None:
    guard(path, final_holdout=final_holdout)
    if path.exists():
        raise FileExistsError(path)
    refs = [source(s, final_holdout=final_holdout) for s in dict.fromkeys(sources)]
    path.write_text(body.rstrip() + "\n" + "\n".join(refs) + "\n", encoding="utf-8")


def manifest(out: Path, generated: list[Path], sources: list[Path], *, final_holdout: bool = False) -> dict:
    return {"version": out.name,
            "files": [{"path": str(p.relative_to(out)), "sha256": sha12(p, final_holdout=final_holdout)}
                      for p in sorted(generated)],
            "sources": [{"path": str(p.resolve()), "sha256": sha12(p, final_holdout=final_holdout)}
                        for p in sorted(set(sources))]}
