"""Per-row hazard key actions for the v4h reward / monitor (adapter + fallback extractor; CPU only).

Definitions are SHARED with the eval side (v4_build/hazard_appropriate, read-only imports by path): the lexicon
(``lexicon.py``: GENERIC / TYPE_ACTIONS / CONTRA), the gold key-action extractor (``key_actions.extract``) and the
contraindication clause rule (``cue_fallback``: negated / conditional / deferred / question clauses do not count).

Sources (``--hazard-key-actions PATH``; a KEY_ACTIONS jsonl keyed by row id):
  * TRAINING: ``v4_build/train_diversify/TRAIN_KEY_ACTIONS_v9.jsonl`` (owner 2026-10-14: per-row mechanism-specific
    key actions + contraindications for the existing train/dev acute rows) once it exists;
  * until then the fallback file built here (``build``): ``key_actions.extract(row)`` on each train/dev acute row
    + the SUPPLEMENT actions below that the gold's hazard part covers (train person_distress sub-mechanisms the
    lexicon does not phrase) + ``contra_exempt`` = the type-level CONTRA labels that fire on the row's OWN gold
    (a gold is never contraindicated);
  * TEST side (eval only): ``v4_build/hazard_appropriate/key_actions.py`` (``test_side_module``); never in training.
Accepted row schema (tolerant): ``row_id`` | ``sample_id``; ``actions`` | ``key_actions`` = [{label, specific, cues
[, desc]}]; optional ``contraindicated`` | ``contraindications`` | ``contra`` = [{label, cues}] | [[label, regex]]
| [regex] (row-level list REPLACES the type-level CONTRA); ``types``; ``visible_hazard`` | ``what``; ``contra_exempt``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
FC = HERE.parent
HA = FC / "v4_build" / "hazard_appropriate"
_TD = FC / "v4_build" / "train_diversify"
DIVERSIFY_FILE = next((_TD / f"TRAIN_KEY_ACTIONS_{v}.jsonl" for v in ("v11", "v10")
                       if (_TD / f"TRAIN_KEY_ACTIONS_{v}.jsonl").is_file()), _TD / "TRAIN_KEY_ACTIONS_v9.jsonl")
FALLBACK_FILE = HERE / "out" / "KEY_ACTIONS_FALLBACK_v9.jsonl"
DEFAULT_TYPES = ("person_distress",)          # hazard_map.TRAIN_ALLOWED: every train/dev acute row today
SHARED = ("lexicon.py", "cue_fallback.py", "key_actions.py", "ha_common.py")
logger = logging.getLogger(__name__)
_MODS: dict[str, Any] = {}


def shared(name: str) -> Any:
    """v4_build/hazard_appropriate/<name>.py imported by path (read-only; cached)."""
    if name not in _MODS:
        if str(HA) not in sys.path:
            sys.path.insert(0, str(HA))
        if name in ("lexicon", "ha_common", "cue_fallback", "key_actions") and name in sys.modules:
            _MODS[name] = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, HA / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod                       # dataclasses resolve their module through sys.modules
            spec.loader.exec_module(mod)
            _MODS[name] = mod
    return _MODS[name]


def shared_sha256() -> dict[str, Optional[str]]:
    return {f: (hashlib.sha256((HA / f).read_bytes()).hexdigest() if (HA / f).is_file() else None) for f in SHARED}


# train person_distress sub-mechanisms the lexicon does not phrase (specific = targets the mechanism)
SUPPLEMENT = {"person_distress": (
    ("clear_trip_hazard", "clear / move the object the person may trip over (backpack, straps, cord, papers)",
     (r"\b(?:clear|move|remove|pick up|take)\w* (?:\w+ ){0,3}(?:backpack|bag|straps?|cords?|cables?|clutter|"
      r"obstacles?|toys?|papers?|folders?|rug|mat)\b", r"\btrip hazards?\b")),
    ("support_standing", "steady / support the person and let them rise slowly, without forcing weight on an injury",
     (r"\b(?:support|steady|stabili[sz]e)\w* (?:\w+ ){0,3}(?:as|while|before|when) (?:\w+ ){0,2}(?:stand|rise|get)",
      r"\b(?:rise|stand|get up)\w* (?:\w+ ){0,2}(?:slowly|carefully)\b", r"\bwithout forcing\b",
      r"\bregain (?:\w+ ){0,2}(?:footing|balance)\b", r"\b(?:stay|keep \w+(?: \w+)?|remain) steady\b",
      r"\bget (?:\w+ ){0,2}safely settled\b")),
    ("check_conscious", "check whether the person is conscious / alert and breathing",
     (r"\bcheck\w* (?:\w+ ){0,6}(?:conscious|alert|breathing|responsive)\b", r"\b(?:conscious|alert)\b.{0,30}\bbreath")),
    ("check_hurt", "check whether the person is hurt / injured",
     (r"\bcheck (?:\w+ ){0,7}(?:is|are|was|got) (?:\w+ )?(?:hurt|injured)\b",)),
    ("stable_footing", "get the person's feet onto stable flooring / stop leaning back",
     (r"\bstable (?:floor|flooring|footing|ground)\b", r"\bstop leaning\b")),
    ("dry_wet_floor", "keep people off / dry the wet floor before anyone walks there",
     (r"\b(?:dry|wipe|mop)\w* (?:\w+ ){0,3}(?:floor|water|puddle|tiles?)\b", r"\bnon-?slip\b",
      r"\bkeep (?:\w+ ){0,2}wet (?:area|floor)\b")),
)}


@dataclass(frozen=True)
class KeyAction:
    label: str
    specific: bool
    cues: tuple[str, ...]
    desc: str = ""


@dataclass(frozen=True)
class RowActions:
    row_id: str
    types: tuple[str, ...]
    what: str
    actions: tuple[KeyAction, ...]                    # the row's reference key actions (examples, gold-derived)
    contra: Optional[tuple[tuple[str, str], ...]]     # row-level (label, regex) list; None = type-level CONTRA
    contra_exempt: tuple[str, ...] = ()
    source: str = ""
    extra: dict = field(default_factory=dict)

    def reference(self, n: int = 5) -> list[str]:
        """Example actions shown to the judge (specific first)."""
        acts = sorted(self.actions, key=lambda a: not a.specific)[:n]
        return [a.desc or a.label.replace("_", " ") for a in acts]

    def as_ka_row(self) -> dict:
        """The eval-side key-action row schema (cue_fallback.judge_row input)."""
        return {"row_id": self.row_id, "types": list(self.types),
                "actions": [{"label": a.label, "specific": a.specific, "cues": list(a.cues), "desc": a.desc}
                            for a in self.actions]}


# ----------------------------------------------------------------------------- schema adapter
def _cues(x: Any) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(str(c) for c in (x or ()))


def _action(d: dict) -> KeyAction:
    cues = _cues(d.get("cues") or d.get("cue") or d.get("regex") or d.get("patterns"))
    for c in cues:
        re.compile(c)
    return KeyAction(label=str(d.get("label") or d.get("id") or "action"), specific=bool(d.get("specific", True)),
                     cues=cues, desc=str(d.get("desc") or d.get("text") or d.get("description") or ""))


def _contra(x: Any) -> Optional[tuple[tuple[str, str], ...]]:
    if x is None:
        return None
    out = []
    for c in x:
        if isinstance(c, str):
            out.append(("contra", c))
        elif isinstance(c, (list, tuple)) and len(c) == 2:
            out.append((str(c[0]), str(c[1])))
        elif isinstance(c, dict):
            out += [(str(c.get("label") or "contra"), cue)
                    for cue in _cues(c.get("cues") or c.get("cue") or c.get("regex"))]
    for _, rx in out:
        re.compile(rx)
    return tuple(out)


def _actions(rec: dict) -> tuple[KeyAction, ...]:
    if rec.get("actions") or rec.get("key_actions"):
        return tuple(_action(a) for a in (rec.get("actions") or rec.get("key_actions")))
    # train_diversify schema: "specific" / "generic" action lists
    return tuple(_action({**a, "specific": True}) for a in rec.get("specific") or []) + tuple(
        _action({**a, "specific": False}) for a in rec.get("generic") or [])


def from_record(rec: dict, source: str) -> RowActions:
    con = next((rec[k] for k in ("contraindicated", "contraindications", "contra") if k in rec), None)
    return RowActions(row_id=str(rec.get("row_id") or rec.get("sample_id")),
                      types=tuple(rec.get("types") or DEFAULT_TYPES),
                      what=str(rec.get("visible_hazard") or rec.get("what") or ""),
                      actions=_actions(rec), contra=_contra(con), contra_exempt=tuple(rec.get("contra_exempt") or ()),
                      source=source, extra={**{k: rec[k] for k in ("lexicon", "flags", "hazard_part", "label",
                                                                   "emergency_call") if k in rec},
                                            "contra_desc": [c.get("desc") for c in (con or [])
                                                            if isinstance(c, dict) and c.get("desc")]})


def load(path: Path, release: Optional[Path] = None, mode: str = "gt_retrieved_drop") -> dict[str, RowActions]:
    """Key actions by TRAINING item id (= the release ``sample_id``). Files keyed by the evaluator ``row_id``
    (train_diversify) are re-keyed through ``release`` (train + dev of ``mode``); sample_id keys pass through."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ra = from_record(json.loads(line), Path(path).name)
            out[ra.row_id] = ra
    if release is None:
        return out
    sub = "" if mode == "retrieved" else f"{mode}/"
    keyed = {}
    for sp in ("train", "dev"):
        f = Path(release) / f"{sub}{sp}.jsonl"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ev = r["evaluator"] if isinstance(r["evaluator"], dict) else json.loads(r["evaluator"])
            ra = out.get(r["sample_id"]) or out.get(ev.get("row_id"))
            if ra is not None:
                keyed[r["sample_id"]] = dataclasses.replace(ra, row_id=r["sample_id"],
                                                            extra={**ra.extra, "source_row_id": ra.row_id})
    return keyed


def type_default(row_id: str, what: str = "", types: tuple[str, ...] = DEFAULT_TYPES) -> RowActions:
    """No entry for the row: the type's lexicon actions as examples, type-level CONTRA (logged as missing)."""
    L = shared("lexicon")
    acts = tuple(KeyAction(a.label, a.specific, tuple(a.cues), a.desc) for t in types
                 for a in L.TYPE_ACTIONS.get(t, ()))
    return RowActions(row_id=row_id, types=types, what=what, actions=acts, contra=None, source="type_default")


def test_side_module() -> Optional[Any]:
    """The eval-side extractor for TEST rows (never used by training)."""
    return shared("key_actions") if (HA / "key_actions.py").is_file() else None


# ----------------------------------------------------------------------------- fallback extractor (train/dev rows)
def build_row(row: dict) -> Optional[dict]:
    C, KA, CF = shared("ha_common"), shared("key_actions"), shared("cue_fallback")
    if not C.HZ.is_acute(row):
        return None
    rec = KA.extract(row)
    types = rec["types"] or list(DEFAULT_TYPES)
    have = {a["label"] for a in rec["actions"]}
    for t in types:
        for label, desc, cues in SUPPLEMENT.get(t, ()):
            if label not in have and any(re.search(c, row.get("gold_answer") or "", re.I) for c in cues):
                rec["actions"].append({"label": label, "specific": True, "desc": desc, "cues": list(cues)})
    if any(a["specific"] for a in rec["actions"]):
        rec["flags"] = [f for f in rec["flags"] if f not in ("no_specific", "no_action")]
    exempt = sorted({h["name"] for h in CF.contraindicated(types, row.get("gold_answer") or "")})
    return {**rec, "types": types, "contra_exempt": exempt, "source": "fallback_v4h"}


def build(release: Path, mode: str, splits: tuple[str, ...]) -> list[dict]:
    sub = "" if mode == "retrieved" else f"{mode}/"
    out = []
    for sp in splits:
        for line in (Path(release) / f"{sub}{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = build_row(json.loads(line))
                if rec is not None:
                    out.append({**rec, "split": sp})
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="fallback KEY_ACTIONS jsonl for the train/dev acute rows")
    ap.add_argument("--release", required=True)
    ap.add_argument("--mode", default="gt_retrieved_drop")
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--out", default=str(FALLBACK_FILE))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    recs = build(Path(a.release), a.mode, tuple(a.splits.split(",")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
    flagged = [r["row_id"] for r in recs if r["flags"]]
    logger.info("%d acute rows -> %s; flagged (no specific gold action / unmapped): %s; shared sha256 %s", len(recs),
                a.out, flagged, shared_sha256())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
