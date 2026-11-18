"""Row-level withholding metrics v5 = metrics_v4 + ONE rule change (owner 2026-10-14; eval side only, CPU only).

New module, overlay on eval_v1/metrics_v4 (not edited; not imported by the running trainings).
Note: final_construction/v4_build/test_validity_v1/METRICS_V5_NOTE.md.

metrics_v4 masks the risk check (risk counts as correct) on every target-absent row (memory_required AND withhold)
without a visible ACUTE hazard (``masked = absent and not t["acute_visual"]``, metrics_v4.row_values), so on such rows
joint = action only. v5 (``absent_visible_risk=True``, the default) scores risk (risk_pred == risk_label) on target-absent
rows whose IMAGE shows a visible hazard, i.e. image-level severity
``evaluator.acute_visual_hazard.image_verdict.severity`` in {"minor", "acute"} (minor: slippers on the walking path, a
cable on the floor, an exposed screwdriver). Rows with severity "none" (the risk depends only on the missing memory
fact) stay masked. Present rows, G0 rows and acute rows are unchanged; every other metric / population is identical
(summarize delegates to metrics_v4.summarize). ``absent_visible_risk=False`` reproduces metrics_v4 exactly.

Mechanics: v5 targets are metrics_v4 targets plus a ``metrics_v5`` block (rule flag, image severity, ``risk_unmasked``);
row_values / summarize require that block (ValueError on a plain v4 target) and hand metrics_v4 an effective copy with
``acute_visual`` = True on the unmasked rows (the only field metrics_v4 reads for masking). Returned targets keep the
original ``acute_visual``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from eval_v1 import metrics_v4 as M4
from eval_v1.metrics_v4 import METRICS, attach_risk_pred, load_release_rows

VISIBLE_SEVERITIES = ("minor", "acute")
RULE = "absent_visible_risk"


def image_severity(row_fields: dict) -> Optional[str]:
    """Image-level hazard severity of a release row (``evaluator``) or of an item's row fields (None if missing)."""
    a = row_fields.get("acute_visual_hazard")
    if a is None and isinstance(row_fields.get("evaluator"), dict):
        a = row_fields["evaluator"].get("acute_visual_hazard")
    return ((a or {}).get("image_verdict") or {}).get("severity")


def is_absent(t: dict) -> bool:
    return bool(t["memory_required"]) and bool(t["withhold"])


def masked_v4(t: dict) -> bool:
    return is_absent(t) and not t["acute_visual"]


def apply_v5(target: dict, severity: Optional[str], absent_visible_risk: bool = True) -> dict:
    """Copy of a metrics_v4 target with the v5 block (``risk_unmasked`` only if the rule is on)."""
    unmask = absent_visible_risk and masked_v4(target) and severity in VISIBLE_SEVERITIES
    return {**target, "metrics_v5": {RULE: bool(absent_visible_risk), "image_severity": severity,
                                     "risk_unmasked": bool(unmask)}}


def row_targets(release_rows: Iterable[tuple[dict, Path]], *, final_holdout: bool = True, arm: str = "memory_call",
                acute_rule: str = "uniform_addressed", absent_visible_risk: bool = True) -> dict[str, dict]:
    """metrics_v4.row_targets + the v5 block (severity from the release row's evaluator)."""
    rows = list(release_rows)
    base = M4.row_targets(rows, final_holdout=final_holdout, arm=arm, acute_rule=acute_rule)
    sev = {row["sample_id"]: image_severity(row) for row, _ in rows}
    return {sid: apply_v5(t, sev[sid], absent_visible_risk) for sid, t in base.items()}


def targets_from_items(items: Iterable[Any], acute_rule: str = "uniform_addressed",
                       absent_visible_risk: bool = True) -> dict[str, dict]:
    """metrics_v4.targets_from_items + the v5 block (severity from ``item.row``)."""
    items = list(items)
    base = M4.targets_from_items(items, acute_rule)
    sev = {it.row_id: image_severity(it.row) for it in items}
    return {sid: apply_v5(t, sev[sid], absent_visible_risk) for sid, t in base.items()}


def _effective(t: dict) -> dict:
    if "metrics_v5" not in t:
        raise ValueError("metrics_v5 needs v5 targets: metrics_v5.row_targets / targets_from_items / apply_v5")
    return {**t, "acute_visual": True} if t["metrics_v5"]["risk_unmasked"] else t


def row_values(r: dict, t: dict) -> dict[str, Optional[float]]:
    """metrics_v4.row_values with the v5 masking rule."""
    return M4.row_values(r, _effective(t))


def summarize(records: list[dict], targets: dict[str, dict], request_form: Optional[dict[str, str]] = None) -> dict:
    """metrics_v4.summarize with the v5 masking rule; adds n_risk_unmasked_v5 (records on unmasked rows)."""
    eff = {k: _effective(t) for k, t in targets.items()}
    out = M4.summarize(records, eff, request_form)
    out["n_risk_unmasked_v5"] = sum(bool(targets[r["row_id"]]["metrics_v5"]["risk_unmasked"]) for r in records)
    return out


__all__ = ["METRICS", "VISIBLE_SEVERITIES", "RULE", "image_severity", "is_absent", "masked_v4", "apply_v5",
           "row_targets", "targets_from_items", "attach_risk_pred", "row_values", "summarize", "load_release_rows"]
