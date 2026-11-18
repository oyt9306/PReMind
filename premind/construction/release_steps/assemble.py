"""G0 rewrite release steps (owner 2026-10-06 option A), one CLI for both passes. CPU only; no model, no API.
  --step v7g: RELEASE_TRAIN_DEV_v7f / RELEASE_HOLDOUT_v3f -> v7g / v3g: G0 rewrites (g0_rewrite_v1), generic text
              patches (inputs/text_qa_patches.jsonl: deterministic + text-only GPT answer repairs, and the memory-text
              fixes of the hallucination audit, which were added by hand) and the request_form label;
  --step v7h: v7g / v3g -> v7h / v3h: normative G0 request rewrites (g0_rewrite_v2; no text patches).
The rewrites / verify verdicts are the recorded outputs of the paid generation step (GPT rewrite + local Qwen image
check; inputs/g0_rewrite_v*/); they are not re-generated here.

1. G0 rewrites (per worklist item, key "<split>:<sample_id>"): the LATEST attempt whose verify row has pass=true and
   that has a rewrite row.
   * unsafe_image_answerable / safe_flip_to_no: new query_text + new gold_answer;
   * safe_styled_grounded_answer: new gold_answer only (the generic request is kept);
   * unsafe_image_answerable rewrites are no longer generic panic/calm requests: query_style (row and evaluator) is
     set to null like every other specific G0 request (reward_v5 scores the call decision of panic/calm G0 rows
     neutral);
   * the reference answer also lives in evaluator.legacy_arms.<retriever>.gold_answer: updated with gold_answer;
   * no passing attempt: row unchanged; unsafe_image_answerable rows get "g0_rewrite_status": "kept_generic";
   * implicit_keep_generic: no text change (marked by request_form only).
   Changed rows carry "g0_rewrite": {version, task, attempt, old_query, old_answer[, old_query_style]}.
2. Text patches: text_patches.apply (all-or-nothing per patch, pattern protection; see text_patches.py).
3. Every row of every file gets "request_form": "implicit" if its query_style is panic/calm, else "explicit".
Every change is applied to the row in EVERY mode file (retrieved, gt_retrieved/, gt_retrieved_drop/) of its split.
Rows are re-serialized as json.dumps(row, ensure_ascii=False) (the source format, checked line by line), so a row
untouched by 1-2 differs from its source line only by the appended request_form field. sample_ids, row order,
MEDIA_INDEX and ROW_PROVENANCE are unchanged. Write-once: refuses existing outputs.
G0 labels besides the answer: decisive_keyphrases are empty on all G0 rows; acute_visual_hazard.gold_pattern is a
reward pattern for the model answer (some quote the old answer): kept; the new answer is checked against it with
eval_v1.hazard_match (report only, like the leading-alarm check on safe answers).
    python -m premind.construction.release_steps.assemble --step v7g --src-td <v7f> --src-ho <v3f> --out-dir <dir>
    python -m premind.construction.release_steps.assemble --step v7h --src-td <v7g> --src-ho <v3g> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
from premind.rewards import hazard_match as HM  # noqa: E402
from premind.construction.release_steps import text_patches as TP  # noqa: E402

INPUTS = HERE / "inputs"
STEPS = {"v7g": {"version": "g0_rewrite_v1", "patches": INPUTS / "text_qa_patches.jsonl", "extra_tasks": (),
                 "names": ("RELEASE_TRAIN_DEV_v7g", "RELEASE_HOLDOUT_v3g")},
         "v7h": {"version": "g0_rewrite_v2", "patches": INPUTS / "no_patches.jsonl", "extra_tasks": ("g0_normative",),
                 "names": ("RELEASE_TRAIN_DEV_v7h", "RELEASE_HOLDOUT_v3h")}}
VERSION = "g0_rewrite_v1"
_TASKS = ("unsafe_image_answerable", "safe_relabel", "safe_flip_to_no", "safe_diversify", "safe_lure", "safe_styled_grounded_answer", "implicit_keep_generic")
_QUERY_TASKS = ("unsafe_image_answerable", "safe_relabel", "safe_flip_to_no", "safe_diversify", "safe_lure")
_RESTYLE_TASKS = ("unsafe_image_answerable", "safe_relabel")          # query_style -> null on rewrite
TASKS, QUERY_TASKS, RESTYLE_TASKS = _TASKS, _QUERY_TASKS, _RESTYLE_TASKS
NO_TEXT_TASKS = ("implicit_keep_generic",)
IMPLICIT_STYLES = ("panic", "calm")
COPY_AS_IS = ("MEDIA_INDEX.jsonl", "ROW_PROVENANCE.jsonl")
# MANIFEST fields computed from the v7f G0 query texts / answers; not recomputed here
STALE_MANIFEST_FIELDS = ("query_assignment", "shortcut_probe_all_g0_panic", "check12_target_met", "styled_safe_share",
                         "s2d_query_label_independence", "g0_safety_coverage")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def select(work: list[dict], rewrites: list[dict], verify: list[dict]) -> dict[str, dict]:
    """key -> decision {status: rewritten|kept_original|implicit_keep_generic, attempt, query, answer, reason}."""
    keys = {w["split"] + ":" + w["sample_id"] for w in work}
    if len(keys) != len(work):
        raise ValueError("duplicate worklist keys")
    unknown = {r["key"] for r in rewrites + verify} - keys
    if unknown:
        raise ValueError(f"rewrite/verify keys not in the worklist: {sorted(unknown)[:5]}")
    rew = {(r["key"], int(r["attempt"])): r for r in rewrites}       # a later line of the same attempt wins
    ver = {(r["key"], int(r["attempt"])): r for r in verify}
    out = {}
    for w in work:
        key = w["split"] + ":" + w["sample_id"]
        if w["task"] in NO_TEXT_TASKS:
            out[key] = {"status": w["task"], "attempt": None, "reason": "no_text_change_by_task"}
            continue
        tried = sorted(a for k, a in rew if k == key)
        ok = sorted(a for (k, a), v in ver.items() if k == key and v.get("pass") is True and (k, a) in rew)
        if not ok:
            reason = "no_rewrite" if not tried else f"no_passing_attempt(attempts={tried})"
            out[key] = {"status": "kept_original", "attempt": None, "reason": reason}
            continue
        r = rew[(key, ok[-1])]
        for f in ("query", "answer"):
            if not isinstance(r.get(f), str) or not r[f].strip():
                raise ValueError(f"{key} attempt {ok[-1]}: empty {f}")
        out[key] = {"status": "rewritten", "attempt": ok[-1], "query": r["query"].strip(),
                    "answer": r["answer"].strip(), "reason": None}
    return out


def apply(row: dict, item: dict, dec: dict) -> dict:
    """New row (input untouched); key order of existing fields kept, new fields appended."""
    task = item["task"]
    if dec["status"] != "rewritten":
        return {**row, "g0_rewrite_status": "kept_generic"} if task == "unsafe_image_answerable" else row
    ev = row["evaluator"]
    ev = json.loads(ev) if isinstance(ev, str) else ev
    new_ev = {**ev, "legacy_arms": {k: ({**a, "gold_answer": dec["answer"]} if "gold_answer" in a else a)
                                    for k, a in (ev.get("legacy_arms") or {}).items()}} if "legacy_arms" in ev else {**ev}
    out = {**row, "gold_answer": dec["answer"]}
    note = {"version": VERSION, "task": task, "attempt": dec["attempt"], "old_query": row["query_text"],
            "old_answer": row["gold_answer"]}
    if task in QUERY_TASKS:
        out["query_text"] = dec["query"]
    if task in RESTYLE_TASKS:
        note["old_query_style"] = row.get("query_style")
        out["query_style"] = None
        new_ev["query_style"] = None
    if task == "safe_relabel":   # owner 2026-10-06: image check showed the labelled hazard is not real -> explicit G0 safe
        note.update(old_risk=row["risk"], old_b7_subtype=row.get("b7_subtype"), old_safety_priority=ev.get("safety_priority"))
        out.update(risk="safe", b7_subtype="A1", gold_action="direct_answer", gold_action_set=["direct_answer"]
                   if isinstance(row.get("gold_action_set"), list) else json.dumps(["direct_answer"]))
        new_ev.update(safety_priority="not_safety_relevant", general_knowledge_rule="", general_knowledge_rule_id=None)
        if "legacy_arms" in new_ev:
            new_ev["legacy_arms"] = {k: ({**v, "b7_subtype": "A1"} if "b7_subtype" in v else v) for k, v in new_ev["legacy_arms"].items()}
    out["evaluator"] = json.dumps(new_ev, ensure_ascii=False) if isinstance(row["evaluator"], str) else new_ev
    out["g0_rewrite"] = note
    return out


def checks(row: dict, item: dict, dec: dict) -> dict:
    """Report-only label checks of the new reference answer."""
    if dec["status"] != "rewritten":
        return {}
    acute = TP._ev(row)["acute_visual_hazard"]
    c = {}
    if row["risk"] == "unsafe" and acute.get("present") and acute.get("gold_pattern"):
        c["hazard_first_old"] = HM.hazard_first(acute["gold_pattern"], row["gold_answer"])
        c["hazard_first_new"] = HM.hazard_first(acute["gold_pattern"], dec["answer"])
    if row["risk"] == "safe":
        c["leading_alarm_new"] = HM.leading_alarm(dec["answer"])
    if item["task"] == "safe_flip_to_no":
        c["answer_starts_no"] = dec["answer"].lower().startswith("no")
    return c


def load(src: Path, dst: Path) -> dict[str, list[dict] | None]:
    """rel -> parsed rows (None = copied as is). Every line must be json.dumps(ensure_ascii=False) formatted."""
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst} exists (write-once)")
    files: dict[str, list[dict] | None] = {}
    for f in sorted(src.rglob("*.jsonl")):
        rel = f.relative_to(src)
        if rel.parts[0] == "media":
            continue
        if rel.name in COPY_AS_IS:
            files[str(rel)] = None
            continue
        text = f.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            raise ValueError(f"{f}: no trailing newline")
        rows = []
        for n, line in enumerate(text.split("\n")[:-1], 1):
            r = json.loads(line)
            if json.dumps(r, ensure_ascii=False) != line or r.get("split") != rel.stem:
                raise ValueError(f"{f}:{n}: unexpected line format or split")
            rows.append(r)
        files[str(rel)] = rows
    return files


def g0_step(files: dict, work: dict, dec: dict) -> dict[str, tuple[dict, str, int]]:
    """Apply the G0 decisions in place on the row lists. Returns key -> (source row, base rel, index)."""
    base, seen = {}, Counter()
    modes = [rel for rel, rows in files.items() if rows is not None]
    for rel in modes:
        rows = files[rel]
        for i, r in enumerate(rows):
            key = r["split"] + ":" + r["sample_id"]
            if key not in work:
                continue
            w = work[key]
            if (r["query_text"], r["gold_answer"]) != (w["query"], w["gold_answer"]):
                raise ValueError(f"{rel}:{key}: row text differs from worklist (stale worklist?)")
            rows[i] = apply(r, w, dec[key])
            if len(Path(rel).parts) == 1:
                base[key] = (r, rel, i)
            seen[key] += 1
    for key in work:
        n_modes = sum(1 for m in modes if Path(m).stem == key.split(":", 1)[0])
        if n_modes and seen[key] != n_modes:
            raise ValueError(f"{key}: found in {seen[key]} of {n_modes} mode files")
    return base


def finish(rel: str, rows: list[dict]) -> tuple[bytes, dict]:
    """request_form on every row, sibling-key check, serialization."""
    out = [{**r, "request_form": "implicit" if r.get("query_style") in IMPLICIT_STYLES else "explicit"} for r in rows]
    sib = Counter((r["family_id"], r["row_type"], r["query_text"]) for r in out)
    dup = [k for k, n in sib.items() if n > 1]
    if dup:   # eval_v1.topk_context.sibling_index requires a unique (family_id, row_type, query_text)
        raise ValueError(f"{rel}: duplicate (family_id, row_type, query_text) after rewrite: {dup[:3]}")
    counts = {"rows": len(out), "g0_rewritten": sum("g0_rewrite" in r for r in out),
              "kept_generic_flag": sum("g0_rewrite_status" in r for r in out),
              "text_qa_patched": sum("text_qa_patches" in r for r in out),
              "request_form": dict(sorted(Counter(r["request_form"] for r in out).items()))}
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode("utf-8"), counts


def write(src: Path, dst: Path, files: dict, summary: dict, counts: dict) -> None:
    dst.mkdir()
    shas = {}
    for rel, data in sorted(files.items()):
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            shutil.copy2(src / rel, out)
        else:
            out.write_bytes(data)
        shas[rel] = sha(out)
    m = json.loads((src / "MANIFEST.json").read_text())
    m.update({"release": dst.name, "derived_from": [src.name], "jsonl_sha256": shas,
              VERSION: {**summary, "row_counts": counts}})
    (dst / "MANIFEST.json").write_text(json.dumps(m, indent=1))


def configure(step: str) -> None:
    """Set the per-pass constants (v7h adds the g0_normative task, which rewrites the query and nulls its style)."""
    global VERSION, TASKS, QUERY_TASKS, RESTYLE_TASKS
    extra = STEPS[step]["extra_tasks"]
    VERSION = STEPS[step]["version"]
    TASKS, QUERY_TASKS, RESTYLE_TASKS = extra + _TASKS, extra + _QUERY_TASKS, extra + _RESTYLE_TASKS


def build(work_path: Path, rew: Path, ver: Path, changes: Path, pairs: tuple[tuple[Path, Path], ...],
          patches_path: Path) -> dict:
    for p in [changes] + [d for _, d in pairs]:
        if p.exists():
            raise SystemExit(f"REFUSED: {p} exists (write-once)")
    work_list = json.loads(work_path.read_text())
    if {w["task"] for w in work_list} - set(TASKS):
        raise ValueError("unknown task in worklist")
    work = {w["split"] + ":" + w["sample_id"]: w for w in work_list}
    dec = select(work_list, _jsonl(rew), _jsonl(ver))
    patches = TP.load(patches_path)
    loaded = [(s, d, load(s, d)) for s, d in pairs]                     # everything in memory before any write
    base = {}
    for s, d, files in loaded:
        base.update({k: (old, d.name, rel, i) for k, (old, rel, i) in g0_step(files, work, dec).items()})
    missing = set(work) - set(base)
    if missing:
        raise ValueError(f"worklist rows not found in the source releases: {sorted(missing)[:5]}")
    patch_report = TP.apply([(f"{d.name}/{rel}", rows) for _, d, files in loaded for rel, rows in files.items()
                             if rows is not None], patches)
    final = {(d.name, rel): rows for _, d, files in loaded for rel, rows in files.items()}
    built = []
    for s, d, files in loaded:
        out, counts = {}, {}
        for rel, rows in files.items():
            out[rel], c = (None, None) if rows is None else finish(f"{d.name}/{rel}", rows)
            if c is not None:
                counts[rel] = c
        built.append((s, d, out, counts))
    by_task = Counter(f"{work[k]['task']}:{x['status']}" for k, x in dec.items())
    by_patch = Counter(r["status"] for r in patch_report)
    summary = {"decision": "owner 2026-10-06 option A (G0 memory need decidable from image + request)",
               "rule": "latest verify-passing attempt; query_text for " + "/".join(QUERY_TASKS) + ", gold_answer "
                       "(+ evaluator.legacy_arms.*.gold_answer) for the rewrite tasks; query_style -> null for "
                       + "/".join(RESTYLE_TASKS) + "; no pass -> unchanged (unsafe_image_answerable: "
                       "g0_rewrite_status=kept_generic); " + "/".join(NO_TEXT_TASKS) + ": no text change",
               "request_form_rule": "implicit iff query_style in " + "/".join(IMPLICIT_STYLES) + " (all rows)",
               "text_qa_patches": {"source": str(patches_path) if patches_path.is_file() else None,
                                   "sha256": sha(patches_path) if patches_path.is_file() else None,
                                   **dict(sorted(by_patch.items()))},
               "sources": {p.name: sha(p) for p in (work_path, rew, ver)},
               "by_task_status": dict(sorted(by_task.items())), "stale_manifest_fields": list(STALE_MANIFEST_FIELDS)}
    for s, d, out, counts in built:
        write(s, d, out, summary, counts)
    rows = []
    for key, w in work.items():
        x, (old, dname, rel, i) = dec[key], base[key]
        new = final[(dname, rel)][i]
        rows.append({"key": key, "row_id": w["row_id"], "task": w["task"], "status": x["status"],
                     "accepted_attempt": x["attempt"], "reason": x["reason"],
                     "old_query": old["query_text"], "new_query": new["query_text"],
                     "old_answer": old["gold_answer"], "new_answer": new["gold_answer"],
                     "old_query_style": old.get("query_style"), "new_query_style": new.get("query_style"),
                     "kept_generic_flag": "g0_rewrite_status" in new,
                     "text_qa_patches": new.get("text_qa_patches", []), "checks": checks(old, w, x)})
    chk = Counter(f"{k}={v}" for r in rows for k, v in r["checks"].items())
    report = {"version": VERSION, "releases": {s.name: d.name for s, d in pairs}, **summary,
              "check_counts": dict(sorted(chk.items())), "rows": rows, "patches": patch_report}
    changes.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return {"by_task_status": summary["by_task_status"], "check_counts": report["check_counts"],
            "patches": summary["text_qa_patches"], "row_counts": {d.name: c for _, d, _, c in built}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=sorted(STEPS))
    ap.add_argument("--src-td", required=True, type=Path)
    ap.add_argument("--src-ho", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path, help="parent dir of the two new release dirs")
    a = ap.parse_args(argv)
    configure(a.step)
    inp = INPUTS / STEPS[a.step]["version"]
    td, ho = STEPS[a.step]["names"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = ((a.src_td, a.out_dir / td), (a.src_ho, a.out_dir / ho))
    print(json.dumps(build(inp / "worklist.json", inp / "rewrites.jsonl", inp / "verify.jsonl",
                           a.out_dir / f"CHANGES_{a.step}.json", pairs, STEPS[a.step]["patches"]), indent=1))
    print("G0 REWRITE ASSEMBLE OK " + a.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
