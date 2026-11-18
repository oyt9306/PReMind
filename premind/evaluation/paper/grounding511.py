"""Table 2 grounding columns on the paper population: 511 target-present on-demand episodes (349 unique requests).

The paper's Table 2 reports Cite F1, Fact use and Misattr. over 511 episodes: the base-4B episodes of the final
evaluation assembly with retrieved memory, memory-call arm, a needed call and a target-present row type. Some requests
occur twice in that list, so these values differ from the 343-request summaries in summarize.py (both are correct;
only the population differs). The id list ships as eval_inputs/paper511_ids.json.

  python -m premind.evaluation.paper.grounding511 <run_dir>     # run_dir holds records.jsonl + texts.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fact_overlay as F  # noqa: E402

IDS = HERE.parents[1] / 'bv3' / 'eval_inputs' / 'paper511_ids.json'


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def grounding511(run: Path) -> dict:
    """Cite F1 (end-to-end: unshown or degenerate answers score 0), Fact use (repaired_normalized) and Misattr. in %."""
    ids = json.loads(IDS.read_text())
    assert len(ids) == 511, len(ids)
    rec = {r['row_id']: r for r in jl(run / 'records.jsonl')
           if (r['mode'], r['arm'], r['kind']) == ('retrieved', 'memory_call', 'greedy')}
    ov = F.score(run, 'main', set(ids))
    fact = {r['row_id']: r['variants']['repaired_normalized']['Fact'] for r in ov['rows']}
    mean = lambda v: 100 * sum(v) / len(v)
    return {'n': len(ids), 'n_unique': len(set(ids)),
            'Cite_F1': mean([(rec[k]['cite_f1'] or 0) if rec[k]['shown'] and rec[k]['n_gold'] else 0 for k in ids]),
            'Fact_use': mean([fact[k] for k in ids]),
            'Misattr': mean([float(bool(rec[k]['shown'] and (rec[k].get('distractor') is True or rec[k].get('wrong_concept') is True)))
                             for k in ids])}


if __name__ == '__main__':
    print(json.dumps(grounding511(Path(sys.argv[1])), indent=1))
