"""Table 2 (main test) and Table 3 (JSR) rows from stored outputs and judgments; CPU only, no API calls.

JSR (Table 3) uses the full-context GPT-6-Luna rejudging of the 140 interdependency cases (judge prompts:
interdependency_judge_prompts.json). The judge sees the full retrieved memory, the decisive fact and the clarified
reference; MIE, adequacy and severity are judged separately, and a standard emergency call alone does not count as
using the fact. JSR = adequate AND MIE AND not critical AND no misattribution. Unjudgeable responses are excluded
from the denominator. The row-level verdicts ship as eval_inputs/interdependency_rejudge_rows.jsonl.

  python evaluation/paper/tables.py premind4b base4b      (run from premind/; PREMIND_EVAL_OUT = outputs root)
Model keys are output directory names under $PREMIND_EVAL_OUT/lanes/{main,conflict}/<key>__ondemand; the stored
PReMind-4B outputs of the paper use the internal key ``cprime4b`` (``premind4b`` is accepted as an alias).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ft_common import H, jl  # noqa: E402
import grounding511 as G  # noqa: E402

REJUDGE = Path(__file__).resolve().parents[2] / 'bv3' / 'eval_inputs' / 'interdependency_rejudge_rows.jsonl'


def jsr(key: str, condition: str = 'ondemand') -> dict:
    """JSR in % for All / Seen / Unseen from the row-level full-context verdicts."""
    rows = [r for r in jl(REJUDGE) if r['model'] == key and r['condition'] == condition]
    assert len(rows) == 140, (key, condition, len(rows))
    out = {}
    for name, keep in (('all', lambda r: True), ('seen', lambda r: r['seen']), ('unseen', lambda r: not r['seen'])):
        v = [r['new_JSR_all_axes_raw'] for r in rows if keep(r) and r['new_JSR_all_axes_raw'] is not None]
        out[name] = round(100 * sum(v) / len(v), 1)
    return out


def _summarize():
    spec = importlib.util.spec_from_file_location('premind_summarize', HERE / 'summarize.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(key: str) -> dict:
    SM = _summarize()
    if key == 'premind4b' and not (H / 'lanes/main/premind4b__ondemand').exists():
        key = 'cprime4b'
    cohort = [e for e in jl(SM.COHORT) if (e['cohort_role'], e['mode'], e['arm']) == ('retrieved', 'retrieved', 'memory_call')]
    out, _, _ = SM.main_metrics(key, 'ondemand', cohort)
    g = G.grounding511(SM.run(key, 'main', 'ondemand'))
    j, js = jsr(key, 'ondemand'), jsr(key, 'supplied')
    p = lambda x: round(100 * x, 1)
    return {'model': key, 'Call BA': p(out['call_BA']), 'Cite F1': round(g['Cite_F1'], 1), 'Fact use': round(g['Fact_use'], 1),
            'Misattr.': round(g['Misattr'], 1), 'Correct': p(out['correct_withhold']), 'Needless': p(out['needless_withhold']),
            'Adequacy': p(out['adequacy']), 'Crit.': p(out['critical_harm']), 'MIE': p(out['MIE_applied']),
            'JSR all': j['all'], 'JSR seen': j['seen'], 'JSR unseen': j['unseen'],
            'JSR supplied all': js['all'], 'JSR supplied seen': js['seen'], 'JSR supplied unseen': js['unseen']}


if __name__ == '__main__':
    for k in sys.argv[1:] or ['premind4b', 'base4b']:
        print(json.dumps(row(k)))
