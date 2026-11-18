"""AE mode/view exposure recomputation, QA keys and CPU diagnostics."""
from pathlib import Path
from dataclasses import replace
import json,sys
HERE=Path(__file__).resolve().parent;FC=HERE.parents[2]
sys.path.insert(0,str(FC))
from v4_build.g0_redesign_v1.rebuild_codex import g0_policy as AD,metrics_ad as OLD_METRICS
from . import qa_rules_v2 as QR

def group(row):
    return 'trap_exposed' if row.get('g0_trap')=='trap' and row.get('g0_trap_exposed') else 'trap_unexposed' if row.get('g0_trap')=='trap' else 'plain'

def install():
    from eval_v1 import release_data as D
    from trainer_v2 import metrics_v9 as M
    from trainer_v2_v4 import reward_v4hg as R
    if getattr(D.convert,'_ae',False):return
    AD.install();OLD_METRICS.install()
    keys={r['canonical_id']:r for r in map(json.loads,(HERE/'G0_QA_KEYS_v20_FINAL.jsonl').read_text().splitlines())}
    original=D.convert
    def convert(row,*args,**kwargs):
        its=original(row,*args,**kwargs);ev=row['evaluator']
        exposed=row.get('g0_trap')=='trap' and any(c['key']==row.get('g0_lure_key') and row.get('g0_lure_sentence','\0') in text for c,text in zip(ev['context_concepts'],row['context_texts']))
        fields={'g0_trap':row.get('g0_trap'),'g0_trap_exposed':bool(exposed),'qa_keys':keys.get(ev['row_id'])}
        return [replace(it,row={**it.row,**fields},target={**it.target,**fields}) for it in its]
    convert._ae=True;D.convert=convert
    score=R.score_episode
    def score_ae(ep,cfg,hz):
        ledger=score(ep,cfg,hz)
        if ep.item.row.get('g0_call_policy')=='active':
            from eval_v1 import hazard_match as HM
            ledger={**ledger,'qa_ae':QR.score(HM.answer_text(ep.stage2_text),ep.item.row['qa_keys'])}
        return ledger
    R.score_episode=score_ae
    record=M.episode_record
    def record_ae(item,ledger,**kw):
        r=record(item,ledger,**kw)
        return {**r,'g0_trap':item.row.get('g0_trap'),'g0_trap_exposed':item.row.get('g0_trap_exposed'),**ledger.get('qa_ae',{})}
    M.episode_record=record_ae
    summary=M.summarize
    def summarize(rs):return {**summary(rs),'priming_ae':diagnostics(rs)}
    M.summarize=summarize

def diagnostics(records):
    out={}
    for mode in sorted({r.get('mode','unknown') for r in records}):
        for name in ['trap_exposed','trap_unexposed','plain']:
            rows=[r for r in records if r.get('mode','unknown')==mode and r.get('g0_call_policy')=='active' and group(r)==name]
            calls=[r for r in rows if r['arm']=='memory_call']
            cell={'n':len(rows),'n_memory_call':len(calls),'call_rate':sum(r.get('decision')=='call_pdb' for r in calls)/len(calls) if calls else None}
            for branch in ['called','not_called']:
                xs=[r for r in calls if (r.get('decision')=='call_pdb')==(branch=='called')]
                cell[branch]={'n':len(xs),**{k:sum(bool(r[k]) for r in xs)/len(xs) if xs and all(k in r for r in xs) else None for k in ['qa_correct','qa_primed']}}
            out[mode+'/'+name]=cell
    return out
