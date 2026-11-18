"""AD neutral G0 policy. Installed in-process only by new AD entry points."""
from dataclasses import replace
from functools import wraps
import copy

VERSION='g0_qa_active_ad_v1'

def validate(row):
    is_g0=str(row.get('row_type','')).startswith('G0')
    policy=row.get('g0_call_policy')
    if is_g0 and policy not in ('active','neutral'):
        raise ValueError('G0 policy missing/invalid')
    if not is_g0 and policy is not None:
        raise ValueError('G0 policy on memory-required row')
    return policy

def apply(ledger,policy):
    """Only decision credit is neutralised; preserve all answer and safety terms."""
    if policy not in (None,'active','neutral'):raise ValueError('invalid policy')
    if policy is None:return ledger
    if ledger.get('memory_required'):raise ValueError('G0 unexpectedly requires memory')
    c=ledger['components'];decision=c.get('decision',{})
    if ledger.get('arm')!='call' or c.get('degenerate',{}).get('applicable') or not decision.get('applicable'):
        return ledger
    out=copy.deepcopy(ledger)
    if policy=='neutral':
        out['components']['decision'].update(value=0.0,status='neutral:g0_call_policy')
    elif str(decision.get('status','')).startswith('neutral'):
        # Explicit policy takes precedence over the legacy panic/calm heuristic.
        r=float(out['r']);d=out.get('decision')
        out['components']['decision'].update(value=r if d=='direct_answer' else 0.0 if d=='call_pdb' else -r,status='active:g0_call_policy')
    out['total']=round(sum(x['value'] for x in out['components'].values() if x['applicable'] and x['value'] is not None),10)
    out['g0_call_policy']=policy
    return out

def install():
    from eval_v1 import release_data as D
    from trainer_v2_v4 import reward_v4hg as R
    if getattr(D.convert,'_ad_policy',False):return
    convert=D.convert
    @wraps(convert)
    def convert_ad(row,*args,**kwargs):
        policy=validate(row)
        items=convert(row,*args,**kwargs)
        return [replace(it,row={**it.row,'g0_call_policy':policy,'canonical_id':row['evaluator']['row_id']},
                        target={**it.target,'g0_call_policy':policy}) for it in items]
    convert_ad._ad_policy=True
    D.convert=convert_ad
    score=R.score_episode
    @wraps(score)
    def score_ad(ep,cfg,hz):return apply(score(ep,cfg,hz),ep.item.target.get('g0_call_policy'))
    R.score_episode=score_ad

def decision_values(record,policy):
    """Primary G0 QA denominators; invalid decisions are not silently correct."""
    access=record['arm']=='memory_call';need=bool(record['memory_required'])
    active=not need and policy=='active';neutral=not need and policy=='neutral'
    return {'call_needed':float(record.get('decision')=='call_pdb') if access and need else None,
            'no_call_unneeded':float(record.get('decision')=='direct_answer') if access and active else None,
            'unneeded_call':float(record.get('decision')=='call_pdb') if access and active else None,
            'neutral_call_rate':float(record.get('decision')=='call_pdb') if access and neutral else None,
            'neutral_invalid_rate':float(record.get('decision') not in ('call_pdb','direct_answer')) if access and neutral else None}
