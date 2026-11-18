"""Active-only call diagnostics for new AD runs, with legacy secondary rates."""
def rates(records):
    call=[r for r in records if r['arm']=='memory_call']
    need=[r for r in call if r['memory_required']]
    active=[r for r in call if not r['memory_required'] and r.get('g0_call_policy')=='active']
    neutral=[r for r in call if not r['memory_required'] and r.get('g0_call_policy')=='neutral']
    legacy=[r for r in call if not r['memory_required']]
    def mean(rs,predicate):return sum(predicate(r) for r in rs)/len(rs) if rs else None
    recall=mean(need,lambda r:r.get('decision')=='call_pdb')
    specificity=mean(active,lambda r:r.get('decision')=='direct_answer')
    oldspec=mean(legacy,lambda r:r.get('decision')=='direct_answer')
    return {'false_call_rate':mean(active,lambda r:r.get('decision')=='call_pdb'),
       'n_false_call_rate':len(active),'g0_call_rate':mean(active,lambda r:r.get('decision')=='call_pdb'),
       'n_g0_call_rate':len(active),'neutral_call_rate':mean(neutral,lambda r:r.get('decision')=='call_pdb'),
       'n_neutral_call_rate':len(neutral),'legacy_all_g0_call_rate':mean(legacy,lambda r:r.get('decision')=='call_pdb'),
       'n_legacy_all_g0_call_rate':len(legacy),'call_ba':(recall+specificity)/2 if recall is not None and specificity is not None else None,
       'n_call_ba':len(need)+len(active),'legacy_all_g0_call_ba':(recall+oldspec)/2 if recall is not None and oldspec is not None else None}

def install():
    from trainer_v2 import metrics_v9 as M
    if getattr(M.episode_record,'_ad_policy',False):return
    original_record=M.episode_record;original_summary=M.summarize
    def record(item,*args,**kwargs):
        r=original_record(item,*args,**kwargs)
        return {**r,'g0_call_policy':item.row.get('g0_call_policy'),'canonical_id':item.row.get('canonical_id')}
    record._ad_policy=True
    def summarize(rows):return {**original_summary(rows),**rates(rows)}
    M.episode_record=record;M.summarize=summarize
