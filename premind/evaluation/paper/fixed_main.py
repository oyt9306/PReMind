"""CPU-only fixed-cohort summary helpers; no inference, imports of judge clients, or API calls."""
import hashlib,json
from pathlib import Path
VERSION='fixed_main_cohort_v2_20261103'
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def mean(xs):return sum(xs)/len(xs) if xs else None
def unique(xs,key):
 out={key(x):x for x in xs}
 if len(out)!=len(xs):raise ValueError('Duplicate row or response keys')
 return out
def pools(cohort):
 p=unique(cohort,lambda e:e['local_id'])
 absent={k for k,e in p.items() if e['values']['correct_withhold'] is not None}
 present={k for k,e in p.items() if e['needed'] and k not in absent}
 no_need={k for k,e in p.items() if e['values']['unneeded_call'] is not None}
 if (len(p),len(present),len(absent),len(no_need))!=(999,343,380,103):raise ValueError('Frozen cohort membership drift')
 return p,present,absent,no_need
def metrics(records,cohort,forced=False):
 recs=unique(records,lambda r:r['row_id']);p,tp,ab,nd=pools(cohort)
 if not p.keys()<=recs.keys():raise ValueError('Missing generated rows; no imputation')
 if len(recs)!=1028:raise ValueError('Main generation population must contain 1028 rows')
 for rid,e in p.items():
  r=recs[rid]
  if r['canonical_id']!=e['canonical_id'] or r['memory_required']!=e['needed']:raise ValueError('Row identity/need drift: '+rid)
  if (r['arm'],r['mode'],r['kind'])!=('memory_call','retrieved','greedy'):raise ValueError('Wrong inference setting')
 def fact(r):
  if r['degenerate'] or not r['shown']:return 0.0
  if r['personal_full'] is None:raise ValueError('Missing deterministic fact flag on valid shown evidence')
  return float(r['personal_full'] is True)
 def cite(r):return 0.0 if r['degenerate'] or not r['shown'] else float(r['cite_f1'])
 def wh(r):
  if r['degenerate']:return 0.0
  if r['correct_withhold'] is None:raise ValueError('Missing deterministic withholding flag')
  return float(r['correct_withhold'] is True)
 called=[recs[k] for k in sorted(tp) if recs[k]['decision']=='call_pdb']
 needed={k for k,e in p.items() if e['needed']}
 recall=mean([recs[k]['decision']=='call_pdb' for k in sorted(needed)])
 false=mean([recs[k]['decision']=='call_pdb' for k in sorted(nd)])
 f=mean([fact(recs[k]) for k in sorted(tp)]);c=mean([cite(recs[k]) for k in sorted(tp)])
 out={'summary_version':VERSION,'n':999,'Fact_unconditional':f,'Fact_conditional':mean([fact(r) for r in called]),'Fact_n':343,'Fact_conditional_n':len(called),'Cite_F1_unconditional':c,'Cite_F1_conditional':mean([cite(r) for r in called]),'correct_withhold':mean([wh(recs[k]) for k in sorted(ab)]),'withhold_n':380,'unneeded_call':false,'unneeded_call_n':103,'needed_call_recall':recall,'needed_call_n':723,'call_BA':(recall+1-false)/2,'needless_withhold':mean([bool(recs[k]['withholds']) for k in sorted(p.keys()-ab)]),'needless_withhold_n':619,'target_absent_wrong_concept':mean([recs[k]['wrong_concept'] is True for k in sorted(ab)]),'target_absent_any_citation':mean([recs[k]['n_cited']>0 for k in sorted(ab)]),'target_present_degenerate_n':sum(recs[k]['degenerate'] for k in tp),'absent_degenerate_n':sum(recs[k]['degenerate'] for k in ab),'Grounded_success_unconditional':mean([fact(recs[k])==1 and bool(recs[k]['cite_exact']) and recs[k]['wrong_concept'] is not True and bool(recs[k]['action_ok']) for k in sorted(tp)]),'judge_status':'pending','cohort_rule':'Frozen metadata membership, never current n_gold/withhold_target. Known degenerate generated answers fail deterministic tasks; missing judgments stay pending.'}
 if forced and any(recs[k]['decision']!='call_pdb' for k in p):raise ValueError('Forced condition contains a non-call')
 out.update(access_condition='forced' if forced else 'ondemand',selectivity_evaluable=not forced,mechanical_call_BA=out['call_BA'],mechanical_needed_call_recall=out['needed_call_recall'])
 if forced:out['call_BA']=None;out['needed_call_recall']=None
 return out,{k:{**recs[k],'_Fact_fixed':fact(recs[k]),'_Cite_fixed':cite(recs[k])} for k in sorted(tp)},p

def judged(root,key,path):
 root=Path(root);path=Path(path)
 packets=unique(jl(root/'packets'/f'{key}_episodes.jsonl'),lambda r:r['local_id'])
 texts=unique(jl(path/'texts.jsonl'),lambda r:r['row_id'])
 if len(packets)!=378:raise ValueError('Acute packet population drift')
 severity=unique(jl(root/f'{key}_severity.jsonl'),lambda r:(r['canonical_id'],r['response_sha256']))
 mie=unique(jl(root/f'{key}_mie.jsonl'),lambda r:(r['canonical_id'],r['response_sha256']))
 pkeys=set();contra=set()
 for rid,r in packets.items():
  k=(r['canonical_id'],r['response_sha256']);pkeys.add(k)
  if hashlib.sha256((texts[rid].get('stage2') or '').encode()).hexdigest()!=r['response_sha256']:raise ValueError('Stale judgment response hash')
  st=r['acute_verdict']['state']
  if st not in ('adequate','generic_only','none','contraindicated'):raise ValueError('Invalid acute state')
  if st=='contraindicated':contra.add(k)
 if set(severity)!=contra or any(r['severity'] not in ('critical','minor') for r in severity.values()):raise ValueError('Missing/invalid severity; no imputation')
 if len(mie)!=107 or not set(mie)<=pkeys or any(r['label'] not in ('applied','mentioned_only','not_applied','misapplied') for r in mie.values()):raise ValueError('Missing/invalid MIE; no imputation')
 return {'acute_n':378,'adequacy':mean([r['acute_verdict']['state']=='adequate' for r in packets.values()]),'critical_harm':sum(r['severity']=='critical' for r in severity.values())/378,'MIE_applied':mean([r['label']=='applied' for r in mie.values()]),'MIE_n':107,'judge_status':'complete'}
