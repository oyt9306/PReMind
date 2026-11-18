"""Frozen Luna operators under ONE cross-process $2 cap; no GPU. Only new local test outputs."""
import argparse,json,os,sys
from pathlib import Path
from ft_common import *
def bridge(cohort):
 if cohort=='main':
  sys.path.insert(0,str(FIX));import judge_fix2b as J
  BR,V=J.bridge(MAIN);inv=V4/'release_test_v4q_fix2b/INVENTORY_TEST_v4q_APPROVED_fix2b.jsonl'
 else:
  sys.path.insert(0,str(C10));import judge_conflict as J
  BR,V=J.bridge();inv=V4/'conflict_fix2_final_v3/INVENTORY_conflict_v10.jsonl'
 return BR,V,inv

def main():
 ap=argparse.ArgumentParser();ap.add_argument('key');ap.add_argument('cohort',choices=['main','conflict']);ap.add_argument('lane',choices=['ondemand','forced']);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert_owner();verify_sources();path=run(a.key,a.cohort,a.lane);root=H/'judges'/a.cohort/f'{a.key}__{a.lane}'
 BR,V,inventory=bridge(a.cohort)
 if not a.execute:
  print(json.dumps({'run':str(path),'n_acute':len(V.run_items(path,('memory_call','retrieved','greedy'))),'cap_total':2,'inventory':str(inventory)}));return
 if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('API key missing')
 from eval_v1 import api_providers as AP
 import importlib.util
 spec=importlib.util.spec_from_file_location('forced_test_budget',CODE/'budget.py');budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
 budget.install(AP,H/'TEST_LUNA_BUDGET.jsonl',f'{a.key}:{a.cohort}:{a.lane}')
 root.mkdir(parents=True,exist_ok=True);acute=root/f'{a.key}_acute.jsonl'
 if not acute.exists():V.main(['--run',str(path),'--setting','memory_call,retrieved,greedy','--backend','luna','--rubric','v4','--usd-cap','.65','--ledger',str(root/'acute_ledger.jsonl'),'--out',str(acute)])
 verdicts=jl(acute);by={v['row_id']:v for v in verdicts}
 expected={x['row_id'] for x in V.run_items(path,('memory_call','retrieved','greedy'))}
 if len(verdicts)!=len(expected) or set(by)!=expected or any(v.get('state') not in V.STATES for v in verdicts):raise RuntimeError('Missing/invalid/duplicate acute judgments; no imputation')
 release=MAIN if a.cohort=='main' else CR;rows={r['sample_id']:r for r in jl(release/'holdout.jsonl')}
 packets=root/'packets'/f'{a.key}_episodes.jsonl'
 if not packets.exists():
  pp=[]
  from eval_v1 import hazard_match as HM
  for t in jl(path/'texts.jsonl'):
   rid=t['row_id']
   if rid not in by:continue
   row=rows[rid];s2=t.get('stage2') or ''
   pp.append(dict(model=a.key,canonical_id=row['evaluator']['row_id'],mode='retrieved',arm='memory_call',kind='greedy',sample_idx=t.get('sample_idx',0),local_id=rid,response_sha256=__import__('hashlib').sha256(s2.encode()).hexdigest(),image=row['evaluator']['source_image_sha256'],run=str(path),acute=True,acute_verdict=by[rid],acute_state=by[rid]['state'],answer=HM.answer_text(s2),item=BR.build_item(row)))
  if len(pp)!=len(expected):raise RuntimeError('Packet population mismatch')
  packets.parent.mkdir(parents=True,exist_ok=True)
  with packets.open('x') as f:f.writelines(json.dumps(x)+'\n' for x in pp)
 sys.path[:0]=[str(V4/'final_eval_assembly'),str(FC)]
 import secondary
 secondary.HERE=root.resolve()
 for task in ('severity','mie'):
  out=root/f'{a.key}_{task}.jsonl'
  if not out.exists():
   args=['--task',task,'--episodes',str(packets),'--backend','luna','--out',str(out),'--execute-paid','--usd-cap','.25','--ledger',str(root/(task+'_ledger.jsonl'))]
   if task=='mie':args+=['--inventory',str(inventory)]
   secondary.main(args)
  want=secondary.packets(jl(packets),task,inventory if task=='mie' else None)
  got=jl(out)
  if len(got)!=len(want) or {x['item_key'] for x in got}!={x['item_key'] for x in want}:raise RuntimeError('Incomplete '+task+' judgments; preserved for inspection')
 (root/'COMPLETE').write_text('All frozen acute/severity/MIE judgments complete\n')
 print('JUDGING_COMPLETE',a.key,a.cohort,a.lane,flush=True)
if __name__=='__main__':main()
