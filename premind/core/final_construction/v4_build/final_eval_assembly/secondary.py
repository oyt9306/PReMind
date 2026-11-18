"""Benchmark adapters for frozen severity v2 / MIE v1; export is the default."""
from assembly_common import *
import argparse,importlib.util
sys.path.insert(0,str(FC/'v4_build/text_knowledge_probe/severity'))
import severity_v2 as SV
sys.path.insert(0,str(FC/'v4_build/mie_metric'))
import mie_v1 as MIE

def packets(rows,task,inventory=None):
 inventory={x['evaluator_row_id']:x for x in jl(inventory)} if inventory else {}
 seen=set();out=[]
 for r in rows:
  k=(r['model'],r['canonical_id'],r['mode'],r['arm'],r['kind'],r['sample_idx'],r['response_sha256'])
  if k in seen:continue
  seen.add(k)
  if task=='severity':
   v=r.get('acute_verdict')
   if not v or v['state']!='contraindicated':continue
   item={'hazard_types':r['item']['hazard_types'],'situation':r['item']['visible_hazard'],
         'key_actions':[a['desc'] for a in r['item']['key_actions']],
         'earlier_reason':v.get('reason',''),'earlier_quote':v.get('harmful_quote','')}
   key=SV.item_key(item,r['answer']);rubric=SV.RUBRIC_SHA256
  else:
   inv=inventory.get(r['canonical_id'])
   if not inv or r['mode'] not in inv['modes']:continue
   item=MIE.build_item(inv);key=MIE.item_key(item,r['answer']);rubric=MIE.RUBRIC_SHA256
  out.append({k:r[k] for k in ('model','canonical_id','mode','arm','kind','sample_idx','response_sha256','image','run')}|{'item':item,'answer':r['answer'],'item_key':key,'rubric_sha256':rubric})
 return out

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--task',choices=['severity','mie'],required=True);ap.add_argument('--episodes',type=Path,required=True);ap.add_argument('--inventory',type=Path);ap.add_argument('--backend',choices=['export','table','luna'],default='export');ap.add_argument('--table',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ledger',type=Path);ap.add_argument('--cache',type=Path);ap.add_argument('--usd-cap',type=float,default=0);ap.add_argument('--execute-paid',action='store_true');a=ap.parse_args(argv)
 if not a.out.resolve().is_relative_to(HERE) or a.out.exists():raise ValueError('new V output required')
 if a.task=='mie' and not a.inventory:raise ValueError('owner-approved MIE inventory required')
 if a.task=='severity':SV.rubric_text()
 ps=packets(jl(a.episodes),a.task,a.inventory)
 if a.backend=='export':lines(a.out,ps);print(json.dumps({'n':len(ps),'out':str(a.out)}));return
 if a.backend=='luna':
  if not a.execute_paid or a.usd_cap<=0 or not a.ledger:raise ValueError('explicit paid authorization, ledger and positive cap required')
  cls=SV.SeverityLunaBackend if a.task=='severity' else MIE.MieLunaBackend
  backend=cls(a.ledger,usd_cap=a.usd_cap,cache_path=a.cache)
  verdicts=(SV.judge if a.task=='severity' else MIE.judge)([(p['item'],p['answer']) for p in ps],backend)
 else:
  table={v['item_key']:v for v in jl(a.table)};verdicts=[table.get(p['item_key'],{}) for p in ps]
 out=[]
 for p,v in zip(ps,verdicts):
  field='severity' if a.task=='severity' else 'label'
  if v.get(field) not in (('critical','minor') if a.task=='severity' else ('applied','not_applied','mentioned_only','misapplied')):continue
  out.append({k:v for k,v in p.items() if k not in ('item','answer')}|{field:v[field],'reason':v.get('reason','')})
 lines(a.out,out)
if __name__=='__main__':main()
