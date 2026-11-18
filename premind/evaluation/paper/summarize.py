"""CPU-only summaries of new test outputs. Missing judgments remain pending; no imputation."""
import importlib.util,json,random,time
from collections import defaultdict
from ft_common import *
COHORT=FC.parent/'eval_inputs'/'episodes_base4b.jsonl'   # frozen 999-request scoring cohort
def mean(v):return sum(v)/len(v) if v else None
def bootstrap(rows):
 g=defaultdict(list)
 for image,a,b in rows:g[image].append(b-a)
 keys=sorted(g);rng=random.Random(0);bs=[]
 for _ in range(2000):
  picked=rng.choices(keys,k=len(keys));v=[x for k in picked for x in g[k]];bs.append(mean(v))
 bs.sort();return {'delta':mean([b-a for _,a,b in rows]),'ci95':[bs[50],bs[1949]],'n':len(rows),'n_images':len(keys)}
def main_metrics(key,lane,cohort):
 import sys
 sys.path.insert(0,str(CODE))
 from fixed_main import metrics,judged
 path=run(key,'main',lane)
 out,tp,P=metrics(jl(path/'records.jsonl'),cohort,forced=(lane=='forced'))
 root=H/'judges/main'/f'{key}__{lane}'
 if (root/'COMPLETE').exists():out.update(judged(root,key,path))
 sys.path.insert(0,str(CODE))
 from fact_overlay import augment
 out,tp=augment(path,cohort,out,tp)
 return out,tp,P

def cr_metrics(key,lane):
 spec=importlib.util.spec_from_file_location('forced_test_cr',str(CODE/'cr_v10.py'));CRF=importlib.util.module_from_spec(spec);spec.loader.exec_module(CRF)
 root=H/'cr_views'/key/lane;src=run(key,'conflict',lane);jd=H/'judges/conflict'/f'{key}__{lane}'
 (root/'runs'/key).mkdir(parents=True,exist_ok=True)
 for link,target in [(root/'runs'/key/'retrieved',src),(root/'judges',jd)]:
  if not link.exists():link.symlink_to(target)
 size,lora=model(key);CRF.C10.LOCAL[key]=(size,lora)
 if key not in CRF.C10.SYSTEMS:CRF.C10.SYSTEMS.append(key)
 CRF.EVAL=root;rows=CRF.meta_rows();es,_=CRF.system_rows(key,rows);es=[e for e in es if e['kind'] in CRF.SCORED_KINDS]
 assert len(es)==140
 fns={'CR':CRF.CRF.cr_strict,'adequacy':CRF.CRM.adequate,'critical_harm':CRF.CRM.critical,'MIE_applied':lambda e:float(e['mie']=='applied')}
 result={'n':140,**{k:mean([f(e) for e in es]) for k,f in fns.items()}}
 for split,flag in [('seen',True),('unseen',False)]:
  sub=[e for e in es if e['seen']==flag];result[split]={'n':len(sub),'CR':mean([CRF.CRF.cr_strict(e) for e in sub])}
 import sys
 sys.path.insert(0,str(CODE))
 from fact_overlay import score
 overlay=score(src,'conflict',[e['sample_id'] for e in es])
 result['Fact_overlay']={k:v for k,v in overlay.items() if k!='rows'}
 overlay_path=src/'FACT_PATTERN_OVERLAY.json'
 overlay_path.write_text(json.dumps(overlay,ensure_ascii=False,indent=2)+'\n')
 return result,{e['sample_id']:(e['image'],float(CRF.CRF.cr_strict(e))) for e in es}
def update():
 out={'updated_epoch':time.time(),'main_scored_cohort_source_sha256':sha(COHORT),'models':{},'paired':{},'pending':[]};cohort=[e for e in jl(COHORT) if (e['cohort_role'],e['mode'],e['arm'])==('retrieved','retrieved','memory_call')]
 mm={};cc={}
 for key in ('base4b','mix4b','cprime4b','base9b','mix9b'):
  out['models'][key]={}
  for lane in ('ondemand','forced'):
   d={};out['models'][key][lane]=d
   if (run(key,'main',lane)/'eval_report.json').exists():
    d['main'],mm[(key,lane)],P=main_metrics(key,lane,cohort)
   else:out['pending'].append([key,'main',lane])
   if (H/'judges/conflict'/f'{key}__{lane}'/'COMPLETE').exists():d['conflict'],cc[(key,lane)]=cr_metrics(key,lane)
   else:out['pending'].append([key,'conflict',lane,'judges_or_generation_pending'])
 for size in ('4b','9b'):
  for lane in ('ondemand','forced'):
   b,o='base'+size,'mix'+size;k=size+'/'+lane
   if (b,lane) in mm and (o,lane) in mm:
    rb,ro=mm[(b,lane)],mm[(o,lane)];images={e['local_id']:e['image'] for e in cohort}
    out['paired'][k+'/main_Fact_unconditional']=bootstrap([(images[rid],rb[rid]['_Fact_fixed'],ro[rid]['_Fact_fixed']) for rid in sorted(rb)])
    for variant in ('raw_strict','repaired_strict','raw_normalized','repaired_normalized'):
     out['paired'][k+'/main_Fact_'+variant]=bootstrap([(images[rid],rb[rid]['_Fact_'+variant],ro[rid]['_Fact_'+variant]) for rid in sorted(rb)])
   if (b,lane) in cc and (o,lane) in cc:
    rb,ro=cc[(b,lane)],cc[(o,lane)];out['paired'][k+'/conflict_CR']=bootstrap([(rb[rid][0],rb[rid][1],ro[rid][1]) for rid in sorted(rb)])
 p=H/'RESULTS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');t.replace(p)
 return out
if __name__=='__main__':
 while True:
  try:d=update();print(json.dumps({'pending':len(d['pending']),'updated_epoch':d['updated_epoch']}),flush=True)
  except Exception as e:
   (H/'SUMMARY_ERROR.txt').write_text(repr(e)+'\n');raise
  j=json.loads((H/'JUDGE.json').read_text()) if (H/'JUDGE.json').exists() else {}
  if j.get('state')=='blocked' or (j.get('state')=='all_judging_complete' and not d['pending']):break
  time.sleep(60)
