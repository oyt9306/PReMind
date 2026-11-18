"""CPU-only result assembly shared with Y. No backend client is instantiated."""
import hashlib,json,random,sys,os
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;FC=HERE.parents[1];Y=FC/'v4_build/v4i_freeze_codex'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True;sys.path[:0]=[str(Y),str(FC),str(FC/'paper_artifacts_v2'),str(FC/'v4_build/v4g_eval_codex')]
from safe_identity import Catalog,canonical,episode,jl,sha
import acute_v4_bridge as A4
import scoring_l_v2 as S
import assemble as BASE
from eval_v1 import metrics_v6 as M6
GROUPS={'person_distress':'G1','fall':'G1','fire_heat':'G2','electrical':'G2','chemical':'G2','combustion_fumes':'G2','battery_overheat':'G2','machine':'G3','falling_object':'G3','sharp':'G3','animal_attack':'G4','stinging_insects':'G4','venomous_creature':'G4','wild_or_aggressive_animal':'G4','water_flood':'G5','intruder':'G5','choking':'G5','structural_hazard':'G5','storm_damage':'G5'}
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(x,h,indent=1,ensure_ascii=False);h.write('\n')
def lines(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:
  for r in rows:h.write(json.dumps(r,ensure_ascii=False)+'\n')
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def config(p):return json.loads(Path(p).read_text())
def summary(pairs,draws=1000,seed=0):
 pairs=[(c,float(v)) for c,v in pairs if v is not None]
 if not pairs:return {'estimate':None,'n':0,'n_images':0,'ci95':None}
 groups=defaultdict(list)
 for c,v in pairs:groups[c].append(v)
 keys=sorted(groups);rg=random.Random(seed);bs=[]
 sums={k:sum(groups[k]) for k in keys};counts={k:len(groups[k]) for k in keys}
 for _ in range(draws):
  pick=rg.choices(keys,k=len(keys));bs.append(sum(sums[k] for k in pick)/sum(counts[k] for k in pick))
 bs.sort();return {'estimate':sum(v for _,v in pairs)/len(pairs),'n':len(pairs),'n_images':len(keys),'ci95':[bs[int(.025*draws)],bs[min(draws-1,int(.975*draws))]]}
def ba(needed,unneeded,draws=1000):
 if not needed or not unneeded:return summary([])
 # Resample image clusters jointly; each stratum keeps its own denominator.
 groups=defaultdict(lambda:[[],[]])
 for j,ps in enumerate((needed,unneeded)):
  for c,v in ps:groups[c][j].append(float(v))
 keys=sorted(groups);rng=random.Random(0);bs=[]
 for _ in range(draws):
  ix=rng.choices(keys,k=len(keys));xs=[[x for k in ix for x in groups[k][j]] for j in (0,1)]
  if all(xs):bs.append(sum(sum(x)/len(x) for x in xs)/2)
 bs.sort();return {'estimate':(sum(v for _,v in needed)/len(needed)+sum(v for _,v in unneeded)/len(unneeded))/2,'n':len(needed)+len(unneeded),'n_images':len(keys),'ci95':[bs[int(.025*len(bs))],bs[min(len(bs)-1,int(.975*len(bs)))]] if bs else None}
def paths(model,role):return [Path(p) for p in model.get('runs',{}).get(role,[]) if (Path(p)/'records.jsonl').is_file() and (Path(p)/'texts.jsonl').is_file() and (Path(p)/'eval_report.json').is_file()]
def row_groups(item):
 g={GROUPS[t] for t in item['hazard_types']}
 return g|{'ALL'}|({'UNSEEN'} if g-{'G1'} else set())
