"""CPU-only diagnostic overlay; never mutate release, target, records or judge verdicts."""
import sys as _s;from pathlib import Path as _P;_s.path.insert(0,str(_P(__file__).resolve().parents[2]));import pathmap;pathmap.install()   # release: original absolute data paths -> local bundle
import sys,json,hashlib,copy
from pathlib import Path
from functools import lru_cache
FC=(Path(__file__).resolve().parents[2] / 'core' / 'final_construction');V4=FC/'v4_build'
REPAIRS=FC.parent/'eval_inputs'/'REPAIRS.json'
FMT=Path(__file__).resolve().parent
sys.path[:0]=[str(FC),str(FMT)]
from trainer_v2 import personal_match as PM
from common_fmt import STRICT_PC,lenient_pc,R2
VERSION='fact_overlay_v1_20261103';VARIANTS=('raw_strict','repaired_strict','raw_normalized','repaired_normalized')
jl=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(None)
def bundle(cohort):
 from eval_v1 import release_data as RD
 release=V4/('release_test_v4q_fix2b/RELEASE_TEST_v4q_fix2b' if cohort=='main' else 'conflict_fix2_final_v3/RELEASE_TEST_v4q_conflict_v10')
 raw={r['sample_id']:r for r in jl(release/'holdout.jsonl')}
 items,_=RD.load_release(release,'holdout','retrieved',arms=('memory_call',),final_holdout=True,online_k=4,online_query_emb=V4/'online/query_emb_test_v4i')
 it={i.row_id:i for i in items};rep={r['id']:r for r in json.loads(REPAIRS.read_text())};assert len(rep)==19
 repaired={};applied={}
 for rid,item in it.items():
  t=copy.deepcopy(item.target);support={int(k):v for k,v in t['support'].items()};indices=[]
  if rid in rep:
   rr=rep[rid];src=raw[rid];concepts=src['evaluator']['context_concepts']
   for fix in rr['idx_repairs']:
    ix=fix['idx'];key=concepts[ix-1]['key']
    if key not in t['rendered_keys']:continue # retrieval miss stays missing; never inject target
    j=t['rendered_keys'].index(key)+1
    if j not in support:continue
    if support[j]!=fix['old_pattern']:raise ValueError('Overlay/version mismatch '+rid)
    support[j]=fix['new_pattern'];indices.append({'release_index':ix,'rendered_index':j,'key':key})
  repaired[rid]=support;applied[rid]=indices
 return raw,it,repaired,applied

def score(run,cohort,selected_ids=None):
 run=Path(run);raw,it,patched,applied=bundle(cohort)
 recs=jl(run/'records.jsonl');texts=jl(run/'texts.jsonl')
 rmap={r['row_id']:r for r in recs};tmap={t['row_id']:t for t in texts}
 if len(rmap)!=len(recs) or len(tmap)!=len(texts) or set(rmap)!=set(tmap):raise ValueError('Duplicate/mismatched rows')
 ids=sorted(set(rmap) if selected_ids is None else set(selected_ids))
 if not set(ids)<=set(rmap):raise ValueError('Missing selected rows')
 result=[];mismatch=[]
 for rid in ids:
  r=rmap[rid];text=tmap[rid];target=it[rid].target
  p=R2.parse(text.get('stage2') or '');answer=p['answer'] or '';cu=p['context_use'];support={int(k):v for k,v in target['support'].items()}
  valid_evidence=r['shown'] and not r['degenerate'] and not target['withhold'] and bool(support)
  row={'row_id':rid,'repair_applied':applied[rid],'stored_personal_full':r['personal_full'],'shown':r['shown'],'degenerate':r['degenerate'],'variants':{}}
  for name in VARIANTS:
   pc=lenient_pc if name.endswith('normalized') else STRICT_PC
   cited,valid=pc(cu,target['k']);sup=patched[rid] if name.startswith('repaired') else support
   flag=bool(valid_evidence and all(j in cited and PM.hit(pat,answer,'soft') for j,pat in sup.items()))
   row['variants'][name]={'Fact':float(flag),'citation_valid':valid,'cited':sorted(cited)}
  if bool(r['personal_full'])!=bool(row['variants']['raw_strict']['Fact']) and not r['degenerate']:mismatch.append(rid)
  result.append(row)
 if mismatch:raise ValueError('Raw strict Fact does not reproduce stored records: '+str(mismatch[:6]))
 n=len(result);summary={v:{'n':n,'Fact':sum(x['variants'][v]['Fact'] for x in result)/n if n else None} for v in VARIANTS}
 return {'version':VERSION,'cohort':cohort,'variants':summary,'primary':'repaired_normalized','repairs_sha256':sha(REPAIRS),'normalizer_sha256':sha(FMT/'common_fmt.py'),'raw_record_reproduction_mismatches':0,'affected_selected_rows':sum(bool(x['repair_applied']) for x in result),'population_note':'Fixed 343 main target-present unique requests; conflict diagnostic uses all 152 generated rows unless caller supplies 140 scored IDs. Does not replace paper 511-entry weighting.','rows':result}

def augment(run,cohort,out,tp):
 result=score(run,'main',tp.keys());by={r['row_id']:r for r in result['rows']}
 out['Fact_overlay']={k:v for k,v in result.items() if k!='rows'}
 out['Fact_primary']=result['variants']['repaired_normalized']['Fact']
 out['Fact_primary_variant']='repaired_normalized'
 for rid,r in tp.items():
  for variant in VARIANTS:r['_Fact_'+variant]=by[rid]['variants'][variant]['Fact']
 # Dedicated sidecar; no writes to raw inference files.
 dst=Path(run)/'FACT_PATTERN_OVERLAY.json';tmp=dst.with_suffix('.overlay.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(dst)
 return out,tp
