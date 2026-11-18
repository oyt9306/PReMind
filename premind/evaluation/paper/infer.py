"""Approved held-out inference on GPU0/1 only. Local weights, no API. Same fixed release for both conditions."""
import argparse,fcntl,json,os,sys
from pathlib import Path
from ft_common import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('key');ap.add_argument('cohort',choices=['main','conflict']);ap.add_argument('lane',choices=['ondemand','forced']);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
 assert_owner();verify_sources()
 for key in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','GOOGLE_API_KEY','GEMINI_API_KEY']:os.environ.pop(key,None)
 for key in ['STAGE1_FROM','EVAL_ROW_IDS_FILE','MODEL','LORA','GEN_BATCH']:os.environ.pop(key,None)
 os.environ.update(TEST_FREEZE_APPROVED='1',OWNER_EVAL_APPROVED='1',G0_QA_KEYS_REVIEWED='1',FINAL_HOLDOUT='1',EVAL_BACKEND='vllm',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu))
 out=run(a.key,a.cohort,a.lane);size,lora=model(a.key)
 if (out/'eval_report.json').exists():return
 if out.exists():raise RuntimeError('Partial output preserved; inspect before retry: '+str(out))
 if lora:
  if json.loads((lora/'adapter_config.json').read_text())['base_model_name_or_path']!=f'Qwen/Qwen3.5-{size.upper()}':raise RuntimeError('Backbone mismatch')
 if a.lane=='forced':
  od=run(a.key,a.cohort,'ondemand');recs={r['row_id']:r for r in jl(od/'records.jsonl')};texts=jl(od/'texts.jsonl');ref=H/'refs'/a.cohort/a.key
  if not ref.exists():
   ref.mkdir(parents=True)
   with (ref/'texts.jsonl').open('x') as f:
    for t in texts:
     own=recs[t['row_id']]['decision']=='call_pdb'
     f.write(json.dumps({'row_id':t['row_id'],'mode':'retrieved','arm':'memory_call','kind':'greedy','stage1':t['stage1'] if own else '<memory_call>call_pdb</memory_call>','own':own})+'\n')
  os.environ['STAGE1_FROM']=str(ref)
 sys.path[:0]=[str(PKG),str(FC)]
 from v4_build.g0_redesign_v1.rebuild_v20 import policy_ae
 from v4_build.g0_redesign_v1 import qa_rules_v3
 policy_ae.QR=qa_rules_v3;policy_ae.install()
 from final_construction.wp_v9_vcm import s6_render as S
 assert S.systems('prod')['memory_call']==(PKG/'POLICY_SYSTEM_memory_call_original.txt').read_text()
 from eval_v1 import evaluate as E,vllm_backend as VB
 merge=VB.merged_dir
 def local_merge(size,lora):
  cache=H/'merged';cache.mkdir(exist_ok=True)
  with (cache/(sha(Path(lora)/'adapter_model.safetensors')+'.lock')).open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);old=VB.__file__
   try:VB.__file__=str(cache/'eval_v1/vllm_backend.py');return merge(size,lora)
   finally:VB.__file__=old
 VB.merged_dir=local_merge
 release=MAIN if a.cohort=='main' else CR;expected=1028 if a.cohort=='main' else 152
 orig=E.write
 def checked(o,recs,texts,meta,**kw):
  if Path(o).resolve()!=out.resolve() or len(recs)!=expected or len(texts)!=expected or len({r['row_id'] for r in recs})!=expected:raise RuntimeError('Population/output mismatch')
  if a.cohort=='main':
   selector=FC.parent/'eval_inputs'/'main_generated_ids.txt'   # the 1,028 generated main-test rows
   if {r['row_id'] for r in recs}!=set(selector.read_text().splitlines()):raise RuntimeError('Main cohort IDs differ from the paper test population')
  if a.lane=='forced' and any(r['decision']!='call_pdb' for r in recs):raise RuntimeError('Forced-call decision mismatch')
  meta['forced_mix_final_test']={'model':a.key,'condition':a.lane,'cohort':a.cohort,'adapter_sha256':sha(lora/'adapter_model.safetensors') if lora else None,'no_test_based_checkpoint_selection':True,'source_pins_sha256':sha(H/'SOURCE_PINS.json')}
  return orig(o,recs,texts,meta,**kw)
 E.write=checked
 argv=['--release',str(release),'--split','holdout','--final-holdout','--modes','retrieved','--arms','memory_call','--model',size,'--backend','vllm','--samples','0','--context-k','8','--r','.1','--reward-preset','main','--bootstrap','200','--out',str(out),'--online-k','4','--online-query-emb',str(EMB)]
 if lora:argv+=['--lora',str(lora)]
 return E.main(argv)
if __name__=='__main__':sys.exit(main())
