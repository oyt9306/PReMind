"""CPU pin and schedule preparation, never training. All outputs write-once."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
FC=HERE.parents[2]
sys.path[:0]=[str(HERE),str(FC)]
from build import dump,lines,sha,TRAIN_NEW

def old_pin(model):
    stem='single_v4hS_cur22s_v14_'+model+('_CUR22S' if model=='4b' else '_CUR22S_9B')
    suffix='.v4hcur22s_pin.json' if model=='4b' else '.v4hcur22s_9b_pin.json'
    return FC/'eval_v1/runs'/(stem+suffix)

def block(run):
    model=run[run.index('--model')+1]
    if model=='4b':from trainer_v2_v4 import pins_v4hcur22s as pins
    elif model=='9b':from trainer_v2_v4 import pins_v4hcur22s_9b as pins
    else:raise ValueError('unsupported model')
    return pins.block(run)

def pin_block(run):
    result=block(run)
    result['g0_policy']={'version':'g0_qa_active_ad_v1','neutral_decision_credit':0.0,
        'argv_sha256':__import__('hashlib').sha256(json.dumps(run).encode()).hexdigest(),
        'active_weights':'unchanged ws52u1: correct +r, missed -2r, unneeded -r',
        'policy_file':str(HERE/'G0_POLICY.jsonl'),'policy_sha256':sha(HERE/'G0_POLICY.jsonl'),
        'code_sha256':{name:sha(HERE/name) for name in ['g0_policy.py','metrics_ad.py','train_ad.py','prepare.py','launch_ad.py']}}
    return result

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [p for k in sorted(set(a)|set(b)) for p in differences(a.get(k),b.get(k),(path+'.' if path else '')+k)]
    return [] if a==b else [path]

def allowed(path):
    exact=['v4h.release','v4h.release_pin','v4h.release_pin_sha256','v4h.key_actions','v4h.key_actions_sha256',
        'cur.schedule_sha256','cur.replay.budget',
        'aa.warning.refs_sha256.train.jsonl','aa.warning.refs_sha256.dev.jsonl']
    return path in exact or any(path==p or path.startswith(p+'.') for p in ['cur.data','cur.quotas','g0_policy'])

def prepare_one(model):
    shell='v4hcur22s_single_plan.sh' if model=='4b' else 'v4hcur22s_9b_plan.sh'
    case='single_cur22s' if model=='4b' else 'single_cur22s_9b'
    env={**os.environ,'PRINT_ONLY':'1'};env.pop('OPENAI_API_KEY',None)
    # This shell branch only assembles and prints arguments; no model or GPU query.
    proc=subprocess.run(['bash',str(FC/'trainer_v2_v4/scripts'/shell),case],env=env,text=True,capture_output=True,check=True)
    line=[x for x in proc.stdout.splitlines() if ' -m torch.distributed.run ' in x][-1]
    tokens=shlex.split(line);entry='trainer_v2_v4.main_v4hcur22s'+('_9b' if model=='9b' else '')
    run=tokens[tokens.index(entry)+1:]
    if run[:1]==['--']:run=run[1:]
    flag='--prereg-v4hcur22s'+('-9b' if model=='9b' else '')
    idx=run.index(flag);run=run[:idx]+run[idx+2:]
    launched=json.loads(old_pin(model).read_text())
    launched={k:v for k,v in launched.items() if k not in ['cur22b_check','cur22s_check']}
    reconstructed=block(run)
    drift=differences(launched,reconstructed)
    if drift:raise ValueError('Old live CUR22S recipe cannot be reproduced: '+str(drift))
    original=run[:]
    def setarg(k,v):run[run.index(k)+1]=str(v)
    setarg('--release',TRAIN_NEW)
    setarg('--prereg',HERE/'PREREG_RELEASE_MAIN_TRAIN_DEV_v16_gt_retrieved_drop.json')
    setarg('--hazard-key-actions',HERE/'TRAIN_KEY_ACTIONS_v16.jsonl')
    setarg('--out',FC/'eval_v1/runs'/('single_v4mS_cur22s_v16_'+model+'_AD'))
    new=pin_block(run);diff=differences(launched,new)
    extra=[p for p in diff if not allowed(p)]
    if extra:raise ValueError('Undeclared pin changes: '+str(extra))
    dump(HERE/f'CUR22S_{model}_AD_PIN.json',new)
    dump(HERE/f'CUR22S_{model}_ARGS.json',run)
    dump(HERE/f'CUR22S_{model}_PIN_DIFF.json',{'source_pin':str(old_pin(model)),
        'source_pin_sha256':sha(old_pin(model)),'reconstructed_old_pin_diff':drift,
        'diff':diff,'unexpected':extra,'unchanged_recipe_blocks':[k for k in launched if k not in ['v4h','cur'] and launched[k]==new[k]],
        'original_argv':original,'new_argv':run})
    from trainer_v2_v4 import main_v4hcur22 as M
    from trainer_v2_v4 import main_v4hcur22s as S
    opts,rest=S.opt_parser().parse_known_args(run)
    from trainer_v2_v4 import main_v4hcur22b as B
    _,r22=B.resolve(rest)
    cur,r=M.resolve(r22);plan,ids,data,items=M.build_plan(r,cur)
    checks=plan.check(ids)
    dump(HERE/f'CURRICULUM_v16_{model}_REPORT.json',{**plan.report(),'checks':checks,'data':data})
    dump(HERE/f'CURRICULUM_v16_{model}_SCHEDULE.json',plan.schedule)
    # Explicit row lists are regenerated from converted new release inputs.
    from trainer_v2_v4 import curriculum_v1 as C
    for stratum,row_ids in plan.strata.items():
        p=HERE/'curriculum_strata'/f'{model}_{stratum}.txt';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(''.join(x+'\n' for x in row_ids))
    print(model,'pin diff fields',len(diff),'schedule',plan.sha,'checks',checks)

if __name__=='__main__':
    os.environ['CUR_ACUTE_LADDER']='2,2'
    os.environ['PIN_WORLD']='2'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for model in ['4b','9b']:prepare_one(model)
