"""AE CPU curriculum/pins; runtime library import preserves launch environment."""
from pathlib import Path
import json,os,sys
HERE=Path(__file__).resolve().parent;FC=HERE.parents[2]
sys.path.insert(0,str(FC))
from v4_build.g0_redesign_v1.rebuild_codex import prepare as AD
from v4_build.g0_redesign_v1.rebuild_v20 import balance_ae as B
from v4_build.g0_redesign_v1.rebuild_codex.build import sha,dump
CODE=['policy_ae.py','qa_rules_v2.py','balance_ae.py','prepare_ae.py','train_ae.py','launch_ae.py']
def block(run):
    B.install();out=AD.pin_block(run)
    out['g0_policy'].update(policy_file=str(HERE/'G0_POLICY.jsonl'),policy_sha256=sha(HERE/'G0_POLICY.jsonl'),
        ae_code_sha256={name:sha(HERE/name) for name in CODE},qa_keys_sha256=sha(HERE/'G0_QA_KEYS_v20_FINAL.jsonl'))
    return out
def allowed(path):
    return path in ['v4h.release','v4h.release_pin','v4h.release_pin_sha256','v4h.key_actions','v4h.key_actions_sha256',
        'aa.warning.refs_sha256.train.jsonl','aa.warning.refs_sha256.dev.jsonl',
        'cur.data.release_train','cur.data.release_train_sha256','cur.schedule_sha256',
        'g0_policy.policy_file','g0_policy.policy_sha256','g0_policy.argv_sha256','g0_policy.qa_keys_sha256'] or path=='g0_policy.ae_code_sha256' or path.startswith('g0_policy.ae_code_sha256.')
def main():
    ad=HERE.parent/'rebuild_v19'
    for model in ['4b','9b']:
        run=json.loads((ad/f'CUR22S_{model}_ARGS.json').read_text())
        mapping={'--release':FC/'wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20',
            '--prereg':HERE/'PREREG_RELEASE_MAIN_TRAIN_DEV_v20_gt_retrieved_drop.json',
            '--hazard-key-actions':HERE/'TRAIN_KEY_ACTIONS_v20.jsonl',
            '--out':FC/'eval_v1/runs'/f'single_v4qS_cur22s_v20_{model}_AG'}
        for flag,value in mapping.items():run[run.index(flag)+1]=str(value)
        out=block(run);old=json.loads((ad/'pins_final'/f'CUR22S_{model}_AG_PIN.json').read_text())
        diff=AD.differences(old,out);extra=[p for p in diff if not allowed(p)]
        if extra:raise ValueError('undeclared AG v19 pin differences '+str(extra))
        assert old['cur']['quotas']==out['cur']['quotas']
        dump(HERE/'pins'/f'CUR22S_{model}_AG_PIN.json',out)
        dump(HERE/f'CUR22S_{model}_ARGS.json',run)
        dump(HERE/f'PIN_DIFF_{model}.json',{'vs':str(ad/'pins_final'/f'CUR22S_{model}_AG_PIN.json'),'diff':diff,'unexpected':extra,'quotas_identical':True})
        from trainer_v2_v4 import main_v4hcur22 as M,main_v4hcur22s as S,main_v4hcur22b as MB
        _,rest=S.opt_parser().parse_known_args(run);_,rest=MB.resolve(rest);cur,rest=M.resolve(rest)
        on,ids,data,_=M.build_plan(rest,cur);on_checks=on.check(ids)
        B.C22.Plan22=B.Original
        try:off,ids,_,_=M.build_plan(rest,cur);off_checks=off.check(ids)
        finally:B.install()
        required=B.trap_ids();safe=set(on.strata['safe_present'])|set(on.strata['safe_absent'])
        assert all(a==b for a,b in zip(on.schedule,off.schedule) if a not in safe or b not in safe)
        report={'on':{**on.report(),'checks':on_checks,'exposed_traps_by150':len(required&set(on.schedule[:600]))},
            'off':{**off.report(),'checks':off_checks,'exposed_traps_by150':len(required&set(off.schedule[:600]))},
            'required_exposed_traps':len(required),'same_slot_table':on.quotas==off.quotas,'same_non_safe_slots':True}
        dump(HERE/f'CURRICULUM_{model}.json',report);dump(HERE/f'SCHEDULE_{model}.json',on.schedule)
        print(model,'schedule',on.sha,'on/off exposed',report['on']['exposed_traps_by150'],report['off']['exposed_traps_by150'])
if __name__=='__main__':
    os.environ['PIN_WORLD']='2';os.environ['CUR_ACUTE_LADDER']='2,2'
    main()
