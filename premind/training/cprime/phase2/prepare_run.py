"""Phase II-quadruple-prime copy (2026-10-24): p2e release / prereg / report, mode p2e, MIE gate, original stillpen (P1 -0.2), original
-1r unneeded call (decision_p2c not installed), 50 steps.

CPU-only pin of the Phase II' run (= the original CUR22S s125 warm-start stillpen v2 run + p2b release, conflict
stream, MIE term, dev hook, 100 steps). ``python prepare_run.py`` writes (write-once) the warm-start release prereg
(trainer_v2.pins.build_ws over the p2b release prereg) and PIN_4B.json. Copy of train_stillpen_ws_4b_v2/prepare_run.py."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
os.environ.setdefault('PIN_WORLD','2');os.environ.setdefault('CUR_ACUTE_LADDER','2,2')
HERE=Path(__file__).resolve().parent;FC=(Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction');sys.path[:0]=[str(HERE),str(FC)]
import stillpen as SP
import mie_train as MT
import dev_hook_p2b as DH
import train_entry as TE
import monitor_p2c as MON
import mastery_p2d as MP
import api_budget as AB
import advweight_p2e as AW
PY=sys.executable
R20=FC/'v4_build/g0_redesign_v1/rebuild_v20'
ORIGINAL_ARGS=R20/'CUR22S_4b_ARGS.json'
WS4=(Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction') / 'v4_build/freeze_shortcut_fix/train_stillpen_ws_4b'
BASE_PREREG=FC/'v4_build/conflict_train_aug_v1/p2e_release/PREREG_RELEASE_MAIN_TRAIN_DEV_v20_hca1_p2e_gt_retrieved_drop.json'
WS_PREREG=HERE/'PREREG_RELEASE_WS_P2E_v1_s125.json'
HCA1=WS4/'hca1'
RELEASE=FC/'wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1_p2e'
P2B_REPORT=FC/'v4_build/conflict_train_aug_v1/p2b_release/BUILD_REPORT_P2B_v1.json'
P2E_REPORT=FC/'v4_build/conflict_train_aug_v1/p2e_release/BUILD_REPORT_P2E_v1.json'
V2_REPORT=FC/'v4_build/conflict_train_aug_v1/BUILD_REPORT_conflict_train_v2.json'
INIT_LORA=Path(os.environ.get('PREMIND_PHASE1_RUN', 'runs/cprime_p1_4b_s0')) / 'checkpoints/step_000125/lora'
ADAPTER_SHA='cd807c90201add0947dcc1f543668769537fd7d56527e95ad6e7b6b8eafaabd4'
GPUS=(4, 5);WORLD=2
# Every argv difference from the original CUR22S 4B AG run (flag -> new value); anything else refuses.
DECLARED={'--checkpoint-every':'5','--wandb-group':'factpc_p2','--out':'runs/cprime_p2_4b_s0',
    '--release':str(RELEASE),'--hazard-key-actions':str(SP.KEY_ACTIONS),
    '--dev-every':'25','--hazard-usd-cap':'0.80','--warning-usd-cap':'0.25','--prereg':str(WS_PREREG),
    '--abl-stop-at-step':str(TE.STOP_AT),'--init-lora':str(INIT_LORA)}
ENTRY_MODULES=['trainer_v2_v4.main_v4hcur22s','trainer_v2_v4.main_v4hcur22b','trainer_v2_v4.main_v4hcur22',
    'trainer_v2_v4.main_v4hcur21','trainer_v2_v4.main_v4hcur','trainer_v2_v4.main_v4haa','trainer_v2_v4.main_v4h',
    'trainer_v2_v4.main_v4','trainer_v2_v4.main_v4b','trainer_v2_v4.main_v4hcur23r','trainer_v2_v4.patch','trainer_v2_v4.patch_h',
    'trainer_v2_v4.reward_v4hg','trainer_v2_v4.hazard_rubric','trainer_v2_v4.hazard_judge','trainer_v2_v4.hazard_judge_api',
    'trainer_v2_v4.warning_credit','trainer_v2_v4.recall_decision_u1','trainer_v2_v4.abl_hooks',
    'trainer_v2_v4.curriculum_v2_2','trainer_v2_v4.key_actions_v4h','trainer_v2_v4.acute_memory_bonus_v1_1',
    'trainer_v2_v4.false_alarm_pen','trainer_v2_v4.absent_visible_risk','trainer_v2_v4.kl_ref','trainer_v2_v4.covip_aux',
    'eval_v1.ddp_main','eval_v1.main_run','eval_v1.release_data','eval_v1.api_providers','eval_v1.stop_rule',
    'eval_v1.absent_sampler','trainer_v2.grpo_v2','trainer_v2.algo_v2','trainer_v2.run_v2','trainer_v2.ddp_v2',
    'trainer_v2.pins','trainer_v2.records_v2','trainer_v2.reward_v5','trainer_v2.hf_backend',
    'v4_build.g0_redesign_v1.rebuild_v20.policy_ae','v4_build.g0_redesign_v1.rebuild_v20.balance_ae',
    'v4_build.g0_redesign_v1.rebuild_v20.prepare_ae','v4_build.mie_metric.mie_v1','eval_v1.acute_adequacy_v1',
    'eval_v1.evaluate','eval_v1.wandb_run']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen_sources():
    """Every module file under final_construction (package excluded) that the entry chain imports, listed by a fresh
    CPU-only interpreter (CUDA hidden), so the set does not depend on the calling process."""
    code=('import sys,json,importlib;sys.path[:0]=[%r,%r]\n'%(str(FC),str(FC.parent))+
          'for m in %r:importlib.import_module(m)\n'%ENTRY_MODULES+
          'print(json.dumps(sorted({getattr(m,"__file__",None) or "" for m in list(sys.modules.values())})))')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PIN_WORLD':str(WORLD),'CUR_ACUTE_LADDER':'2,2','PYTHONDONTWRITEBYTECODE':'1'}
    out=subprocess.run([PY,'-B','-c',code],env=env,cwd=str(FC),capture_output=True,text=True,check=True).stdout
    files=[Path(f).resolve() for f in json.loads(out.strip().splitlines()[-1]) if f]
    keep=sorted(f for f in files if f.suffix=='.py' and FC in f.parents and HERE not in f.parents)
    return {str(f.relative_to(FC)):sha(f) for f in keep}
def argv_diff(old,new):
    def pairs(a):
        out,i={},0
        while i<len(a):
            if a[i].startswith('--'):
                v=a[i+1] if i+1<len(a) and not a[i+1].startswith('--') else None
                out[a[i]]=v;i+=2 if v is not None else 1
            else:raise ValueError('positional argv '+a[i])
        return out
    o,n=pairs(old),pairs(new)
    return {f:[o.get(f),n.get(f)] for f in sorted(set(o)|set(n)) if o.get(f)!=n.get(f)}
def orig_state_line():
    for line in SP.ORIG_CUR_LOG.read_text().splitlines():
        if json.loads(line)['step']==SP.ORIG_STATE_STEP:return hashlib.sha256(line.encode()).hexdigest()
    raise ValueError('no original curriculum state')
def block(args):
    from v4_build.g0_redesign_v1.rebuild_v20 import prepare_ae as OLD
    original=json.loads(ORIGINAL_ARGS.read_text())
    diff=argv_diff(original,args)
    if {f:v[1] for f,v in diff.items()}!=DECLARED:raise ValueError('undeclared argv difference vs CUR22S AG: '+json.dumps(diff))
    if sha(INIT_LORA/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('warm-start adapter changed')
    rep,rep_e=json.loads(P2B_REPORT.read_text()),json.loads(P2E_REPORT.read_text())
    from trainer_v2_v4 import recall_decision_u1 as RDU
    return {'version':'p2e_mie_v1 (Phase II-quadruple-prime = p2d + move-required conflict rows, MIE hazard gate, original stillpen -0.2, natural conflict share, G0 stream 0.02, original -1r unneeded call, 50 steps)','argv':args,'original_args':str(ORIGINAL_ARGS),'original_args_sha256':sha(ORIGINAL_ARGS),
      'argv_diff_vs_original':diff,'base_recipe':OLD.block(original),'frozen_source_sha256':frozen_sources(),
      'package_code_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')},
      'warm_start':{'lora':str(INIT_LORA),'adapter_sha256':ADAPTER_SHA,'adapter_config_sha256':sha(INIT_LORA/'adapter_config.json'),
                    'complete':(INIT_LORA.parent/'COMPLETE').read_text().strip(),'optimizer':'fresh AdamW (ddp_main warm start)'},
      'ws_prereg_sha256':sha(WS_PREREG),'key_actions_sha256':sha(SP.KEY_ACTIONS),'release_train_sha256':sha(SP.RELEASE_TRAIN),
      'hca1':{'base_prereg_sha256':sha(BASE_PREREG),'release_manifest_sha256':sha(RELEASE/'MANIFEST.json'),
              'files_sha256':{p.name:sha(p) for p in sorted(HCA1.iterdir()) if p.is_file() and p.suffix in ('.py','.jsonl','.json','.txt')},
              'plan':'hca1 rows outside the curriculum plan (stillpen.make_plan_class; schedule sha unchanged)'},
      'curriculum_state':{'log':str(SP.ORIG_CUR_LOG),'step':SP.ORIG_STATE_STEP,'cursor':SP.ORIG_CURSOR,'line_sha256':orig_state_line(),
                          'step_offset':SP.STEP_OFFSET},
      'penalty':{'lambda':SP.LAMBDA,'still':SP.STILL.pattern,'emerg':SP.EMERG.pattern,'flags':'IGNORECASE',
                 'terms':{'p1':'acute & not stillness_supported & STILL','p2':'acute & emergency_contraindicated & EMERG',
                          'p3':'non-acute safe & (STILL | EMERG)','p4':'move_required (hca1) & STILL, additive to p1','p5':'stillness-supported acute & no STILL match, weight LAMBDA_P5'},'lambda_p5':SP.LAMBDA_P5,'applied_to':'hazard_rubric.answer_text(stage2)','step_log_key':'stillpen'},
      'p2b_release':{'dir':str(FC/'wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20_hca1_p2b'),'build_report_sha256':sha(P2B_REPORT),
                     'n_train':rep['n_train'],'n_dev':rep['n_dev'],'excluded_cdev_family_rows':rep['n_excluded_cdev_family_rows'],
                     'excluded_families':rep['excluded_families'],'n_added_conflict':rep['n_added_conflict'],
                     'builder_sha256':sha(FC/'v4_build/conflict_train_aug_v1/p2b_release/build_release_p2b.py')},
      'p2e_release':{'dir':str(RELEASE),'manifest_sha256':sha(RELEASE/'MANIFEST.json'),'build_report_sha256':sha(P2E_REPORT),
                     'n_train':rep_e['n_train'],'n_dev':rep_e['n_dev'],'n_added_conflict_v2':rep_e['n_added_conflict_v2'],
                     'n_mie_inventory':rep_e['n_mie_inventory'],'move_required_share_of_conflict':rep_e['move_required_share_of_conflict'],
                     'builder_sha256':sha(FC/'v4_build/conflict_train_aug_v1/p2e_release/build_release_p2e.py'),
                     'conflict_v2_builder_sha256':sha(FC/'v4_build/conflict_train_aug_v1/build_conflict_train_v2.py'),
                     'conflict_v2_report_sha256':sha(V2_REPORT)},
      'mie':{'w':MT.RUN_W,'w_not_applied':MT.W_NOT_APPLIED,'w_mentioned_only':MT.W_MENTIONED,'gate':MT.RUN_GATE,
             'gate_rule':'applied paid only if final ledger v4h state == specific OR contraindicated (else applied_ungated, 0)',
             'module_defaults':[MT.DEFAULT_W,MT.DEFAULT_W_NA,MT.DEFAULT_W_MO,MT.DEFAULT_GATE],'signs':MT.SIGN,
             'inventory':str(MT.INVENTORY),'inventory_sha256':sha(MT.INVENTORY),'usd_cap':MT.MIE_USD_CAP,
             'judge_input':'hazard_rubric.answer_text(stage2) only','install_order':'after stillpen.install()'},
      'dev_hook':{'full_dev':'dev_eval/step_N (109 rows)','subset':DH.SUBSET_DIR+'/step_N (90 orig rows)','payload':'90 orig rows',
                  'mie':'dev_eval/step_N/mie.json (19 cdev rows)','cdev_inventory_sha256':sha(DH.CDEV_INV),'usd_cap':DH.DEV_MIE_USD_CAP,
                  'dev_every':args[args.index('--dev-every')+1],'checkpoint_every':args[args.index('--checkpoint-every')+1]},
      'streams':{'mode':'p2e','p_given_non_hca1_position':SP.STREAMS_P2E,'conflict_displaces':'present memory-required curriculum rows only','seed':SP.STREAM_SEED,'neutral_styles':list(SP.NEUTRAL_STYLES),
                 'pools':{'conflict':'MIE inventory rows (blocked in the curriculum)','g0':'G0* rows, no key action, query_style not panic/calm (not blocked)'},
                 'natural_conflict_share':rep_e['natural_conflict_share']},
      'decision':{'unneeded_call_x_r':RDU.UNNEEDED_CALL,'missed_call_x_r':RDU.MISSED_CALL,'neutral_g0':0.0,'override':None,
                  'source':'trainer_v2_v4/recall_decision_u1 (ws52u1) unchanged = the original Phase II -1r (decision_p2c not installed)'},
      'monitors':{'keys':list(MON.KEYS),'step_key':'p2c_monitor','p2d_step_keys':['p2d_mastery','retention'],
                  'dev_retention':{'reference_correct_withhold':DH.REF_CORRECT_WITHHOLD,'warn_drop':DH.WARN_DROP}},
      'bands':{**SP.BANDS,'tiers_file':str(SP.TIERS),'tiers_sha256':sha(SP.TIERS),'counts':json.loads(SP.TIERS.read_text())['counts'],
               'curation_tiers_sha256':sha(SP.CUR_TIERS),'profile_tool_sha256':sha(SP.TIERS.parent.parent/'profile_p2e/profile_p2e.py')},
      'advantage_weight':{'core_first':AW.W_CORE_FIRST,'core_replay':AW.W_CORE_REPLAY,'late':AW.W_LATE,'other':1.0,'cap':AW.CAP,
                          'mechanism':'grpo_v2.grouped_advantages x per-row weight (as train_mie_p3_4b/replay_p3.install_weight); rewards unchanged'},
      'mastery':{**SP.MASTERY_CFG,'seed':SP.MASTERY_SEED,'mirrors':'trainer_v2_v4/curriculum_v2.py MASTERY / REPLAY / MAX_VISITS acute',
                 'temperature':'grpo_v2.rollout temperature argument, separate call for HOT prompts (mastery_p2d.split_rollout)'},
      'boost':{'mode':'p2e',**SP.MODES['p2e'],'seed':SP.BOOST_SEED,'pools':'all boost positions: hca1 rows; v20 acute row only if all 24 are gap-blocked (counted)',
               'order':'image round-robin per pool','curriculum_bookkeeping':'excluded (n/counts/last/observe/requeue)'},
      'hazard_first_gate':{'rule':'stillpen.install_hf_gate: after the existing gate (specific & not contraindicated) the bonus needs a hazard-specific key-action cue in the first two answer sentences','excluded_labels':list(SP.HF_EXCLUDED),'v2_still_supported':"stillness-supported rows: dont_move_keep_still cues + STILL count; call_emergency never"},
      'layout':{'world':WORLD,'prompts_per_rank':2,'global_prompts_per_step':4,'gen_batch':16,'num_generations':8,'gpus':list(GPUS)},
      'owner_cap_usd':{'total':AB.CAP_USD,'hazard':float(args[args.index('--hazard-usd-cap')+1]),
                       'warning':float(args[args.index('--warning-usd-cap')+1]),'mie':MT.MIE_USD_CAP,'mie_dev':DH.DEV_MIE_USD_CAP},
      'forced_trial':{'forced_mix_kwargs':'seed=0,p_present=0.40,p_absent=0.40,p_safe=0.50,w6=0.10,exclude_safe=False,w_f=0.15','fact_bonus':{'w_f':0.15,'min_words':8,'echo_words_log_only':15,'rule':'forced_mix.fact_bonus_check: +w_f on FULL use (existing gate); needs call shown, gold cited, no harm flag, >=8 words','targeted_replay':{'rule':'stillpen._fact_observe/_fact_pick: groups with BOTH a safe full-use hit and a fact-omission miss; one replay per row; <=1 per step; only in existing replay slots with no due conflict entry; all-fail groups logged not replayed','cfg':'stillpen.FACT_REPLAY'},'base_pkg':'forced_p2_factbonus_v1/pkg_factbonus_p2','base_forced_mix_sha256':'cd8a99f89fc239c881da5eb837bf91c89adb7fde9ca2097817d21b2fc016af3f'},'gt_clean_edits_sha256':sha(str((Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction') / 'cprime_inputs' / 'GT_CLEAN_EDITS.jsonl')),'key_actions_gtclean_sha256':sha(str((Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction') / 'cprime_inputs' / 'TRAIN_KEY_ACTIONS_P2E_v1_gtclean.jsonl')),'mie_reward_enabled':False,'mie_judging_for_replay':True,'p6':'forced_mix W6 on both phases; stillpen P1-P5 unchanged (P4 0.2)','phase1_out':'${PHASE1_RUN}','phase1_curriculum_log_is_this_trial':True},
      'steps_run':TE.STOP_AT,'execute_env':TE.EXEC_FLAG+'=1'}
def write_ws_prereg():
    from trainer_v2 import pins
    from trainer_v2_v4 import main_v4h as MH
    MH.release_pin_presets()
    rows=Path(json.loads(BASE_PREREG.read_text())['data']['rows'])
    pre=pins.build_ws(BASE_PREREG,rows,INIT_LORA)
    with WS_PREREG.open('x') as f:f.write(json.dumps(pre,indent=1)+'\n')
    pins.verify(WS_PREREG,rows=rows)
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if not WS_PREREG.exists():write_ws_prereg();print('WS prereg written')
    args=json.loads((HERE/'ARGS_4B.json').read_text())
    with (HERE/'PIN_4B.json').open('x') as f:f.write(json.dumps(block(args),indent=2,default=str)+'\n')
    print('Pin written')
