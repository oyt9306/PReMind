"""Phase I of the forced-call trial = rebuild_v20.train_ae (paper AG launch) + forced_mix hooks.  CPU pin check first."""
import sys as _s;from pathlib import Path as _P;_s.path.insert(0,str(_P(__file__).resolve().parents[3]));import pathmap;pathmap.install()   # release: original absolute data paths -> local bundle
from pathlib import Path
import json,os,sys
HERE=Path(__file__).resolve().parent;FC=(Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction')
sys.dont_write_bytecode=True;sys.path[:0]=[str(HERE),str(FC)]
EXEC_FLAG='FORCED_P1_EXECUTE';PILOT_ENV='FORCED_P1_PILOT_STOP_AT'
GT_EDITS=str((Path(__file__).resolve().parents[3] / 'bv3' / 'final_construction') / 'cprime_inputs' / 'GT_CLEAN_EDITS.jsonl')
def block(args):
    """Paper AG pin block of THIS argv (prepare_ae.block) + the forced-trial additions."""
    from trainer_v2_v4 import main_v4hcur22s as E0
    E0.STOP_AT=125                              # in-process only (the module refuses != 150); same as Phase II's ENTRY.STOP_AT
    from v4_build.g0_redesign_v1.rebuild_v20 import prepare_ae as P
    from hashlib import sha256
    sh=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    return {'ae_block':json.loads(json.dumps(P.block(args),default=str)),
            'forced_trial':{'forced_mix_py_sha256':sh(HERE/'forced_mix.py'),'forced_mix_kwargs':'seed=0,p_present=0.40,p_absent=0.40,p_safe=0.50,w6=0.10,exclude_safe=False',
                            'gt_clean_edits':GT_EDITS,'gt_clean_edits_sha256':sh(GT_EDITS),'train_entry_sha256':sh(__file__),
                            'steps':125}}
def pilot_args(args,n):
    if not 1<=n<=5:raise RuntimeError('pilot steps must be 1-5')
    a=list(args);i=a.index('--out')+1;a[i]=a[i]+f'_pilot{n}'
    i=a.index('--abl-stop-at-step')+1;a[i]=str(n)
    return a
def install_all(args):
    from v4_build.g0_redesign_v1.rebuild_v20 import balance_ae as B,policy_ae as POLICY
    import forced_mix as FM
    out=Path(args[args.index('--out')+1])
    B.install();POLICY.install()                      # = rebuild_v20.train_ae (the original AG launch)
    FM.install_gt_clean(Path(GT_EDITS))
    FM.install(out,seed=0,p_present=0.40,p_absent=0.40,p_safe=0.50,w6=0.10,exclude_safe=False)                            # last: outermost
    from trainer_v2_v4 import main_v4hcur22s as E
    E.STOP_AT=125
    return E
def load_args(name):
    """args_4b.json with ${FC}/${PKG}/${OUT}/${PHASE1_RUN} expanded (release; the original run used absolute paths)."""
    sub={'${FC}':str(FC),'${PKG}':str(HERE),'${OUT}':os.environ.get('PREMIND_OUT','runs/cprime_%s_4b_s0'%name),
         '${PHASE1_RUN}':os.environ.get('PREMIND_PHASE1_RUN','runs/cprime_p1_4b_s0')}
    out=[]
    for a in json.loads((HERE/'args_4b.json').read_text()):
        for k,v in sub.items():a=a.replace(k,v)
        out.append(a)
    return out
def pin_check(ok):
    """The original run compared a pin block (absolute paths + file hashes) to PIN_*.json; the release keeps the
    original pins in provenance/ and only checks them on request (PREMIND_PIN_CHECK=1, same machine layout)."""
    if os.environ.get('PREMIND_PIN_CHECK')=='1' and not ok():raise RuntimeError('pin mismatch')
def main():
    os.environ['PIN_WORLD']='2';os.environ['CUR_ACUTE_LADDER']='2,2'
    args=load_args('p1')
    pin_check(lambda:block(args)==json.loads((HERE.parent/'provenance'/'PIN_P1.json').read_text()))
    if os.environ.get(EXEC_FLAG)!='1':raise RuntimeError(f'Execution flag missing ({EXEC_FLAG}=1)')
    n=os.environ.get(PILOT_ENV)
    if n:args=pilot_args(args,int(n))
    return install_all(args).main(args)
if __name__=='__main__':raise SystemExit(main())
