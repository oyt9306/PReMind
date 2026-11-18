"""Phase II-quadruple-prime (2026-10-24, PREREG_PHASE3_MIE.md Amendment 9 + owner updates 2026-10-24) =
train_p2d_mie_4b/train_entry.py with stillpen mode "p2e" (p2e release incl. 87 move-required conflict rows; conflict
stream at the natural share; G0 replay stream 0.02; stillpen unchanged: unsupported stay-still -0.2), mie_train with the hazard gate
(applied paid only if the hazard state is specific and not contraindicated), the ORIGINAL unneeded-call penalty -1r of
recall_decision_u1 / ws52u1 (decision_p2c NOT installed), STOP_AT = 50 (checkpoints / dev_eval at 25 and 50), flags P2E_*.

Phase II-triple-prime = train_p2c_mie_4b/train_entry.py with stillpen mode "p2d" (mastery-aware conflict stream,
PMR-only displacement) + mastery_p2d (outcome MIE labels, no_signal temperature 1.1, p2d / retention step blocks;
installed last), flags P2D_*.

Phase II'' (REWARD_BREAKEVEN_v1.md section 5) = train_p2b_mie_4b/train_entry.py with: unneeded call -2r
(decision_p2c), stillpen mode "p2c" (conflict stream ~1.5x natural + non-acute G0 replay stream ~12 %), MIE mentioned_only
-0.05, hacking monitors (monitor_p2c, installed last), flags P2C_*.

Phase II' (PREREG_PHASE3_MIE.md Amendment 2 + owner amendments 2026-10-23): the original Phase II run
(train_stillpen_ws_4b_v2/train_entry.py: CUR22S 4B warm-started from the AG step-125 LoRA, restored curriculum state,
stillpen v2, lr 1e-5) re-run with
  * the p2b release (hca1 train minus the 4 conflict-dev families + 63 conflict rows; dev 90 + 19 conflict-dev rows)
  * stillpen mode "p2b" (conflict stream at 1/4 of the draw positions, curriculum otherwise restored and unchanged)
  * mie_train (applied +RUN_W, misapplied -RUN_W, not_applied -W_NOT_APPLIED) installed after stillpen.install()
  * dev_hook_p2b (full 109-row dev_eval + 90-row subset report / stop-rule payload + conflict-dev MIE judge)
  * STOP_AT = 100 (checkpoints / dev_eval every 25)
Frozen sources are imported read-only; module constants of main_v4hcur22s (REFUSED, STOP_AT) are overridden
in-process only."""
import sys as _s;from pathlib import Path as _P;_s.path.insert(0,str(_P(__file__).resolve().parents[3]));import pathmap;pathmap.install()   # release: original absolute data paths -> local bundle
from pathlib import Path
import json,os,sys
HERE=Path(__file__).resolve().parent;FC=(Path(__file__).resolve().parents[3] / 'core' / 'final_construction')
sys.path[:0]=[str(HERE),str(FC)]
STOP_AT=25
EXEC_FLAG='P2E_EXECUTE'
PILOT_ENV='P2E_PILOT_STOP_AT'          # cost pilot: stop after N (1-5) steps, out -> <out>_pilot<N> (after the pin check)
def pilot_args(args,n):
    if not 1<=n<=5:raise RuntimeError('pilot steps must be 1-5')
    a=list(args);i=a.index('--out')+1;a[i]=a[i]+f'_pilot{n}'
    i=a.index('--abl-stop-at-step')+1;a[i]=str(n)
    return a
def install_all(args,stop_at=STOP_AT,mie_call=None,dev_call=None):
    """Every in-process hook, in launch order (CPU tests pass fake judge calls)."""
    from v4_build.g0_redesign_v1.rebuild_v20 import balance_ae as B,policy_ae as POLICY
    from trainer_v2_v4 import main_v4hcur22s as ENTRY
    import api_budget,stillpen,mie_train,dev_hook_p2b,monitor_p2c,mastery_p2d,advweight_p2e
    out=Path(args[args.index('--out')+1])
    B.install();POLICY.install()                      # = rebuild_v20.train_ae (the original AG launch)
    api_budget.install(out)
    stillpen.install('p2e')
    mie_train.install(out,w=mie_train.RUN_W,w_na=mie_train.W_NOT_APPLIED,w_mo=mie_train.W_MENTIONED,gate=mie_train.RUN_GATE,call=mie_call,reward_enabled=False)   # MIE judged for replay only
    dev_hook_p2b.install(out,call=dev_call)
    # unneeded call: the original ws52u1 -1r (owner 2026-10-24: decision_p2c's -2r override dropped)
    monitor_p2c.install(stillpen.load_labels(conflict=stillpen.conflict_ids()))
    advweight_p2e.install(stillpen.tier_map())        # per-size tiers: core 1.0 / 1.5 on replay, late 0.5, learned 0.5, cap 1.25x
    mastery_p2d.install()                             # (rollout temperature split)
    import forced_mix as FM;FM.install_gt_clean((Path(__file__).resolve().parents[3] / 'core' / 'final_construction') / 'cprime_inputs' / 'GT_CLEAN_EDITS.jsonl');FM.install(out,seed=0,p_present=0.40,p_absent=0.40,p_safe=0.50,w6=0.10,exclude_safe=False,w_f=0.15)   # forced trial: last = outermost
    ENTRY.REFUSED=tuple(f for f in ENTRY.REFUSED if f!='--init-lora');ENTRY.STOP_AT=stop_at
    return ENTRY
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
    from prepare_run import block
    args=load_args('p2')
    pin_check(lambda:block(args)==json.loads((HERE.parent/'provenance'/'PIN_4B.json').read_text()))
    if os.environ.get(EXEC_FLAG)!='1':raise RuntimeError(f'Execution flag missing ({EXEC_FLAG}=1)')
    n=os.environ.get(PILOT_ENV)
    if n:args=pilot_args(args,int(n));return install_all(args,int(n)).main(args)
    return install_all(args).main(args)
if __name__=='__main__':raise SystemExit(main())
