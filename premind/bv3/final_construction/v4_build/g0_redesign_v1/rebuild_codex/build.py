"""AD immutable release builder. Pure CPU JSON work; no backend/model imports."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
from collections import Counter

HERE=Path(__file__).resolve().parent
G=HERE.parent
FC=HERE.parents[2]
MODES=('retrieved','gt_retrieved','gt_retrieved_drop')
TRAIN=FC/'wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v14'
TEST=FC/'wp_v9_vcm/holdout_v9_sealed/RELEASE_TEST_v4l'
TRAIN_NEW=TRAIN.with_name('RELEASE_TRAIN_DEV_v16')
TEST_NEW=TEST.with_name('RELEASE_TEST_v4m')
PREV=FC/'v4_build/test_freeze_ac_v1/owner_applied_v4l'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def cid(r):return r['evaluator']['row_id']
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def lines(p,rs):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def file(rel,sp,m):return rel/('' if m=='retrieved' else m)/('holdout.jsonl' if sp=='test' else sp+'.jsonl')
def manifest(rel):
    dump(rel/'MANIFEST.json',{'files':{str(p.relative_to(rel)):{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(rel.rglob('*')) if p.is_file()}})

def main():
    for p in [TRAIN_NEW,TEST_NEW,HERE/'BUILD_REPORT.json']:
        if p.exists():raise FileExistsError('Write-once destination exists: '+str(p))
    inv=jl(G/'G0_INVENTORY.json') if False else json.loads((G/'G0_INVENTORY.json').read_text())
    inventory={(r['split'],r['row_id']):r for r in inv};assert len(inventory)==610
    rewrites={(r['split'],r['row_id']):r for r in jl(G/'G0_REWRITES_v2.jsonl')};assert len(rewrites)==247
    assert Counter(s for s,c in rewrites)=={'train':120,'dev':15,'test':112}
    unsafe=json.loads((G/'G0_UNSAFE_GOLD_PATCH_v1.json').read_text())
    # Resolve the two input local IDs in their own original inventory only.
    unsafe_by_cid={}
    for p in unsafe:
        matches=[r for r in inv if r['split']==p['split'] and r['sample_id']==p['sample_id']]
        assert len(matches)==1
        unsafe_by_cid[(p['split'],matches[0]['row_id'])]=p
    assert len(unsafe_by_cid)==2
    all_rows={};source_pins={};stats={};changed=[];provenance=[];seen=set()
    policy=[]
    for sp in ['train','dev','test']:
        source=TEST if sp=='test' else TRAIN;dest=TEST_NEW if sp=='test' else TRAIN_NEW
        for mode in MODES:
            p=file(source,sp,mode);source_pins[str(p)]=sha(p);rows=jl(p)
            assert len({cid(r) for r in rows})==len(rows)
            out=[]
            for source_row in rows:
                r=copy.deepcopy(source_row);key=(sp,cid(r));ev=r['evaluator'];fields=[]
                if r['row_type'].startswith('G0'):
                    assert key in inventory, ('unclassified G0',key,mode)
                    category=1 if key in rewrites else int(inventory[key]['category'][0])
                    assert category in [1,2,3]
                    value='active' if category==1 else 'neutral'
                    r['g0_call_policy']=value;r['g0_category']=category
                    ev['g0_call_policy']=value;ev['g0_category']=category
                    policy.append({'split':sp,'mode':mode,'canonical_id':cid(r),
                         'sample_id':r['sample_id'],'g0_call_policy':value,'g0_category':category})
                    fields+=['g0_call_policy','g0_category','evaluator.g0_call_policy','evaluator.g0_category']
                if key in rewrites:
                    q=rewrites[key];seen.add(key)
                    assert r['row_type'].startswith('G0') and q['family_id']==r['family_id']
                    r.update(query_text=q['new_query'],gold_answer=q['new_gold'],risk='safe',
                        gold_action='direct_answer',gold_action_set=['direct_answer'],
                        gold_citations=[],memory_required=False)
                    ev.update(expected_memory_behavior='ignore',safety_priority='not_safety_relevant',memory_required=False,
                        support_patterns_by_display_index={},wrong_concept_patterns={},decisive_keyphrases=[])
                    assert not ev['acute_visual_hazard']['present'],('QA on acute image',key)
                    for a in ev.get('legacy_arms',{}).values():
                        a.update(gold_answer=q['new_gold'],expected_model_action='direct_answer',
                            gold_action_set=['direct_answer'],gold_citation_set=[],expected_memory_behavior='ignore',
                            support_patterns_by_display_index={},wrong_concept_patterns={},decisive_keyphrases=[])
                    fields+=['query_text','gold_answer','risk','gold_action','gold_action_set','gold_citations',
                         'memory_required','evaluator.QA_labels','evaluator.legacy_arms.QA_labels']
                if key in unsafe_by_cid:
                    assert r['risk']=='unsafe' and r['g0_call_policy']=='neutral'
                    r['gold_answer']=unsafe_by_cid[key]['new_gold']
                    for a in ev.get('legacy_arms',{}).values():a['gold_answer']=r['gold_answer']
                    fields+=['gold_answer','evaluator.legacy_arms.gold_answer']
                if fields:changed.append({'split':sp,'mode':mode,'canonical_id':cid(r),'sample_id':r['sample_id'],'fields':fields})
                provenance.append({'split':sp,'mode':mode,'canonical_id':cid(r),'sample_id':r['sample_id'],
                    'source':str(p),'source_file_sha256':sha_cached(source_pins,p),'changed_fields':fields})
                out.append(r)
            lines(file(dest,sp,mode),out);all_rows[(sp,mode)]=out
            counts=Counter(r.get('g0_category') for r in out if r['row_type'].startswith('G0'))
            stats[sp+'/'+mode]={'rows':len(out),'row_types':dict(Counter(r['row_type'] for r in out)),
                'g0_categories':dict(counts),'rewrites':sum((sp,cid(r)) in rewrites for r in out),
                'unsafe_gold_patches':sum((sp,cid(r)) in unsafe_by_cid for r in out),'sha256':sha(file(dest,sp,mode))}
            for label,select in [('ALL',lambda r:True),('G0_ACTIVE',lambda r:r.get('g0_call_policy')=='active'),
                ('G0_NEUTRAL',lambda r:r.get('g0_call_policy')=='neutral')]:
                p=HERE/'ids'/f'{sp}_{mode}_{label}.txt';p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('x') as f:f.write(''.join(r['sample_id']+'\n' for r in out if select(r)))
    assert seen==set(rewrites),('missing rewrites',set(rewrites)-seen)
    for source,dest in [(TRAIN,TRAIN_NEW),(TEST,TEST_NEW)]:
        for name in ['RETRIEVAL_SIDECARS.json','LICENSES_PER_FILE.json','AC_EXCLUSIONS.json','OWNER_EXCLUSIONS_v9.json','MEDIA_INDEX.jsonl']:
            if (source/name).is_file():
                source_pins[str(source/name)]=sha(source/name);shutil.copy2(source/name,dest/name)
        lines(dest/'ROW_PROVENANCE_AD.jsonl',[p for p in provenance if (p['split']=='test')==(dest==TEST_NEW)])
    lines(HERE/'G0_POLICY.jsonl',policy)
    lines(HERE/'ROW_TEXT_AND_LABEL_CHANGES.jsonl',changed)
    train_ka=FC/'v4_build/train_diversify/TRAIN_KEY_ACTIONS_v14.jsonl'
    source_pins[str(train_ka)]=sha(train_ka)
    qat={c for s,c in rewrites if s in ['train','dev']}
    ka=[r for r in jl(train_ka) if r['row_id'] not in qat]
    lines(HERE/'TRAIN_KEY_ACTIONS_v16.jsonl',ka)
    old_refs=PREV/'KEY_ACTIONS_CANONICAL_v4l.jsonl';source_pins[str(old_refs)]=sha(old_refs)
    refs=[r for r in jl(old_refs) if ('test',r['row_id']) not in rewrites]
    lines(HERE/'KEY_ACTIONS_CANONICAL_v4m.jsonl',refs);refmap={r['row_id']:r for r in refs}
    for mode in MODES:
        lines(HERE/f'TEST_KEY_ACTIONS_v4m_{mode}.jsonl',[
            {**refmap[cid(r)],'row_id':r['sample_id'],'canonical_row_id':cid(r)}
            for r in all_rows[('test',mode)] if cid(r) in refmap])
    old_inv=PREV/'INVENTORY_TEST_v4l_APPROVED.jsonl';source_pins[str(old_inv)]=sha(old_inv)
    mie=jl(old_inv)
    for r in mie:
        assert ('test',r['evaluator_row_id']) not in rewrites
        r['version']='mie_inventory_v4m_owner_approved'
        for mode,b in r['modes'].items():
            row=next(x for x in all_rows[('test',mode)] if cid(x)==r['evaluator_row_id'])
            assert b['sample_id']==row['sample_id'] and b['image_sha256']==row['evaluator']['source_image_sha256']
            assert r['fact_span'] in row['context_texts'][b['display_index']-1]
    lines(HERE/'INVENTORY_TEST_v4m_APPROVED.jsonl',mie)
    old_pin=FC/'eval_v1/PREREG_RELEASE_MAIN_TRAIN_DEV_v14_gt_retrieved_drop.json'
    source_pins[str(old_pin)]=sha(old_pin)
    pin=json.loads(old_pin.read_text());pin['data']['rows']=str(file(TRAIN_NEW,'train','gt_retrieved_drop'))
    pin['data']['sha256']=sha(Path(pin['data']['rows']))
    dump(HERE/'PREREG_RELEASE_MAIN_TRAIN_DEV_v16_gt_retrieved_drop.json',pin)
    for p in [G/'G0_INVENTORY.json',G/'G0_REWRITES_v2.jsonl',G/'G0_UNSAFE_GOLD_PATCH_v1.json',PREV/'BINDING.json']:
        source_pins[str(p)]=sha(p)
    binding={'train_release':str(TRAIN_NEW),'release':str(TEST_NEW),'previous_test':str(TEST),
        'release_sha256':{m:sha(file(TEST_NEW,'test',m)) for m in MODES},
        'key_actions':str(HERE/'KEY_ACTIONS_CANONICAL_v4m.jsonl'),
        'key_actions_sha256':sha(HERE/'KEY_ACTIONS_CANONICAL_v4m.jsonl'),
        'mie_inventory':str(HERE/'INVENTORY_TEST_v4m_APPROVED.jsonl'),
        'mie_inventory_sha256':sha(HERE/'INVENTORY_TEST_v4m_APPROVED.jsonl'),
        'false_call_definition':'no-memory active G0 QA only; neutral G0 call rates reported separately',
        'source_pins':source_pins,'version_name_resolution':'v4l existed before AD; v4m preserves write-once releases',
        'api_authorized':False,'public_distribution_authorized':False}
    dump(HERE/'BINDING.json',binding)
    dump(TEST_NEW/'EVALUATION_BINDING.json',binding)
    for dest in [TRAIN_NEW,TEST_NEW]:
        (dest/'README.md').write_text('# G0 image-grounded QA redesign (AD)\n\nDerived without image or context changes. Explicit QA is active; hazard advice and generic G0 are neutral. Canonical identity and mode-safe joins are used. See '+str(HERE/'BINDING.json')+'. Existing source releases and code are unchanged. Model/API execution and public upload are not authorized by this build.\n')
        manifest(dest)
    dump(HERE/'BUILD_REPORT.json',{'counts':stats,'rewrite_counts':dict(Counter(s for s,c in seen)),
        'unsafe_patch_canonical_ids':[c for s,c in unsafe_by_cid],'train_key_actions':len(ka),
        'test_key_actions':len(refs),'mie_rows':len(mie),'source_pins':source_pins})
    assert all(sha(p)==h for p,h in source_pins.items())
    print(json.dumps({'releases':[str(TRAIN_NEW),str(TEST_NEW)],'rewrites':dict(Counter(s for s,c in seen)),'mie':len(mie)}))

def sha_cached(pins,p):return pins[str(p)]
if __name__=='__main__':main()
