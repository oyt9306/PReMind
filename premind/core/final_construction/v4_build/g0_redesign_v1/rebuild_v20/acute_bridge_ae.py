"""Mode-safe canonical rubric-v4 bridge for excluded release; no clinical edits."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;FC=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(FC/'v4_build/v4i_freeze_codex'))
import acute_v4_bridge as OLD
from safe_identity import Catalog,canonical,jl,sha
V=OLD.V;RUBRICS=OLD.RUBRICS;STATES=OLD.STATES
KEY_ACTIONS=(HERE/'KEY_ACTIONS_CANONICAL_v4q.jsonl',json.loads((HERE/'BINDING.json').read_text())['key_actions_sha256'])
item_text=V.item_text;item_key=V.item_key;rubric_text=V.rubric_text
CAT=None;REFS=None
def configure(release=None,key_actions=None):
 global CAT,REFS
 CAT=Catalog(release or FC/'wp_v9_vcm/holdout_v9_sealed/RELEASE_TEST_v4q')
 p=Path(key_actions or KEY_ACTIONS[0])
 if sha(p)!=KEY_ACTIONS[1]:raise ValueError('canonical key-action freeze changed')
 rr=jl(p);REFS={r['row_id']:r for r in rr}
 if len(REFS)!=len(rr):raise ValueError('duplicate canonical reference')
 for cid,r in REFS.items():
  matches=[rs[cid] for rs in CAT.canon.values() if cid in rs]
  if not matches or any(x['evaluator']['source_image_sha256']!=r['source_image_sha256'] for x in matches):raise ValueError('unbound/mismatched reference '+cid)
 V.KEY_ACTIONS=(p,sha(p));V.rubric_text('v4');return CAT
def build_item(row,mode=None):
 if CAT is None:configure()
 cid=canonical(row)
 if mode is not None and canonical(CAT.row(mode,row['sample_id']))!=cid:raise ValueError('mode/canonical mismatch')
 k=REFS[cid]
 if row['evaluator']['source_image_sha256']!=k['source_image_sha256']:raise ValueError('reference image mismatch')
 return {'hazard_types':k['types'],'visible_hazard':k['visible_hazard'],'user_request':row.get('query_text') or '', 'key_actions':[V._act(a) for a in k['actions']],'flags':k['flags']}
