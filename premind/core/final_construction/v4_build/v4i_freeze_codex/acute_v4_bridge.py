"""Strict rubric-v4 adapter. No legacy extractor, no cross-mode local-ID lookup."""
import importlib.util,sys
from pathlib import Path
from safe_identity import Catalog,FC,canonical,jl,sha
P=FC/'v4_build/acute_adequacy_v4/acute_adequacy_v4.py'
sys.path.insert(0,str(FC))
spec=importlib.util.spec_from_file_location('frozen_acute_v4_y',P);V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
RUBRICS=V.RUBRICS;STATES=V.STATES;KEY_ACTIONS=V.KEY_ACTIONS
item_text=V.item_text;item_key=V.item_key;rubric_text=V.rubric_text
CAT=None;REFS=None
def configure(release=None,key_actions=None):
 global CAT,REFS
 CAT=Catalog(release) if release else Catalog()
 p=Path(key_actions) if key_actions else Path(KEY_ACTIONS[0])
 if key_actions is None and sha(p)!=KEY_ACTIONS[1]:raise ValueError('key-action freeze changed')
 REFS={}
 for k in jl(p):
  rid=k['row_id']
  source=CAT.local['retrieved'].get(rid)
  if source is None:raise ValueError('unbound key-action ID: '+rid)
  cid=canonical(source)
  if cid in REFS:raise ValueError('duplicate key-action canonical identity')
  REFS[cid]=k
 V.rubric_text()
 return CAT

def build_item(row,mode=None):
 if CAT is None:configure()
 cid=canonical(row)
 if mode is not None:
  if canonical(CAT.row(mode,row['sample_id']))!=cid:raise ValueError('mode/canonical mismatch')
 if cid not in REFS:raise ValueError('no frozen canonical v4 reference: '+cid)
 reference=CAT.canon['retrieved'][cid]
 if row['evaluator']['source_image_sha256']!=reference['evaluator']['source_image_sha256']:raise ValueError('reference image mismatch')
 k=REFS[cid]
 if not k['types'] or not any(a.get('specific') for a in k['actions']):raise ValueError('empty registered types/specific actions')
 return {'hazard_types':k['types'],'visible_hazard':k['visible_hazard'],'user_request':row.get('query_text') or '',
         'key_actions':[V._act(a) for a in k['actions']],'flags':k['flags']}

def refuse_legacy(row):
 if row.get('acute_v5') or row.get('collapsed_v1'):raise ValueError('legacy acute reference path retired; use acute_v4_bridge')
