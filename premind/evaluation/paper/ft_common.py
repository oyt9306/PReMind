import os as _o;_o.environ.setdefault('PREMIND_4B_LORA',_o.path.abspath('PReMind-4B'));import sys as _s;from pathlib import Path as _P;_s.path.insert(0,str(_P(__file__).resolve().parents[2]));import pathmap;pathmap.install()   # release: original absolute data paths -> local bundle
import hashlib,json,os,sys
from pathlib import Path
H=Path(os.environ.get('PREMIND_EVAL_OUT','runs/paper_eval')).resolve()   # release: outputs (lanes/judges) live here, not next to the code
CODE=Path(__file__).resolve().parent
FC=(Path(__file__).resolve().parents[2] / 'core' / 'final_construction');V4=FC/'v4_build'
PY=sys.executable
MAIN=V4/'release_test_v4q_fix2b/RELEASE_TEST_v4q_fix2b';CR=V4/'conflict_fix2_final_v3/RELEASE_TEST_v4q_conflict_v10';EMB=V4/'online/query_emb_test_v4i'
FIX=CODE
PKG=V4/'freeze_shortcut_fix/train_scene_action_v9_4b';C10=V4/'conflict_fix2_final_v3/eval_v10'
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model(key):
 """Release: base4b/base9b = zero-shot Qwen3.5; premind4b = the PReMind-4B adapter (PREMIND_4B_LORA, a local LoRA dir)."""
 if key in ('base4b','base9b'):return key[-2:],None
 if key in ('premind4b','cprime4b'):return '4b',Path(os.environ.get('PREMIND_4B_LORA','PReMind-4B'))
 raise KeyError(key)
def run(key,cohort,lane):return H/'lanes'/cohort/f'{key}__{lane}'
def verify_sources():
 """The paper run pinned every source/input in SOURCE_PINS.json; checked only when that file is supplied."""
 p=H/'SOURCE_PINS.json'
 if not p.exists():return
 for name,digest in json.loads(p.read_text()).items():
  if sha(name)!=digest:raise RuntimeError('Source/input drift: '+name)
def assert_owner():
 """Paper run: an internal approval file gated test inference/judging. Release: no-op."""
 return
