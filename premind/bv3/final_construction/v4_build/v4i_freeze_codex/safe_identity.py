"""Y shared identity contract: resolve local IDs in their OWN mode; never flatten modes."""
import hashlib,json,re
from pathlib import Path
MODES=('retrieved','gt_retrieved','gt_retrieved_drop')
FC=Path(__file__).resolve().parents[2]
DEFAULT_RELEASE=FC/'wp_v9_vcm/holdout_v9_sealed/RELEASE_TEST_v4i'
def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):
 e=r['evaluator'];e=json.loads(e) if isinstance(e,str) else e
 return e['row_id']
def episode(r):return (r['mode'],r['row_id'],r['arm'],r['kind'],r.get('sample_idx',0))
class Catalog:
 def __init__(self,release=DEFAULT_RELEASE):
  self.release=Path(release).resolve();self.local={};self.canon={};self.files={};self.hashes={}
  for mode in MODES:
   p=self.release/('' if mode=='retrieved' else mode)/'holdout.jsonl'
   rs=jl(p);self.files[mode]=p;self.hashes[mode]=sha(p)
   self.local[mode]={r['sample_id']:r for r in rs};self.canon[mode]={canonical(r):r for r in rs}
   if len(self.local[mode])!=len(rs) or len(self.canon[mode])!=len(rs):raise ValueError('duplicate release identity: '+mode)
 def row(self,mode,local_id):return self.local[mode][local_id]
 def key(self,mode,local_id):return (mode,canonical(self.row(mode,local_id)))
 def record(self,r):return self.row(r['mode'],r['row_id'])
 def image(self,r):
  h=r['evaluator'].get('source_image_sha256','')
  actual=sha((self.release.parent/r['query']).resolve())
  if not re.fullmatch('[a-f0-9]{64}',h) or actual!=h:raise ValueError('image identity mismatch')
  return h
 def check_run(self,run):
  meta=json.loads((Path(run)/'eval_report.json').read_text())['meta']
  for mode,d in meta['data'].items():
   if mode not in MODES:continue
   # Snapshot files may be subsets, but may never claim the full release hash.
   if sha(Path(d['path']))!=d['sha256']:raise ValueError('run source changed: '+mode)
   if d['sha256']!=self.hashes[mode]:raise ValueError('run not bound to requested release: '+mode)
  return meta
