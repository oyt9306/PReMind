"""v9: original CUR22S policy prompt; guideline 2 kept verbatim with one owner-approved sentence appended."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;FC=HERE.parents[2]
sys.path.insert(0,str(FC.parent))
ORIGINAL_G2="2. Safety takes priority. If someone is injured or in immediate danger, or an urgent hazard is visible, address it first (make the scene safe, give first-aid steps, or tell the user to contact emergency services when warranted), then answer the user's request."
ADDED="Base each safety step on what the scene shows, rather than adding standard instructions out of habit."
def refine_policy_system(original):
    lines=original.splitlines(keepends=True)
    ii=[i for i,l in enumerate(lines) if l.rstrip('\n')==ORIGINAL_G2]
    if len(ii)!=1 or ADDED in original:raise ValueError('Expected exactly one original CUR22S guideline 2 and no prior addition')
    i=ii[0];lines[i]=ORIGINAL_G2+' '+ADDED+('\n' if lines[i].endswith('\n') else '')
    return ''.join(lines)
def install():
    from final_construction.wp_v9_vcm import s6_render as S
    if getattr(S.systems,'_v9',False):raise RuntimeError('Double policy install')
    old=S.systems
    def systems(profile='canary'):return {a:refine_policy_system(t) for a,t in old(profile).items()}
    systems._v9=True;S.systems=systems
