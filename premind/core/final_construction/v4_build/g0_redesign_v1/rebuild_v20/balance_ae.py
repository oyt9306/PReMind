"""Owner-approved exposed traps in existing safe slots before step150."""
from pathlib import Path
import json
from trainer_v2_v4 import curriculum_v2_2 as C22,curriculum_v1 as C
HERE=Path(__file__).resolve().parent;FC=HERE.parents[2]
Original=C22.Plan22
def trap_ids():
    p=FC/'wp_v9_vcm/prod_v9/RELEASE_TRAIN_DEV_v20/gt_retrieved_drop/train.jsonl'
    return {r['sample_id'] for r in map(json.loads,p.read_text().splitlines()) if r.get('g0_trap_exposed')}
class Balanced22(Original):
    def build(self,cycle):
        schedule=super().build(cycle);required=trap_ids();early=150*self.p_global
        eligible=set(self.strata['safe_present'])|set(self.strata['safe_absent'])
        if not required<=eligible:raise ValueError('trap outside safe strata')
        for side in ['safe_present','safe_absent']:
            pool=set(self.strata[side]);wanted=sorted(required&pool)
            positions=[i for i,r in enumerate(schedule[:early]) if r in pool]
            if len(positions)<len(wanted):raise ValueError('safe prefix quota cannot cover exposed traps')
            for rid in wanted:
                if rid in schedule[:early]:continue
                dst=next(i for i in positions if schedule[i] not in required)
                if rid in schedule:
                    src=schedule.index(rid);schedule[src],schedule[dst]=schedule[dst],schedule[src]
                else:schedule[dst]=rid
            safe=[r for r in schedule if r in pool]
            if len(safe)!=len(set(safe)):raise ValueError('safe duplicate introduced')
        if not required<=set(schedule[:early]):raise ValueError('trap guarantee failed')
        return schedule
def install():C22.Plan22=Balanced22
