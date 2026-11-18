"""Process-shared total owner cap for hazard, warning AND MIE (train + dev) judge calls, including retries
(Phase II' copy: only CAP_USD and the message changed)."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from eval_v1 import api_providers as AP


CAP_USD=1.8  # Phase II-triple-prime owner total = max(3.0, 1.5 x CPU estimate 0.85 for 100 steps); hazard 0.80 + warning 0.25 via ARGS, MIE 1.70 + dev MIE 0.25


class GlobalBudgetExceeded(RuntimeError):pass


class Bank:
    def __init__(self,path,cap=CAP_USD,prior=0.0):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cap=cap;self.prior=prior

    def update(self,fn):
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            rows=[json.loads(l) for l in f if l.strip()]
            event=fn(rows)
            f.seek(0,2);f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
            return event

    def reserve(self,amount):
        def fn(rows):
            active={}
            for r in rows:active[r['id']]=r
            spent=self.prior+sum(r['usd'] for r in active.values())
            if spent+amount>self.cap:raise GlobalBudgetExceeded(f'Phase II-prime owner judge cap ${self.cap} exhausted')
            return {'id':f'{os.getpid()}-{time.time_ns()}','status':'reserved','usd':amount,'prior_usd':self.prior,'cap_usd':self.cap}
        return self.update(fn)['id']

    def settle(self,key,cost):
        return self.update(lambda rows:{'id':key,'status':'settled','usd':cost})


def install(out):
    if getattr(AP.openai_call,'_scene_budget',False):raise RuntimeError('Global budget installed twice')
    original=AP.openai_call
    bank=Bank(Path(str(out)+'.judge')/'OWNER_API_BUDGET.jsonl')
    def call(model,req,**kw):
        if req.images():raise RuntimeError('Training judge API must be text-only')
        byte_bound=len(req.system.encode())+sum(len(v.encode()) for _,parts in req.turns for k,v in parts if k=='text')+2048
        reserve=AP.cost_usd(model,byte_bound,req.max_tokens+AP.reasoning_allowance(model))
        key=bank.reserve(reserve)
        # Unknown failures retain the full reservation; no cost can disappear on resume.
        resp=original(model,req,**kw)
        ex=resp.extra or {}
        cost=AP.cost_usd(model,int(ex.get('billed_input_tokens',resp.input_tokens)),int(ex.get('billed_output_tokens',resp.output_tokens)))
        if cost>reserve:raise GlobalBudgetExceeded('Provider usage exceeded conservative reservation')
        bank.settle(key,cost)
        return resp
    call._scene_budget=True
    AP.openai_call=call
