"""Process/thread-shared $2 ceiling for this experiment's Luna calls, including retries.

Uses the existing provider profile's token prices and billed usage, with conservative
UTF-8-byte input and full reasoning/output allowance reservations. Failed calls retain
their reservation. No network call exists in this module.
"""
import fcntl,json,os,time
from pathlib import Path

class BudgetExceeded(RuntimeError):pass

class Bank:
    def __init__(self,path,cap=16.0):self.path=Path(path);self.cap=cap;self.path.parent.mkdir(parents=True,exist_ok=True)
    def update(self,fn):
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            latest={}
            for line in f:
                r=json.loads(line);latest[r['id']]=r
            event=fn(latest)
            f.seek(0,2);f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
            return event
    def reserve(self,amount,meta):
        # paper run: an external overall-spend monitor file was also checked here (removed in the release)
        if amount<0:raise ValueError('negative reservation')
        def fn(rows):
            if sum(x['usd'] for x in rows.values())+amount>self.cap:raise BudgetExceeded('Forced-call control Luna cap $16 exhausted; no further paid call')
            return {'id':f'{os.getpid()}-{time.time_ns()}','status':'reserved','usd':amount,'cap':self.cap,**meta}
        return self.update(fn)['id']
    def settle(self,key,cost):
        def fn(rows):
            old=rows[key]
            if cost<0 or cost>old['usd']:raise BudgetExceeded('Billed usage exceeded conservative reservation')
            return {**old,'status':'settled','usd':cost}
        return self.update(fn)

def install(AP,ledger,tag):
    if getattr(AP.openai_call,'_fc_cap',False):return
    original=AP.openai_call;bank=Bank(ledger)
    def call(model,req,**kw):
        if AP.api_model_id(model)!='gpt-6-luna':raise RuntimeError('Only Luna is authorized')
        if req.images():raise RuntimeError('This budget authorizes the baseline-compatible text judge only')
        byte_bound=len(req.system.encode())+sum(len(v.encode()) for _,parts in req.turns for k,v in parts if k=='text')+2048
        reserve=AP.cost_usd(model,byte_bound,req.max_tokens+AP.reasoning_allowance(model))
        prof=AP.profile(model)
        key=bank.reserve(reserve,{'tag':tag,'model':model,'usd_in_per_mtok':prof.usd_in_per_mtok,'usd_out_per_mtok':prof.usd_out_per_mtok})
        # SDK retries are off in AP.openai_call; caller retries re-enter this wrapper.
        response=original(model,req,**kw)
        ex=response.extra or {}
        billed=AP.cost_usd(model,int(ex.get('billed_input_tokens',response.input_tokens)),
                           int(ex.get('billed_output_tokens',response.output_tokens)))
        bank.settle(key,billed)
        return response
    call._fc_cap=True;AP.openai_call=call
    def disallowed(*args,**kw):raise RuntimeError('This experiment authorizes Luna only, no other paid provider')
    for name in ('anthropic_call','gemini_call'):
        if hasattr(AP,name):setattr(AP,name,disallowed)

