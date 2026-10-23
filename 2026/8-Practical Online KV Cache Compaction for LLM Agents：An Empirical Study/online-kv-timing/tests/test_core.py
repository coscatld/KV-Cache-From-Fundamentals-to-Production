import json,time
import pytest
import torch
from online_kv.compression import Bank,Segment,te,BudgetExceeded
from online_kv.metrics import Metrics
from online_kv.policies import Policy,Controller,core_policies
from online_kv.experiments import execute

def seg(turn=0,n=10):
    kv=torch.arange(2*2*2*n*4).float().reshape(2,2,2,n,4)
    positions=torch.arange(n).expand(2,2,n).clone()
    protected=torch.zeros_like(positions,dtype=torch.bool);protected[...,0]=True
    return Segment(turn,kv,positions,protected,True,False,n)

def test_te_budget_shapes_dtype_and_protected():
    s=seg();q=torch.ones(2,2,3,4)
    result=te(s,q,.3)
    assert result.kv.shape==(2,2,2,3,4) and result.kv.dtype==s.kv.dtype
    assert (result.positions==0).any(-1).all()
    assert result.kv.untyped_storage().data_ptr()!=s.kv.untyped_storage().data_ptr()
    assert result.nbytes<s.nbytes

def test_prefix_active_frozen_not_evicted():
    s=seg(-1)
    with pytest.raises(ValueError):te(s,torch.ones(2,2,1,4),.2)
    s=seg();s.closed=False
    with pytest.raises(ValueError):te(s,torch.ones(2,2,1,4),.2)
    s.closed=True;s.frozen=True
    with pytest.raises(ValueError):te(s,torch.ones(2,2,1,4),.2)

@pytest.mark.parametrize("delay",[1,3,5])
def test_delay_and_no_future(delay):
    bank=Bank(1000000,Metrics());ctl=Controller(bank,Policy(name="fixed",delay=delay))
    for t in range(delay+1):
        s=seg(t);bank.segments.append(s)
        q=torch.full((2,2,1,4),float(t+1));ctl.observe(t,q,q)
        decisions=ctl.act()
        if t<delay:assert not decisions
    assert bank.segments[0].frozen
    assert bank.metrics.events[0]["age"]==delay
    assert bank.metrics.events[0]["query_source"]=="future"
    with pytest.raises(ValueError):ctl.observe(0,q,q)

def test_interval_and_safety_override():
    bank=Bank(100000,Metrics());ctl=Controller(bank,Policy(min_interval=2,max_interval=5,drift_threshold=2))
    for t in range(3):
        bank.segments.append(seg(t));q=torch.ones(2,2,1,4);ctl.observe(t,q,q)
        decisions=ctl.act()
        if t<2:assert not decisions
    assert bank.segments[0].frozen
    ctl.bank.cap=ctl.bank.nbytes*2
    ctl.act(reserve=int(ctl.bank.cap*.2))
    assert any(e["forced"] for e in bank.metrics.events)
    assert not any(not s.closed and s.frozen for s in bank.segments)

def test_cap_and_infeasibility():
    bank=Bank(10,Metrics());s=seg()
    with pytest.raises(BudgetExceeded):bank.append(0,s.kv,s.positions)
    assert not bank.segments

def test_separate_T_R_TR():
    ps=core_policies()
    assert ps["T"].name=="adaptive" and not ps["T"].adaptive_ratio
    assert ps["R"].name=="fixed" and ps["R"].adaptive_ratio
    assert ps["TR"].name=="adaptive" and ps["TR"].adaptive_ratio

def test_wallclock():
    m=Metrics();start=time.perf_counter()
    with m.timer("outer"):
        with m.timer("inner"):time.sleep(.003)
    elapsed=time.perf_counter()-start
    assert .002<m.seconds["inner"]<=m.seconds["outer"]<=elapsed

def test_resume_and_partial_record(tmp_path):
    dataset=tmp_path/"data.json";dataset.write_text(json.dumps([dict(id="v1",split="validation",index=0)]))
    cfg=dict(backend="synthetic",dataset=str(dataset),seed=42,peak_kv_bytes=1000000,synthetic_turns=6,policies=["delay1","T"])
    output=tmp_path/"out";execute(cfg,output)
    log=output/"tasks.jsonl";before=log.read_text()
    with log.open("a") as f:f.write('{"task_id":')
    execute(cfg,output)
    assert log.read_text()==before
    assert len(before.splitlines())==2
    with pytest.raises(ValueError):execute(cfg,output,split="test")

def test_ratio_protection_floor():
    s=seg();s.protected[...,:5]=True
    assert te(s,torch.ones(2,2,1,4),.2).tokens==5


def test_query_provenance():
    bank=Bank(1000000,Metrics());ctl=Controller(bank,Policy(name="fixed",delay=3))
    bank.segments.append(seg(0))
    for t in range(3):
        q=torch.full((2,2,1,4),float(t));ctl.observe(t,q,q)
    q,source=ctl.query_for(bank.segments[0])
    assert source=="future" and q.shape[-2]==2
    assert torch.equal(q[0,0,:,0],torch.tensor([1.,2.]))

def test_final_record_without_newline(tmp_path):
    dataset=tmp_path/"data.json";dataset.write_text(json.dumps([dict(id="v1",split="validation",index=0)]))
    cfg=dict(backend="synthetic",dataset=str(dataset),seed=42,peak_kv_bytes=1000000,synthetic_turns=2,policies=["T"])
    out=tmp_path/"out";execute(cfg,out)
    log=out/"tasks.jsonl";log.write_text(log.read_text().rstrip("\n"))
    execute(cfg,out)
    assert len(log.read_text().splitlines())==1 and log.read_text().endswith("\n")

def test_minimum_global_interval_except_safety():
    bank=Bank(1000000,Metrics());ctl=Controller(bank,Policy(min_interval=2,max_interval=3,drift_threshold=2))
    for t in range(10):
        bank.segments.append(seg(t));q=torch.ones(2,2,1,4);ctl.observe(t,q,q);ctl.act()
    events=[e for e in bank.metrics.events if not e["forced"]]
    assert all(b["at_turn"]-a["at_turn"]>=2 for a,b in zip(events,events[1:]))
