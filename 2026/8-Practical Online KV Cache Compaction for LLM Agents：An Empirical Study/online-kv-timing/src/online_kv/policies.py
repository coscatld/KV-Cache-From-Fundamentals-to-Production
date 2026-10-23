from dataclasses import dataclass,asdict
import torch
from .compression import BudgetExceeded

@dataclass
class Policy:
    name: str="adaptive"
    delay: int=1
    ratio: float=.2
    min_interval: int=1
    max_interval: int=5
    safety_fraction: float=.65
    drift_threshold: float=.15
    growth_threshold: float=.1
    adaptive_ratio: bool=False
    ratio_min: float=.1
    ratio_max: float=.4
    def __post_init__(self):
        if self.name not in ("boundary","fixed","adaptive"): raise ValueError("unknown policy")
        if not 0<self.ratio_min<=self.ratio<=self.ratio_max<=1: raise ValueError("invalid ratios")
        if not 0<self.safety_fraction<1 or not 1<=self.min_interval<=self.max_interval or self.delay<0:
            raise ValueError("invalid policy thresholds")

class Controller:
    def __init__(self,bank,policy):
        self.bank=bank;self.policy=policy;self.last=-1;self.previous=None;self.drift=1.;self.growth=0.;self.previous_bytes=0
        self.queries={};self.boundaries={};self.current=-1;self.drifts=[]
    def observe(self,turn,queries,boundary):
        if turn<=self.current: raise ValueError("events must be chronological")
        with self.bank.metrics.timer("query_statistics"):
            if queries.ndim!=4 or boundary.shape[:2]!=queries.shape[:2]: raise ValueError("invalid query shape")
            self.current=turn;self.queries[turn]=queries.detach().clone();self.boundaries[turn]=boundary.detach().clone()
            mean=torch.nn.functional.normalize(queries.float().mean(-2).flatten(),dim=0)
            self.drift=1. if self.previous is None else float((1-(mean*self.previous).sum()).clamp(0,2))
            self.previous=mean;self.drifts.append(self.drift)
            self.growth=max(0,self.bank.nbytes-self.previous_bytes)/self.bank.cap
            self.previous_bytes=self.bank.nbytes
    def query_for(self,s):
        p=self.policy
        future=[q for t,q in self.queries.items() if s.turn<t<=self.current]
        # Safety fallback uses only an already-observed boundary; never a future query.
        if p.name=="boundary" or not future: return self.boundaries[s.turn],"boundary"
        return torch.cat(future,-2),"future"
    def ratio(self):
        p=self.policy
        if not p.adaptive_ratio:return p.ratio
        # More drift => retain more, clamp under memory pressure.
        value=p.ratio_min+(p.ratio_max-p.ratio_min)*min(self.drift/2,1)
        return max(p.ratio_min,min(value,p.ratio)) if self.bank.nbytes/self.bank.cap>=p.safety_fraction else value
    def act(self,reserve=0):
        p=self.policy; decisions=[]
        for s in list(self.bank.segments):
            if s.turn<0 or not s.closed or s.frozen: continue
            age=self.current-s.turn
            pressure=(self.bank.nbytes+reserve)/self.bank.cap>=p.safety_fraction
            elapsed=self.current-self.last
            if pressure: reason="safety_pressure"
            elif p.name=="boundary":reason="boundary"
            elif p.name=="fixed" and age>=p.delay:reason="fixed_delay"
            elif p.name=="adaptive" and age>=p.max_interval and elapsed>=p.min_interval: reason="maximum_age"
            elif p.name=="adaptive" and elapsed>=p.max_interval and age>=p.min_interval:reason="maximum_interval"
            elif p.name=="adaptive" and age>=p.min_interval and elapsed>=p.min_interval and (self.drift<=p.drift_threshold or self.growth>=p.growth_threshold):reason="stable_queries" if self.drift<=p.drift_threshold else "rapid_growth"
            else: continue
            q,source=self.query_for(s)
            self.bank.compact(s,q,self.ratio(),reason,pressure)
            self.bank.metrics.events[-1].update(at_turn=self.current,age=age,query_source=source,drift=self.drift,growth=self.growth,added_tokens_during_delay=sum(x.original_tokens for x in self.bank.segments if s.turn<x.turn<=self.current))
            self.last=self.current;decisions.append(s.turn)
        # Queries for frozen turns no longer needed as boundary fallbacks.
        raw=[s.turn for s in self.bank.segments if s.turn>=0 and not s.frozen]
        cutoff=min(raw) if raw else self.current
        self.queries={t:q for t,q in self.queries.items() if t>=cutoff}
        self.boundaries={t:q for t,q in self.boundaries.items() if t in raw}
        self.bank.require(reserve)
        return decisions
    @property
    def query_bytes(self):
        return sum(q.numel()*q.element_size() for q in list(self.queries.values())+list(self.boundaries.values()))

def core_policies():
    base={"A_boundary":Policy(name="boundary")}
    for k in (1,3,5):base[f"delay{k}"]=Policy(name="fixed",delay=k)
    base["T"]=Policy(name="adaptive")
    base["R"]=Policy(name="fixed",delay=1,adaptive_ratio=True)
    base["TR"]=Policy(name="adaptive",adaptive_ratio=True)
    return base
