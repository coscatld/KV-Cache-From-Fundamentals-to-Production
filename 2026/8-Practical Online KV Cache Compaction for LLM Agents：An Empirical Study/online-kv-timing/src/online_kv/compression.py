from dataclasses import dataclass
import math
import torch

class BudgetExceeded(RuntimeError): pass

@dataclass
class Segment:
    turn: int
    kv: torch.Tensor  # [layers,2,KV_heads,tokens,head_dim], post-RoPE keys
    positions: torch.Tensor  # [layers,heads,tokens], absolute RoPE positions
    protected: torch.Tensor  # same shape
    closed: bool=False
    frozen: bool=False
    original_tokens: int=0
    @property
    def tokens(self): return self.kv.shape[-2]
    @property
    def nbytes(self):
        return sum(x.numel()*x.element_size() for x in (self.kv,self.positions,self.protected))

def te(segment,queries,ratio):
    """Per-layer, per-head RMS attention. Every policy calls this same function."""
    if not segment.closed or segment.frozen or segment.turn<0:
        raise ValueError("only closed raw non-prefix turns are eligible")
    if not 0<ratio<=1: raise ValueError("ratio must be in (0,1]")
    l,_,h,n,d=segment.kv.shape
    if queries.shape[:2]!=(l,h) or queries.shape[-1]!=d or queries.shape[-2]<1:
        raise ValueError("queries must be [layers,KV_heads,Q,head_dim]")
    keep=max(math.ceil(n*ratio),int(segment.protected.sum(-1).max()))
    logits=queries.float()@segment.kv[:,0].float().transpose(-1,-2)/math.sqrt(d)
    scores=logits.softmax(-1).square().mean(-2).sqrt()

    scores=scores.masked_fill(segment.protected,float("inf"))
    indices=scores.topk(keep,dim=-1).indices.sort(-1).values

    selected=segment.kv.gather(-2,indices[:,None,:,:,None].expand(l,2,h,keep,d)).contiguous()
    pos=segment.positions.gather(-1,indices).contiguous()
    prot=segment.protected.gather(-1,indices).contiguous()
    
    return Segment(segment.turn,selected,pos,prot,True,True,segment.original_tokens)

class Bank:
    def __init__(self,cap_bytes,metrics):
        self.segments=[]; self.cap=cap_bytes; self.metrics=metrics
    @property
    def nbytes(self): return sum(s.nbytes for s in self.segments)
    @property
    def tokens(self): return sum(s.tokens for s in self.segments)
    def require(self,extra=0):
        if self.nbytes+extra>self.cap: raise BudgetExceeded("persistent KV + metadata cap exceeded")
        self.metrics.observe(self.nbytes+extra)
    def append(self,turn,kv,positions,protected=None):
        if protected is None: protected=torch.zeros_like(positions,dtype=torch.bool)
        extra=kv.numel()*kv.element_size()+positions.numel()*positions.element_size()+protected.numel()
        # Account old and new buffers coexisting during torch.cat.
        previous=next((s for s in self.segments if s.turn==turn),None)
        if previous:
            if previous.closed: raise ValueError("cannot append to closed turn")
            self.require(extra+previous.nbytes+extra)
            previous.kv=torch.cat((previous.kv,kv),-2)
            previous.positions=torch.cat((previous.positions,positions),-1)
            previous.protected=torch.cat((previous.protected,protected),-1)
            previous.original_tokens+=kv.shape[-2]
        else:
            self.require(extra)
            self.segments.append(Segment(turn,kv.clone(),positions.clone(),protected.clone(),original_tokens=kv.shape[-2]))
        self.require()
    def close(self,turn):
        for s in self.segments:
            if s.turn==turn: s.closed=True; return
        raise ValueError("missing turn")
    def layer(self,index):
        if not self.segments: return None
        return tuple(torch.cat([s.kv[index,j] for s in self.segments],-2) for j in (0,1))
    def compact(self,s,q,ratio,reason,forced=False):
        before=self.nbytes
        # Preflight output payload, conservative FP32 logits/softmax workspace.
        keep=max(math.ceil(s.tokens*ratio),int(s.protected.sum(-1).max()))
        output=int(s.nbytes*keep/s.tokens)
        workspace=q.shape[0]*q.shape[1]*q.shape[-2]*s.tokens*16
        self.require(output)  # KV-only cap includes coexistence of old + compact KV.
        with self.metrics.timer("compression"):
            replacement=te(s,q,ratio)
        self.metrics.observe(before+replacement.nbytes)
        index=next(i for i,item in enumerate(self.segments) if item is s)
        self.segments[index]=replacement
        self.metrics.events.append(dict(turn=s.turn,reason=reason,forced=forced,ratio=ratio,before_tokens=s.tokens,after_tokens=replacement.tokens,before_bytes=before,after_bytes=self.nbytes,workspace_estimate_bytes=workspace))
        return replacement
