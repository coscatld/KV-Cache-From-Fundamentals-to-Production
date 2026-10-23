import math
import torch
from .decomposition import Factors

def dense_attention(q,k,v,query_positions=None,key_positions=None):
    h,t,d = q.shape
    if h % k.shape[0]:
        raise ValueError("query heads must be divisible by KV heads")
    k=k.repeat_interleave(h//k.shape[0],0).float()
    v=v.repeat_interleave(h//v.shape[0],0).float()
    kp=torch.arange(k.shape[1],device=q.device) if key_positions is None else key_positions
    qp=torch.arange(k.shape[1]-t,k.shape[1],device=q.device) if query_positions is None else query_positions
    logits=q.float() @ k.transpose(-1,-2)/math.sqrt(d)
    logits=logits.masked_fill(kp[None,:]>qp[:,None],-torch.inf)
    weights=torch.nan_to_num(logits.softmax(-1))
    return (weights@v).to(q.dtype)

def attention(q, cache, query_positions=None, backend="torch"):
    """Global stable softmax; no historical K/V reconstruction. q=[Hq,Q,D]."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be torch or triton")
    if not cache.pages:
        raise ValueError("empty cache")
    h,t,d=q.shape
    hk=len(cache.pages[0].keys)
    if h % hk:
        raise ValueError("invalid GQA head mapping")
    qp=torch.arange(cache.length-t,cache.length,device=q.device) if query_positions is None else query_positions
    outputs=[]
    for head in range(h):
        kvhead=head//(h//hk)
        chunks=[]
        for p in cache.pages:
            k=p.keys[kvhead]
            if isinstance(k,Factors):
                if backend == "triton":
                    from .triton_kernels import lowrank_logits
                    scores=lowrank_logits(q[head],k.a,k.b)
                else:
                    scores=(q[head].float()@k.b.float().T)@k.a.float().T
            else:
                scores=q[head].float()@k.float().T
            positions=torch.arange(p.start,p.start+p.length,device=q.device)
            chunks.append((scores/math.sqrt(d)).masked_fill(positions[None,:]>qp[:,None],-torch.inf))
        weights=torch.nan_to_num(torch.cat(chunks,-1).softmax(-1))
        out=torch.zeros((t,cache.layout[2]),device=q.device,dtype=torch.float32)
        offset=0
        for p in cache.pages:
            w=weights[:,offset:offset+p.length].contiguous(); v=p.values[kvhead]
            if isinstance(v,Factors):
                if backend == "triton":
                    from .triton_kernels import lowrank_value
                    out+=lowrank_value(w,v.a,v.b)
                else:
                    out+=(w@v.a.float())@v.b.float()
            else:
                out+=w@v.float()
            offset+=p.length
        outputs.append(out)
    return torch.stack(outputs).to(q.dtype)
