import pytest
import torch
from puzzlekv import Config,KVCache,attention,dense_attention
from puzzlekv.decomposition import decompose

@pytest.mark.parametrize("key,value",[(True,False),(False,True),(True,True)])
def test_direct(key,value,device):
    c=KVCache(Config(compute_dtype=torch.float32,page_size=7,compression_enabled=False,device=device))
    k=torch.randn(2,7,13,device=device); v=torch.randn(2,7,11,device=device)
    c.append(k,v); p=c.pages[0]
    if key: p.keys=[decompose(x,3) for x in p.keys]
    if value: p.values=[decompose(x,2) for x in p.values]
    kr=torch.stack([x.reconstruct() if key else x for x in p.keys]); vr=torch.stack([x.reconstruct() if value else x for x in p.values])
    q=torch.randn(4,3,13,device=device)
    torch.testing.assert_close(attention(q,c),dense_attention(q,kr,vr),atol=2e-6,rtol=2e-5)
