import torch
from puzzlekv import Config,KVCache,attention,dense_attention
from puzzlekv.decomposition import Factors

def test_mixed(device):
    c=KVCache(Config(compute_dtype=torch.float32,page_size=8,key_rank=2,value_rank=2,sink_size=8,recent_window_size=8,device=device))
    c.append(torch.randn(2,35,13,device=device),torch.randn(2,35,11,device=device))
    assert any(p.compressed for p in c.pages) and not c.pages[-1].compressed
    q=torch.randn(4,5,13,device=device)
    def reconstruct(attr):
        return torch.stack([torch.cat([x.reconstruct() if isinstance(x,Factors) else x for x in [getattr(p,attr)[h] for p in c.pages]]) for h in range(2)])
    expected=dense_attention(q,reconstruct("keys"),reconstruct("values"))
    actual=attention(q,c)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    c.pages=list(reversed(c.pages))
    torch.testing.assert_close(attention(q,c),actual,atol=2e-6,rtol=2e-5)
