import torch
from puzzlekv import Config,KVCache,attention,dense_attention

def test_incremental(device):
    c=KVCache(Config(compute_dtype=torch.float32,page_size=8,key_rank=2,value_rank=2,sink_size=8,recent_window_size=8,device=device))
    k=torch.randn(2,33,13,device=device); v=torch.randn(2,33,11,device=device)
    for i in range(33):
        c.append(k[:,i:i+1],v[:,i:i+1])
        assert c.length==i+1
        assert [p.start for p in c.pages]==list(range(0,i+1,8))
        for p in c.pages:
            if p.compressed:
                assert p.length==8 and p.start>=8 and p.start+8<=c.length-8
    assert c.pages[1].compressed and not c.pages[-1].compressed
    assert c.storage()["tensor_bytes"]<c.storage()["full_tensor_bytes"]

def test_disabled_decode(device):
    c=KVCache(Config(compute_dtype=torch.float32,page_size=7,compression_enabled=False,device=device))
    k=torch.randn(2,17,13,device=device); v=torch.randn(2,17,11,device=device)
    for i in range(17):
        c.append(k[:,i:i+1],v[:,i:i+1]); q=torch.randn(4,1,13,device=device)
        torch.testing.assert_close(attention(q,c),dense_attention(q,k[:,:i+1],v[:,:i+1]),atol=2e-6,rtol=2e-5)
