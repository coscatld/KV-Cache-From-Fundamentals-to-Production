import pytest
import torch
from puzzlekv.triton_kernels import lowrank_logits,lowrank_value
from puzzlekv import Config,KVCache,attention

@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("p,r,d",[(7,3,13),(32,8,64),(41,17,79)])
def test_kernels(dtype,p,r,d,device):
    pytest.importorskip("triton")
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("NOT RUN: BF16 unsupported by GPU")
    q=torch.randn(3,d,device=device,dtype=dtype); a=torch.randn(p,r,device=device,dtype=dtype); b=torch.randn(r,d,device=device,dtype=dtype)
    torch.testing.assert_close(lowrank_logits(q,a,b),(q.float()@b.float().T)@a.float().T,atol=3e-4,rtol=3e-4)
    w=torch.randn(3,p,device=device,dtype=dtype)
    torch.testing.assert_close(lowrank_value(w,a,b),(w.float()@a.float())@b.float(),atol=3e-4,rtol=3e-4)
    c=KVCache(Config(page_size=16,key_rank=3,value_rank=3,sink_size=0,recent_window_size=0,device=device,compute_dtype=dtype))
    c.append(torch.randn(2,35,d,device=device,dtype=dtype),torch.randn(2,35,d,device=device,dtype=dtype))
    query=torch.randn(4,3,d,device=device,dtype=dtype)
    torch.testing.assert_close(attention(query,c,backend="triton"),attention(query,c),atol=3e-3,rtol=3e-3)
