import pytest
import torch
from puzzlekv.decomposition import decompose,rank_for_budget

@pytest.mark.parametrize("p,d,r",[(7,13,3),(32,16,8),(5,9,20)])
def test_svd(p,d,r,device):
    x=torch.randn(p,d,device=device); f=decompose(x,r)
    assert f.a.shape==(p,min(p,d,r))
    error=(x-f.reconstruct()).norm()
    s=torch.linalg.svdvals(x)
    torch.testing.assert_close(error,s[min(p,d,r):].norm(),atol=2e-5,rtol=2e-5)
    assert error<=x.norm()
    assert rank_for_budget(p,d,4*(p+d)*2)==min(p,d,2)
