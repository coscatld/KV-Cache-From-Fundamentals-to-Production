import torch
from puzzlekv.attention_reference import dense_attention

def test_sdpa(device):
    q=torch.randn(4,5,13,device=device); k=torch.randn(2,9,13,device=device); v=torch.randn(2,9,7,device=device)
    mask=torch.arange(9,device=device)[None,:]<=torch.arange(4,9,device=device)[:,None]
    expected=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,0),v.repeat_interleave(2,0),attn_mask=mask)
    torch.testing.assert_close(dense_attention(q,k,v),expected)
