import pytest
import torch
from online_kv.compression import Bank
from online_kv.metrics import Metrics
from online_kv.policies import Controller,Policy

def test_actual_llama_incremental_and_eviction():
    if not torch.cuda.is_available():pytest.skip("NOT RUN: CUDA required for real model test")
    from transformers import LlamaConfig,LlamaForCausalLM
    from online_kv.model import LlamaBackend
    torch.manual_seed(42)
    config=LlamaConfig(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    config._attn_implementation="eager"
    model=LlamaForCausalLM(config).cuda().eval()
    tokens=torch.randint(0,97,(1,12),device="cuda")
    with torch.inference_mode():expected=model(tokens,use_cache=False).logits[:,-1]
    bank=Bank(1000000,Metrics("cuda"));ctl=Controller(bank,Policy(name="fixed",delay=5));adapter=LlamaBackend(model,bank,ctl)
    try:
        adapter.feed(tokens[:,:8],-1,protected=True);bank.close(-1)
        actual=adapter.feed(tokens[:,8:],0,collect=True)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        adapter.complete(0)
        before=bank.nbytes
        bank.compact(bank.segments[1],ctl.boundaries[0],.5,"test")
        assert bank.nbytes<before
        assert bank.segments[0].tokens==8
        assert torch.isfinite(adapter.feed(tokens[:,:1],1,collect=True)).all()
        assert adapter.position==13
    finally:adapter.restore()
