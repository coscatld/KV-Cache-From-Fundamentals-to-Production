import torch
from puzzlekv import Config
from puzzlekv.integration import tiny_llama_comparison

def test_hf_disabled(device):
    result=tiny_llama_comparison(Config(compute_dtype=torch.float32,page_size=8,key_rank=2,value_rank=2,compression_enabled=False,device=device))
    assert result["logits_relative_error"]<2e-5

def test_hf_compressed(device):
    result=tiny_llama_comparison(Config(compute_dtype=torch.float32,page_size=8,key_rank=2,value_rank=2,sink_size=8,recent_window_size=8,device=device))
    assert torch.isfinite(torch.tensor(result["logits_relative_error"]))
    print(result)
    assert result["layers_storage"][0]["tensor_bytes"]<result["layers_storage"][0]["full_tensor_bytes"]
