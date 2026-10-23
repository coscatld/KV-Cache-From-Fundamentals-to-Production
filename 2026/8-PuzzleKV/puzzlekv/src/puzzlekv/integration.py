"""Independent adapter and a tiny Hugging Face Llama integration.
The adapter replaces self_attn.forward on eager Llama modules only.
"""
import types
import torch
from .cache import KVCache
from .attention_reference import attention

class Experiment:
    def __init__(self,config):
        self.cache=KVCache(config)
    def step(self,q,k,v,positions=None,backend="torch"):
        self.cache.append(k,v)
        return attention(q,self.cache,positions,backend)

def attach_llama(model,config,backend="torch"):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    caches=[]; originals=[]
    for layer in model.model.layers:
        module=layer.self_attn
        originals.append((module,module.forward))
        cache=KVCache(config); caches.append(cache)
        def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,cache_position=None,position_ids=None,**kwargs):
            if hidden_states.shape[0]!=1:
                raise ValueError("only batch=1 supported")
            if attention_mask is not None and attention_mask.ndim==2 and not bool(attention_mask.all()):
                raise ValueError("padding unsupported")
            shape=(*hidden_states.shape[:-1],-1,self.head_dim)
            q=self.q_proj(hidden_states).view(shape).transpose(1,2)
            k=self.k_proj(hidden_states).view(shape).transpose(1,2)
            v=self.v_proj(hidden_states).view(shape).transpose(1,2)
            cos,sin=position_embeddings
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            c=self._puzzlekv_cache
            start=c.length
            positions=torch.arange(start,start+hidden_states.shape[1],device=q.device)
            c.append(k[0],v[0])
            out=attention(q[0],c,positions,backend)
            out=out.transpose(0,1).reshape(1,hidden_states.shape[1],-1)
            return self.o_proj(out),None
        module._puzzlekv_cache=cache
        module.forward=types.MethodType(forward,module)
    def restore():
        for module,original in originals:
            module.forward=original
            del module._puzzlekv_cache
    return caches,restore

def tiny_llama_comparison(config,backend="torch"):
    """No downloads; actual HF random-weight model, logits and greedy decode."""
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(42)
    cfg=LlamaConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,attention_dropout=0.0)
    cfg._attn_implementation="eager"
    model=LlamaForCausalLM(cfg).to(device=config.device,dtype=config.compute_dtype).eval()
    tokens=torch.randint(0,97,(1,48),device=config.device)
    full=[]; compressed=[]
    with torch.inference_mode():
        for length in range(40,49):
            full.append(model(tokens[:,:length],use_cache=False).logits[:,-1].float())
        caches,restore=attach_llama(model,config,backend)
        try:
            compressed.append(model(tokens[:,:40],use_cache=False,position_ids=torch.arange(40,device=config.device)[None]).logits[:,-1].float())
            for index in range(40,48):
                compressed.append(model(tokens[:,index:index+1],use_cache=False,position_ids=torch.tensor([[index]],device=config.device)).logits[:,-1].float())
        finally:
            restore()
    a=torch.stack(full); b=torch.stack(compressed)
    return {"logits_relative_error":((a-b).norm()/a.norm()).item(),"greedy_agreement":(a.argmax(-1)==b.argmax(-1)).float().mean().item(),"layers_storage":[c.storage() for c in caches]}
