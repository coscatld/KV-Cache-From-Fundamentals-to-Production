"""Minimal eager Llama adapter. True tensor eviction, post-RoPE keys.
No Transformers Cache compatibility claims; use_cache=False + explicit positions.
"""
import types,time
import torch

class LlamaBackend:
    def __init__(self,model,bank,controller):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        if model.config.model_type!="llama": raise ValueError("only standard Llama supported")
        self.model=model.eval();self.bank=bank;self.controller=controller;self.position=0;self.turn=-1
        self.pending={};self.collected=[];self.max_queries=32;self.originals=[]
        adapter=self
        for index,layer in enumerate(model.model.layers):
            module=layer.self_attn;self.originals.append((module,module.forward))
            def forward(this,hidden_states,position_embeddings,attention_mask=None,**kwargs):
                i=this._online_index
                if hidden_states.shape[0]!=1:raise ValueError("batch=1 only")
                shape=(1,hidden_states.shape[1],-1,this.head_dim)
                q=this.q_proj(hidden_states).view(shape).transpose(1,2)
                k=this.k_proj(hidden_states).view(shape).transpose(1,2)
                v=this.v_proj(hidden_states).view(shape).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
                adapter.pending[i]=(k[0],v[0],q[0])
                old=adapter.bank.layer(i)
                keys=k[0] if old is None else torch.cat((old[0],k[0]),-2)
                values=v[0] if old is None else torch.cat((old[1],v[0]),-2)
                groups=q.shape[1]//keys.shape[0]
                keys=keys.repeat_interleave(groups,0);values=values.repeat_interleave(groups,0)
                old_n=keys.shape[-2]-k.shape[-2];n=k.shape[-2]
                mask=torch.arange(old_n+n,device=q.device)[None,:]<=old_n+torch.arange(n,device=q.device)[:,None]
                out=torch.nn.functional.scaled_dot_product_attention(q[0],keys,values,attn_mask=mask,dropout_p=0.)
                return this.o_proj(out.transpose(0,1).reshape(1,n,-1)),None
            module._online_index=index;module.forward=types.MethodType(forward,module)
    @torch.inference_mode()
    def feed(self,ids,turn,collect=False,protected=False):
        if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError("batch=1")
        n=ids.shape[1];cfg=self.model.config
        # Conservative append coexistence + protected position metadata.
        per=len(self.model.model.layers)*cfg.num_key_value_heads*(2*self.model.model.layers[0].self_attn.head_dim*next(self.model.parameters()).element_size()+9)
        active=next((s.nbytes for s in self.bank.segments if s.turn==turn),0)
        self.controller.act(reserve=active+2*n*per) if self.controller.current>=0 else self.bank.require(active+2*n*per)
        self.pending={}
        device=ids.device
        with self.bank.metrics.timer("decode" if collect else "prefill"):
            out=self.model(ids,use_cache=False,position_ids=torch.arange(self.position,self.position+n,device=device)[None])
            kv=torch.stack([torch.stack(self.pending[i][:2]) for i in range(len(self.model.model.layers))])
            l,_,h,_,_=kv.shape
            positions=torch.arange(self.position,self.position+n,device=device).expand(l,h,n).clone()
            self.bank.append(turn,kv,positions,torch.full_like(positions,protected,dtype=torch.bool))
        if collect:
            with self.bank.metrics.timer("query_collection"):
                qs=[]
                for i in range(l):
                    q=self.pending[i][2]
                    qs.append(q.reshape(h,q.shape[0]//h,n,q.shape[-1]).mean(1))
                self.collected.append(torch.stack(qs).detach().clone())
                self.collected=self.collected[-self.max_queries:]
        self.position+=n
        return out.logits[:,-1]
    def complete(self,turn):
        if not self.collected:raise ValueError("turn produced no query")
        q=torch.cat(self.collected,-2);self.collected=[]
        qs=[]
        for i in range(len(self.model.model.layers)):
            raw=self.pending[i][2];h=self.pending[i][0].shape[0]
            qs.append(raw.reshape(h,raw.shape[0]//h,raw.shape[1],raw.shape[2]).mean(1)[:,-1:])
        boundary=torch.stack(qs)
        self.bank.close(turn);self.controller.observe(turn,q,boundary);self.controller.act()
    def restore(self):
        for module,forward in self.originals:module.forward=forward;del module._online_index

def load_model(path,revision,dtype):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError("NOT RUN: real model requires CUDA")
    if not revision:raise ValueError("model revision must be pinned")
    tokenizer=AutoTokenizer.from_pretrained(path,revision=revision,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,revision=revision,local_files_only=True,torch_dtype=getattr(torch,dtype),attn_implementation="eager").to("cuda")
    return model,tokenizer
