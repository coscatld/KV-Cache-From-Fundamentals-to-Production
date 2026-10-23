import argparse, json, platform, statistics, time
from pathlib import Path
import torch
import importlib.metadata
from puzzlekv import Config,KVCache,attention,dense_attention
from puzzlekv.decomposition import decompose,Factors
from puzzlekv.integration import tiny_llama_comparison

def measure(fn,repeats):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
    times.sort()
    return dict(median_ms=statistics.median(times),p90_ms=times[min(len(times)-1,int(.9*len(times)))],peak_increment_bytes=torch.cuda.max_memory_allocated()-baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--length",type=int,default=1024); parser.add_argument("--page-size",type=int,default=32)
    parser.add_argument("--rank",type=int,default=8); parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="float16")
    parser.add_argument("--output",default="results/gpu.json"); parser.add_argument("--hf",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("NOT RUN: benchmark requires CUDA; no CPU substitute")
    if args.repeats<1 or args.length<args.page_size: raise ValueError("positive repeats and length >= page_size required")
    torch.manual_seed(42); dtype=getattr(torch,args.dtype)
    cfg=Config(page_size=args.page_size,key_rank=args.rank,value_rank=args.rank,sink_size=args.page_size,recent_window_size=args.page_size,device="cuda",compute_dtype=dtype)
    k=torch.randn(2,args.length,64,device="cuda",dtype=dtype); v=torch.randn_like(k); q=torch.randn(4,1,64,device="cuda",dtype=dtype)
    cache=KVCache(cfg); prefill=measure(lambda:KVCache(cfg).append(k,v),args.repeats); cache.append(k,v)
    def reconstruct(attr):
        return torch.stack([torch.cat([x.reconstruct() if isinstance(x,Factors) else x for x in [getattr(p,attr)[h] for p in cache.pages]]) for h in range(2)])
    def reconstructed(): return dense_attention(q,reconstruct("keys"),reconstruct("values"))
    page=k[0,:args.page_size].contiguous()
    def incremental():
        c=KVCache(cfg); c.append(k,v)
        for _ in range(args.page_size): c.append(k[:,:1],v[:,:1]); attention(q,c)
    # Incremental timing below isolates appending to a fresh full dense page before eligibility.
    def compression():
        c=KVCache(Config(**{**vars(cfg),"compression_enabled":False})); c.append(k,v)
        c.config=cfg; c.compress()
    full=dense_attention(q,k,v).float(); direct=attention(q,cache).float()
    kr=reconstruct("keys").float(); vr=reconstruct("values").float()
    def rel(a,b): return ((a-b).norm()/a.norm().clamp_min(1e-12)).item()
    report={"environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"seed":42,"triton":importlib.metadata.version("triton"),"transformers":importlib.metadata.version("transformers")},"parameters":vars(args),"storage":cache.storage(),"errors":{"key_relative":rel(k.float(),kr),"value_relative":rel(v.float(),vr),"logits_relative":rel(q.float()@k.repeat_interleave(2,0).float().transpose(-1,-2),q.float()@kr.repeat_interleave(2,0).transpose(-1,-2)),"attention_relative":rel(full,direct),"direct_vs_reconstruction_relative":rel(reconstructed().float(),direct)}}
    jobs={"single_page_svd":lambda:decompose(page,args.rank),"batched_svd":lambda:torch.linalg.svd(k[:,:args.length//args.page_size*args.page_size].reshape(-1,args.page_size,64).float(),full_matrices=False),"prefill_append_and_svd":lambda:KVCache(cfg).append(k,v),"full_kv":lambda:dense_attention(q,k,v),"reconstructed":reconstructed,"torch_mixed":lambda:attention(q,cache),"incremental_setup_append_decode":incremental,"compression_including_dense_setup":compression}
    # Isolated all-low-rank attention and append eligibility include independent CUDA measurements.
    from puzzlekv.triton_kernels import lowrank_logits,lowrank_value
    f=decompose(page,args.rank); weights=torch.softmax(torch.randn(1,page.shape[0],device="cuda"),-1)
    jobs["torch_lowrank_page"]=lambda: ((q[0].float()@f.b.float().T)@f.a.float().T, (weights@f.a.float())@f.b.float())
    jobs["triton_lowrank_page"]=lambda:(lowrank_logits(q[0].contiguous(),f.a,f.b),lowrank_value(weights,f.a,f.b))
    jobs["triton_mixed"]=lambda:attention(q,cache,backend="triton")
    report["timings"]={name:measure(fn,args.repeats) for name,fn in jobs.items()}
    report["prefill"]=prefill
    report["triton_speedup_vs_torch_mixed"]=report["timings"]["torch_mixed"]["median_ms"]/report["timings"]["triton_mixed"]["median_ms"]
    # Decode with setup outside timed interval, reporting per-token end-to-end GPU time.
    c=KVCache(cfg); c.append(k,v)
    report["decode_per_token"]=measure(lambda:(c.append(k[:,:1],v[:,:1]),attention(q,c)),args.repeats)
    report["decode_note"]="Growing cache; includes periodic SVD. Other mixed attention timings use fixed context."
    if args.hf: report["hf"]=tiny_llama_comparison(cfg)
    target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
