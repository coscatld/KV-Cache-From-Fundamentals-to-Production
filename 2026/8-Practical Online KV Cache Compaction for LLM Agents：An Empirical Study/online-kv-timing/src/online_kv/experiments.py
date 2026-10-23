import argparse,csv,hashlib,itertools,json,os,platform,random,subprocess,time
from dataclasses import asdict
from pathlib import Path
import torch
from .compression import Bank,BudgetExceeded
from .metrics import Metrics
from .policies import Policy,Controller,core_policies

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def runtime():
    import importlib.metadata
    try: version=importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:version=None
    return dict(python=platform.python_version(),torch=torch.__version__,transformers=version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)

def search(corpus,query):
    words=set(query.lower().split())
    scored=sorted(enumerate(corpus),key=lambda item:(-len(words&set(item[1]["text"].lower().split())),item[0]))
    return "\n".join(row["text"] for _,row in scored[:2])

def synthetic(task,bank,ctl,cfg):
    # Diagnostic only: random KV tensors, no language model or task success claim.
    g=torch.Generator().manual_seed(cfg["seed"]+task["index"])
    def block(turn,n):
        kv=torch.randn(2,2,2,n,8,generator=g)
        pos=torch.arange(max(0,turn)*n,max(0,turn)*n+n).expand(2,2,n).clone()
        return kv,pos
    kv,pos=block(-1,4);bank.append(-1,kv,pos,torch.ones_like(pos,dtype=torch.bool));bank.close(-1)
    for turn in range(cfg["synthetic_turns"]):
        kv,pos=block(turn,12)
        ctl.act(reserve=kv.numel()*kv.element_size()+pos.numel()*9) if turn else None
        with bank.metrics.timer("prefill"):bank.append(turn,kv,pos)
        bank.close(turn)
        with bank.metrics.timer("query_collection"):q=torch.randn(2,2,4,8,generator=g)
        ctl.observe(turn,q,q[:,:,-1:]);ctl.act()
    return dict(task_score=None,success=None,failure_type=None,synthetic_completed=True,tool_calls=0,duplicate_tool_rate=None)

def real_agent(task,bank,ctl,cfg,model,tokenizer):
    from .model import LlamaBackend
    backend=LlamaBackend(model,bank,ctl);calls=[];answer="";failure=None
    device=next(model.parameters()).device
    def ids(text):return tokenizer(text,return_tensors="pt",add_special_tokens=False).input_ids.to(device)
    def finish(turn):
        backend.feed(ids("\nEndTurn\n"),turn,protected=True)
        backend.complete(turn)
    prefix=cfg["system_prompt"]+"\nQuestion: "+task["question"]+"\n"
    try:
        backend.feed(ids(prefix),-1,protected=True);bank.close(-1)
        for turn in range(cfg["max_turns"]):
            logits=backend.feed(ids("\nAssistant: "),turn,collect=True,protected=True)
            generated=[]
            for step in range(cfg["max_new_tokens"]):
                token=logits.argmax(-1).reshape(1,1)
                logits=backend.feed(token,turn,collect=True)
                generated.append(int(token.item()))
                text=tokenizer.decode(generated,skip_special_tokens=True)
                if token.item()==tokenizer.eos_token_id or "\n" in text:break
            action=tokenizer.decode(generated,skip_special_tokens=True).strip()
            try: parsed=json.loads(action)
            except json.JSONDecodeError:failure="invalid_action_json";finish(turn);break
            if "answer" in parsed:
                answer=str(parsed["answer"]);finish(turn);break
            if parsed.get("tool")!="search" or not isinstance(parsed.get("query"),str):failure="invalid_tool_action";finish(turn);break
            query=parsed["query"];calls.append(query)
            with bank.metrics.timer("tool_total"):
                with bank.metrics.timer("retrieval"): result=search(task["corpus"],query)
            backend.feed(ids("\nTool: "+result+"\n"),turn,protected=False)
            # Preserve structural final delimiter without protecting the entire response.
            finish(turn)
        if not answer and failure is None:failure="turn_limit"
    finally:backend.restore()
    norm=lambda x:" ".join(x.lower().split())
    correct=norm(answer) in [norm(a) for a in task["answers"]]
    duplicate=0 if not calls else (len(calls)-len(set(calls)))/len(calls)
    return dict(answer=answer,task_score=float(correct),success=correct,failure_type=failure,tool_calls=len(calls),duplicate_tool_rate=duplicate,tool_queries=calls,absolute_positions=backend.position)

def run_one(task,name,policy,cfg,model=None,tokenizer=None):
    seed=cfg["seed"]+task.get("index",0);random.seed(seed);torch.manual_seed(seed)
    device="cpu" if cfg["backend"]=="synthetic" else "cuda"
    metrics=Metrics(device);bank=Bank(cfg["peak_kv_bytes"],metrics);ctl=Controller(bank,policy)
    if device=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();out={};status="ok"
    try:out=synthetic(task,bank,ctl,cfg) if device=="cpu" else real_agent(task,bank,ctl,cfg,model,tokenizer)
    except BudgetExceeded as e:status="infeasible";out=dict(task_score=None,success=False,failure_type="kv_budget",error=str(e))
    except Exception as e:status="error";out=dict(task_score=None,success=False,failure_type=type(e).__name__,error=str(e))
    metrics.sync();elapsed=time.perf_counter()-started
    return dict(task_id=task["id"],split=task["split"],policy=name,parameters=asdict(policy),seed=seed,backend=cfg["backend"],status=status,**out,wall_seconds=elapsed,times=metrics.seconds,peak_kv_bytes=metrics.peak_kv_bytes,final_kv_bytes=bank.nbytes,kv_tokens=bank.tokens,query_buffer_bytes=ctl.query_bytes,process_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None,process_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=="cuda" else None,compression_count=len(metrics.events),forced_count=sum(e["forced"] for e in metrics.events),events=metrics.events,extra_prefill_rebuild_seconds=0.0,actual_kv_tensor_bytes=sum(s.kv.numel()*s.kv.element_size() for s in bank.segments),position_protection_bytes=sum(s.nbytes-s.kv.numel()*s.kv.element_size() for s in bank.segments),completed_turns=ctl.current+1,metadata_note="KV cap includes position/protection metadata and TE output coexistence; query buffers and FP32 workspace reported separately")

def summarize(records,destination):
    groups={}
    for r in records:groups.setdefault(r["policy"],[]).append(r)
    rows=[]
    for name,rs in groups.items():
        successful=[r for r in rs if r["status"]=="ok"]
        scores=[r.get("task_score") or 0 for r in rs]
        rows.append(dict(policy=name,n=len(rs),ok=len(successful),infeasible=sum(r["status"]=="infeasible" for r in rs),errors=sum(r["status"]=="error" for r in rs),quality=sum(scores)/len(rs) if rs[0]["backend"]!="synthetic" else None,mean_wall_seconds=sum(r["wall_seconds"] for r in rs)/len(rs),peak_kv_bytes=max(r["peak_kv_bytes"] for r in rs),forced_count=sum(r["forced_count"] for r in rs)))
    if rows:
        with open(destination,"w") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    return rows

def execute(cfg,output,split="validation",frozen=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tasks=[t for t in json.loads(Path(cfg["dataset"]).read_text()) if t["split"]==split]
    if not tasks:raise ValueError("empty split")
    all_tasks=json.loads(Path(cfg["dataset"]).read_text())
    if len({t["id"] for t in all_tasks})!=len(all_tasks):raise ValueError("duplicate IDs across splits")
    if split=="test" and frozen is None:raise ValueError("test requires a frozen validation artifact")
    policies=core_policies()
    for name,values in cfg.get("policy_overrides",{}).items():
        policies[name]=Policy(**values)
    if frozen:
        artifact=json.loads(Path(frozen).read_text())
        if artifact["selection_split"]!="validation" or artifact["dataset_hash"]!=digest(all_tasks):raise ValueError("invalid frozen selection")
        if artifact["control_hash"]!=digest({k:v for k,v in cfg.items() if k not in ("policies","policy_overrides")}):raise ValueError("frozen controls mismatch")
        policies["R"]=Policy(**artifact["ratio_policy"])
        policies["T"]=Policy(**artifact["policy"])
        policies["TR"]=Policy(**{**artifact["policy"],"adaptive_ratio":True,"ratio_min":artifact["ratio_policy"]["ratio_min"],"ratio_max":artifact["ratio_policy"]["ratio_max"]})
    policies={name:policy for name,policy in policies.items() if name in cfg.get("policies",list(policies))}
    # Every resume key includes code, config, dataset and frozen policy bytes.
    code_hash=digest({str(f.relative_to(Path(__file__).parent)):f.read_text() for f in sorted(Path(__file__).parent.glob("*.py"))})
    manifest=dict(config=cfg,dataset_hash=digest(all_tasks),code_hash=code_hash,policies={n:asdict(p) for n,p in policies.items()},split=split,environment=runtime())
    run_hash=digest(manifest);manifest["run_hash"]=run_hash
    manifest_path=output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["run_hash"]!=run_hash:raise ValueError("output belongs to different run")
    manifest_path.write_text(json.dumps(manifest,indent=2))
    journal=output/"tasks.jsonl";records=[]
    if journal.exists():
        # Recover an interrupted final JSONL write, preserve all complete records.
        raw=journal.read_bytes();lines=raw.splitlines(keepends=True);good=0
        for i,line in enumerate(lines):
            try:record=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                break
            records.append(record);good+=len(line)
        if good!=len(raw):journal.write_bytes(raw[:good])
        elif raw and not raw.endswith(b"\n"):journal.write_bytes(raw+b"\n")
    done={(r["task_id"],r["policy"]) for r in records}
    model=tokenizer=None
    if cfg["backend"]!="synthetic":
        from .model import load_model
        model,tokenizer=load_model(cfg["model"],cfg["revision"],cfg["dtype"])
    with journal.open("a") as f:
        for task in tasks:
            for name,policy in policies.items():
                if (task["id"],name) in done:continue
                r=run_one(task,name,policy,cfg,model,tokenizer);r["run_hash"]=run_hash;r["time_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
                f.write(json.dumps(r)+"\n");f.flush();os.fsync(f.fileno());records.append(r)
    return summarize(records,output/"summary.csv")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/synthetic.json");ap.add_argument("--output",default="results/synthetic");ap.add_argument("--split",choices=["dev","validation","test"],default="validation");ap.add_argument("--frozen")
    args=ap.parse_args();print(json.dumps(execute(json.loads(Path(args.config).read_text()),args.output,args.split,args.frozen),indent=2))
if __name__=="__main__":main()
