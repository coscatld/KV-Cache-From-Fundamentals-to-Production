"""Explicit validation grid; no synthetic policy selection, no test labels."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from .experiments import execute,digest
from .policies import Policy

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--grid",default="configs/grid.json");ap.add_argument("--output",default="results/search");args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if cfg["backend"]=="synthetic":raise ValueError("cannot tune quality thresholds on synthetic tensor diagnostics")
    candidates=json.loads(Path(args.grid).read_text());trials=[];start=time.perf_counter()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for i,candidate in enumerate(candidates):
        overrides={"T":{**asdict(Policy()),**candidate["T"]},"R":{**asdict(Policy(name="fixed",delay=1,adaptive_ratio=True)),**candidate["R"]}}
        run_cfg={**cfg,"policies":["T","R"],"policy_overrides":overrides}
        rows=execute(run_cfg,out/f"trial-{i}","validation")
        trials.append(dict(index=i,parameters=overrides,summary=rows))
        (out/"all-trials.json").write_text(json.dumps(trials,indent=2))
    selected={}
    for group in ("T","R"):
        valid=[(t,next(r for r in t["summary"] if r["policy"]==group)) for t in trials]
        valid=[(t,r) for t,r in valid if r["infeasible"]==0 and r["errors"]==0]
        if not valid:raise RuntimeError("no feasible validation candidate")
        t,row=max(valid,key=lambda tr:(tr[1]["quality"],-tr[1]["mean_wall_seconds"]))
        selected[group]=t["parameters"][group]
    artifact=dict(selection_split="validation",dataset_hash=digest(json.loads(Path(cfg["dataset"]).read_text())),control_hash=digest({k:v for k,v in cfg.items() if k not in ("policies","policy_overrides")}),policy=selected["T"],ratio_policy=selected["R"],all_trials="all-trials.json",search_wall_seconds=time.perf_counter()-start,criterion="max quality among feasible; ties min wall time")
    (out/"frozen.json").write_text(json.dumps(artifact,indent=2))
if __name__=="__main__":main()
