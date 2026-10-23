"""Generate summary from raw logs and a quality/time figure, no fabricated scores."""
import argparse,json
from pathlib import Path
from online_kv.experiments import summarize
ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args()
folder=Path(args.run);rows=summarize([json.loads(line) for line in (folder/"tasks.jsonl").read_text().splitlines()],folder/"summary.csv")
valid=[r for r in rows if r["quality"] is not None]
if not valid:raise SystemExit("NOT RUN: synthetic diagnostics have no task quality; no quality curve generated")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
for row in valid:
    ax.scatter(row["mean_wall_seconds"],row["quality"])
    ax.annotate(f'{row["policy"]} ({row["peak_kv_bytes"]/1048576:.1f} MiB; infeasible={row["infeasible"]})',(row["mean_wall_seconds"],row["quality"]),fontsize=8)
ax.set(xlabel="Mean end-to-end wall time (s)",ylabel="Exact-match success rate",title="Paired tasks, fixed cache cap; failures counted as zero")
fig.tight_layout();fig.savefig(folder/"quality-time.png",dpi=180)
# Pair each policy with the delay1 baseline on identical task IDs, failures retained.
records=[json.loads(line) for line in (folder/"tasks.jsonl").read_text().splitlines()]
base={r["task_id"]:r for r in records if r["policy"]=="delay1"};pairs=[]
for r in records:
    b=base.get(r["task_id"])
    if b and r["policy"]!="delay1":pairs.append(dict(task_id=r["task_id"],policy=r["policy"],score_delta=(r.get("task_score") or 0)-(b.get("task_score") or 0),wall_delta=r["wall_seconds"]-b["wall_seconds"]))
(folder/"paired-deltas.json").write_text(json.dumps(pairs,indent=2))
