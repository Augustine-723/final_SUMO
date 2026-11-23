"""Audit the frozen test matrix and summarize paired tests conditional on the trained models."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONF = json.loads((ROOT / "configs/phase7.json").read_text())
METRICS = ("mean_vehicle_wait_s", "mean_pedestrian_wait_s", "mean_total_queue", "mean_vehicle_trip_s", "vehicle_arrivals")
LABELS = {"offpeak":"Off-peak", "balanced":"Balanced", "directional":"E-heavy", "reversal":"Gate reversal", "ped_peak":"Pedestrian peak"}
CORE = ["fixed","actuated","DQN","PPO"]


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summarize(files, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in files:
        with file.open() as f:
            rows.extend(csv.DictReader(f))
    cases = defaultdict(dict)
    seeds = range(CONF["traffic_seed_first"], CONF["traffic_seed_first"]+CONF["traffic_seed_count"])
    profiles = list(CONF["profiles"])
    expected = {(m,str(s)) for m in ("DQN","PPO") for s in CONF["models"]} | {("fixed",""),("actuated",""),("PPO_low_lr","11"),("PPO_pedestrian","11")}
    for r in rows:
        case = cases[r["scenario"], int(r["traffic_seed"])]
        key = r["method"], r["model_seed"]
        if key in case:
            raise ValueError(f"Duplicate {key}")
        case[key] = r
    if set(cases) != {(p,s) for p in profiles for s in seeds}:
        raise ValueError("Incomplete test matrix")
    for (p,s), case in cases.items():
        if set(case) != expected or len({r["route_sha256"] for r in case.values()}) != 1:
            raise ValueError(f"Pairing failure {p} {s}")
        spec = CONF["profiles"][p]
        for r in case.values():
            if int(r["vehicle_arrivals"])+int(r["unfinished_vehicles"]) != sum(map(sum,spec["vehicles"])) or int(r["pedestrian_arrivals"])+int(r["unfinished_pedestrians"]) != 4*sum(spec["walkers"]):
                raise ValueError("Demand accounting failure")
    write(out/"episodes.csv",rows)
    summary, contrasts, behavior = [], [], []
    rng = np.random.default_rng(7001)
    for p in profiles:
        values = {}
        for method in CORE + ["PPO_low_lr","PPO_pedestrian","PPO_seed11"]:
            def selected(seed):
                return [r for (m,t),r in cases[p,seed].items() if (m,t)==("PPO","11")] if method=="PPO_seed11" else [r for (m,t),r in cases[p,seed].items() if m==method]
            for metric in METRICS:
                arr = np.array([np.mean([float(r[metric]) for r in selected(s)]) for s in seeds])
                values[method,metric] = arr
                summary.append(dict(scenario=p,method=method,metric=metric,mean=float(arr.mean()),sd=float(arr.std(ddof=1))))
        for first,ref in [("DQN","fixed"),("PPO","fixed"),("DQN","actuated"),("PPO_pedestrian","PPO_seed11"),("PPO_low_lr","PPO_seed11")]:
            for metric in METRICS[:2]:
                a,b=values[first,metric],values[ref,metric]
                d=a-b
                lo,hi=np.quantile(rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1),[.025,.975])
                contrasts.append(dict(scenario=p,method=first,reference=ref,metric=metric,
                                      delta=float(d.mean()),ci_low=float(lo),ci_high=float(hi),
                                      change_pct=float(100*d.mean()/b.mean())))
        for m,t in sorted(expected):
            if not t:
                continue
            rr=[cases[p,s][m,t] for s in seeds]
            behavior.append(dict(scenario=p,method=m,model_seed=t,
                                 ew_request_fraction=sum(int(r["ew_requests"]) for r in rr)/(540*len(rr)),
                                 unique_action_traces=len({r["action_sha256"] for r in rr}),
                                 matches_dqn11_phase=sum(r["phase_sha256"]==cases[p,s]["DQN","11"]["phase_sha256"] for r,s in zip(rr,seeds))))
    write(out/"summary.csv",summary)
    write(out/"paired.csv",contrasts)
    write(out/"behavior.csv",behavior)
    safety={k:sum(int(r[k]) for r in rows) for k in ("unfinished_vehicles","unfinished_pedestrians","collisions","vehicle_teleports")}
    (out/"audit.json").write_text(json.dumps({"episodes":len(rows),"paired_cases":len(cases),**safety},indent=2)+"\n")
    palette=["#535e69","#238071","#315e98","#c16a36"]
    fig,axes=plt.subplots(2,1,figsize=(9,6.5),constrained_layout=True)
    for ax,metric,title in zip(axes,METRICS,["Vehicle waiting time (s)","Pedestrian stopped time (s)"]):
        for idx,m in enumerate(CORE):
            rr=[next(r for r in summary if (r["scenario"],r["method"],r["metric"])==(p,m,metric)) for p in profiles]
            ax.bar(np.arange(len(profiles))+(idx-1.5)*.18,[r["mean"] for r in rr],.18,label=m,color=palette[idx])
        ax.set_ylabel(title)
        ax.set_xticks(range(len(profiles)),[LABELS[p] for p in profiles])
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(ncol=4,frameon=False)
    fig.savefig(out/"comparison.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    summarize(sorted(args.shards.glob("**/*_[0-3].csv")),args.out)
