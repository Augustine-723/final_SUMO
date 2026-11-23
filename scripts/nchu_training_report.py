"""Audit Phase 6 schedules and pairings, aggregate validation and plot learning curves."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nchu_demand import ROOT


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    config = json.loads((ROOT / "configs/phase6.json").read_text())
    seeds = config["training_seeds"]
    names = config["scenarios"]
    test_cases = {(name, str(seed)) for name in names for seed in config["validation_traffic_seeds"]}
    required = {(alg, "core", seed) for alg in ("DQN", "PPO") for seed in seeds}
    required.update(("PPO", variant, seeds[0]) for variant in ("low_lr", "pedestrian"))
    valid = []
    references = {}
    schedules = {}
    curves = defaultdict(list)
    baseline = read_csv(root / "baselines.csv")
    for policy in ("fixed", "actuated"):
        selected = [r for r in baseline if r["policy"] == policy]
        if {(r["counts"], r["traffic_seed"]) for r in selected} != test_cases or len(selected) != len(test_cases):
            raise ValueError("Incomplete baseline comparison")
        for row in selected:
            key = (row["counts"], row["traffic_seed"])
            if references.setdefault(key, row["route_sha256"]) != row["route_sha256"]:
                raise ValueError("Baseline traffic mismatch")
            valid.append({"method": policy, "model_seed": "", **{k:v for k,v in row.items() if k != "policy"}})
    for file in sorted(root.glob("*/run.json")):
        meta = json.loads(file.read_text())
        algo, variant, seed = meta["algorithm"], meta["variant"], meta["training_seed"]
        required.remove((algo, variant, seed))
        method = algo if variant == "core" else f"{algo}_{variant}"
        folder = file.parent
        if hashlib.sha256((folder / "final.zip").read_bytes()).hexdigest() != meta["model_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        training = read_csv(folder / "training.csv")
        seq = [(r["counts"], r["traffic_seed"], r["route_sha256"]) for r in training]
        if (len(seq) != config["episodes"] or schedules.setdefault(seed, seq) != seq
                or Counter(r[0] for r in seq) != {n: config["episodes"] // len(names) for n in names}
                or any((n, s) in test_cases for n, s, _ in seq)):
            raise ValueError("Unpaired or incomplete training schedule")
        curves[method].append([float(r["episode_reward"]) for r in training])
        rows = read_csv(folder / "validation.csv")
        if len(rows) != len(test_cases) or {(r["counts"],r["traffic_seed"]) for r in rows} != test_cases:
            raise ValueError("Validation cases differ")
        for row in rows:
            if row["route_sha256"] != references[row["counts"], row["traffic_seed"]]:
                raise ValueError("Validation routes differ from baseline")
            valid.append({"method": method, "model_seed": seed, **row})
    if required:
        raise ValueError(f"Missing training runs: {required}")
    save_csv(root / "validation_all.csv", valid)
    groups = defaultdict(list)
    for row in valid:
        groups[row["method"], row["counts"]].append(row)
    summary = []
    for (method, scenario), rows in sorted(groups.items()):
        rec = {"method":method, "scenario":scenario, "episodes":len(rows)}
        for key in ("mean_vehicle_wait_s","mean_pedestrian_wait_s","mean_total_queue","vehicle_arrivals"):
            rec[key] = round(mean(float(r[key]) for r in rows), 3)
        for key in ("unfinished_vehicles","unfinished_pedestrians","collisions","vehicle_teleports"):
            rec[key] = sum(int(r[key]) for r in rows)
        summary.append(rec)
    save_csv(root / "summary.csv", summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in ("DQN", "PPO"):
        series = curves[method]
        for i, run in enumerate(series):
            values = [mean(run[j:j+3]) for j in range(0, len(run), 3)]
            ax.plot(range(3, len(run)+1, 3), values, color=("tab:blue" if method=="DQN" else "tab:orange"),
                    alpha=.22, linewidth=1)
        average = [mean(run[j+k] for run in series for k in range(3)) for j in range(0, len(series[0]), 3)]
        ax.plot(range(3, len(series[0])+1, 3), average, label=method, linewidth=2)
    ax.set(xlabel="Training episode (three-profile blocks)", ylabel="Mean raw queue return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)
    print("PHASE6_SUMMARY_BEGIN")
    print((root / "summary.csv").read_text().strip())
    print("PHASE6_SUMMARY_END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
