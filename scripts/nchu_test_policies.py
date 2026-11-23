"""Evaluate frozen Phase 6 models on paired Phase 7 traffic draws."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch

from nchu_baselines import run
from nchu_demand import ARMS, FIELDS, ROOT
from nchu_env import GateEnv
from nchu_train import ALGORITHMS, save_csv

CONF = json.loads((ROOT / "configs/phase7.json").read_text())


def profile_file(name, directory):
    spec = CONF["profiles"][name]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".csv")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        for slot, volumes in enumerate(spec["vehicles"]):
            for i, arm in enumerate(ARMS):
                fields = ["nchu_gate", "synthetic", name, "", 900 * slot, arm]
                writer.writerow([*fields, arm, "pedestrian", spec["walkers"][slot]])
                total = volumes[i]
                right, through = total // 4, total // 2
                for offset, count in ((3,right),(2,through),(1,total-right-through)):
                    dest = ARMS[(i+offset) % 4]
                    writer.writerow([*fields, dest, "car", count-count//3])
                    writer.writerow([*fields, dest, "motorcycle", count//3])
    return path


def load_models(root):
    models = []
    for file in sorted(root.glob("policy-*/run.json")):
        meta = json.loads(file.read_text())
        checkpoint = file.parent / "final.zip"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta["model_sha256"]:
            raise ValueError("Invalid frozen checkpoint")
        model = ALGORITHMS[meta["algorithm"]].load(checkpoint, device="cpu")
        name = meta["algorithm"] if meta["variant"] == "core" else "PPO_" + meta["variant"]
        models.append((name, meta["training_seed"], model))
    expected = {(a, s) for a in ALGORITHMS for s in CONF["models"]} | {("PPO_low_lr",11),("PPO_pedestrian",11)}
    if {(a,s) for a,s,_ in models} != expected or len(models) != len(expected):
        raise ValueError("Missing or duplicate Phase 6 final models")
    return models


def evaluate(model, path, seed):
    env = GateEnv(path)
    try:
        obs, _ = env.reset(seed=seed)
        requests = []
        phases = []
        while True:
            action = int(model.predict(obs, deterministic=True)[0])
            requests.append(str(action))
            obs, _, terminated, truncated, metrics = env.step(action)
            phases.append(str(env.signal.phase))
            if terminated or truncated:
                return {**metrics, "ew_requests": requests.count("0"),
                        "action_sha256": hashlib.sha256("".join(requests).encode()).hexdigest(),
                        "phase_sha256": hashlib.sha256("".join(phases).encode()).hexdigest()}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("scenario", choices=CONF["profiles"])
    parser.add_argument("shard", type=int, choices=range(4))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = profile_file(args.scenario, args.out / "inputs")
    models = load_models(args.models)
    rows = []
    start = CONF["traffic_seed_first"] + 5 * args.shard
    for seed in range(start, start + 5):
        paired = []
        for method in ("fixed","actuated"):
            info = run(path, seed, method)
            for field in ("policy","counts"):
                info.pop(field)
            paired.append({"scenario":args.scenario,"method":method,"model_seed":"",**info,
                           "ew_requests":"","action_sha256":"","phase_sha256":""})
        for method, model_seed, model in models:
            paired.append({"scenario":args.scenario,"method":method,"model_seed":model_seed,
                           **evaluate(model,path,seed)})
        if len({r["route_sha256"] for r in paired}) != 1:
            raise ValueError("Unpaired routes")
        rows.extend(paired)
    save_csv(args.out / f"{args.scenario}_{args.shard}.csv", rows)


if __name__ == "__main__":
    main()
