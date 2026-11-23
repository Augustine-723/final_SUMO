"""Run paired full-horizon baseline episodes on the NCHU gate."""
import argparse
import csv
from pathlib import Path

from nchu_demand import ARMS, ROOT
from nchu_env import GateEnv


def run(counts, seed, policy):
    env = GateEnv(counts)
    try:
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        while True:
            group = env.signal.phase // 4
            if policy == "fixed":
                switch = env.signal.elapsed >= 20
            else:
                demand = [sum(env.queues[ARMS.index(a)] for a in arms) for arms in env.signal.arms]
                switch = demand[1 - group] > demand[group]
            action = 1 - group if switch else group
            obs, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(obs):
                raise RuntimeError("Observation outside declared space")
            total_reward += reward
            if terminated or truncated:
                if total_reward != info["episode_reward"]:
                    raise RuntimeError("Queue accounting mismatch")
                return {"counts": counts.name, "policy": policy, **info}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, nargs="+", default=sorted((ROOT / "data/nchu").glob("*_synthetic.csv")))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out", type=Path, default=ROOT / "results/phase5/baselines.csv")
    args = parser.parse_args()
    rows = []
    for path in args.counts:
        for seed in args.seeds:
            pair = [run(path, seed, policy) for policy in ("fixed", "actuated")]
            if pair[0]["route_sha256"] != pair[1]["route_sha256"]:
                raise RuntimeError("The controllers received different trips")
            rows.extend(pair)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)
