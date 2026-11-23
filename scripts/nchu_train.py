"""Train a Phase 6 policy, reload its checkpoint and validate on paired traffic."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from nchu_demand import ROOT
from nchu_env import GateEnv

CONFIG = ROOT / "configs/phase6.json"
DATA = ROOT / "data/nchu"
ALGORITHMS = {"DQN": DQN, "PPO": PPO}


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule(config, seed, episodes):
    order = random.Random(seed)
    counts = config["scenarios"].copy()
    sequence = []
    while len(sequence) <= episodes:  # SB3 automatically resets after the final episode.
        order.shuffle(counts)
        for name in counts:
            sequence.append((name, 1_000_000 + seed * 10000 + len(sequence)))
    return sequence


class TrainingTraffic(gym.Env):
    def __init__(self, sequence, pedestrian_weight):
        self.sequence = sequence
        self.weight = pedestrian_weight
        self.index = 0
        self.env = GateEnv(DATA / sequence[0][0])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.horizon = self.env.horizon
        self.history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.close()
        name, traffic_seed = self.sequence[self.index]
        self.index += 1
        self.env = GateEnv(DATA / name)
        if self.env.horizon != self.horizon:
            raise ValueError("Training profiles must have the same episode duration")
        self.reward_total = 0.0
        self.ped_stopped = 0
        return self.env.reset(seed=traffic_seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stopped = sum(self.env.pedestrian_wait.values())
        reward -= self.weight * (stopped - self.ped_stopped)
        self.ped_stopped = stopped
        self.reward_total += reward
        if terminated or truncated:
            self.history.append({"episode": self.index, "counts": self.env.counts.name,
                                 "training_return": self.reward_total, **info})
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


class Midpoint(BaseCallback):
    def __init__(self, step, path):
        super().__init__()
        self.step = step
        self.path = path

    def _on_step(self):
        if self.num_timesteps == self.step:
            self.model.save(self.path)
        return True


def evaluate(model, counts, seed):
    env = GateEnv(counts)
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                return {"counts": counts.name, **info}
    finally:
        env.close()


def train(algorithm, seed, variant, pilot, dest):
    conf = json.loads(CONFIG.read_text())
    config = conf[algorithm].copy()
    episodes = 3 if pilot else conf["episodes"]
    midpoint = 1 if pilot else conf["checkpoint_episode"]
    weight = 0
    if variant == "low_lr":
        config["learning_rate"] = conf["ppo_lr_pilot"]
    elif variant == "pedestrian":
        weight = conf["ppo_pedestrian_weight"]
    torch.set_num_threads(1)
    sequence = schedule(conf, seed, episodes)
    env = TrainingTraffic(sequence, weight)
    decisions = env.horizon // 5
    steps = episodes * decisions
    if algorithm == "PPO" and steps % config["n_steps"]:
        raise ValueError("PPO rollout would exceed the interaction budget")
    model_class = ALGORITHMS[algorithm]
    try:
        model = model_class("MlpPolicy", env, device="cpu", seed=seed,
                            policy_kwargs={"net_arch": [64, 64]}, **config)
        model.learn(steps, callback=Midpoint(decisions * midpoint, dest / "midpoint.zip"))
        if model.num_timesteps != steps or len(env.history) != episodes or model._n_updates < 1:
            raise RuntimeError("Incomplete training or no gradient updates")
        if any(not torch.isfinite(p).all().item() for p in model.policy.parameters()):
            raise RuntimeError("Nonfinite policy parameters")
        model.save(dest / "final.zip")
        save_csv(dest / "training.csv", env.history)
    finally:
        env.close()
    restored = model_class.load(dest / "final.zip", device="cpu")
    for key, tensor in model.policy.state_dict().items():
        if not torch.equal(tensor, restored.policy.state_dict()[key]):
            raise RuntimeError(f"Reloaded parameters differ at {key}")
    validation = [evaluate(restored, DATA / name, traffic_seed) for name in conf["scenarios"]
                  for traffic_seed in conf["validation_traffic_seeds"]]
    save_csv(dest / "validation.csv", validation)
    metadata = {"algorithm": algorithm, "variant": variant, "training_seed": seed, "steps": steps,
                "episodes": episodes, "midpoint_step": decisions * midpoint, "pedestrian_weight": weight,
                "policy_layers": [64, 64], "hyperparameters": config, "updates_sb3": model._n_updates,
                "python": platform.python_version(), "versions": {p: version(p) for p in (
                    "stable-baselines3", "torch", "gymnasium", "traci", "numpy")},
                "sumo": subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0],
                "input_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                                 for name in conf["scenarios"]},
                "model_sha256": hashlib.sha256((dest / "final.zip").read_bytes()).hexdigest()}
    (dest / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished {algorithm} {variant} seed {seed}: {steps} decisions, {len(validation)} validation episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, choices=[11, 22, 33], required=True)
    parser.add_argument("--variant", choices=["core", "low_lr", "pedestrian"], default="core")
    parser.add_argument("--pilot", action="store_true", help="Three full training episodes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.algorithm != "PPO" and args.variant != "core":
        parser.error("Sensitivity runs use PPO")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new run path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        train(args.algorithm, args.seed, args.variant, args.pilot, stage)
        os.rename(stage, args.out)
