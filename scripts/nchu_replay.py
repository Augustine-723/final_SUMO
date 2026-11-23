"""Reload one saved Phase 6 model and run a full NCHU episode."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from nchu_train import ALGORITHMS, DATA, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing run.json and final.zip")
    parser.add_argument("--counts", type=Path, default=DATA / "morning_synthetic.csv")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    metadata = json.loads((args.run / "run.json").read_text())
    checkpoint = args.run / "final.zip"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["model_sha256"]:
        parser.error("Checkpoint differs from the recorded SHA-256")
    torch.set_num_threads(1)
    model = ALGORITHMS[metadata["algorithm"]].load(checkpoint, device="cpu")
    print(json.dumps(evaluate(model, args.counts, args.seed), indent=2))
