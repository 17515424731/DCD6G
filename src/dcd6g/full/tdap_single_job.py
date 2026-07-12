from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from .tdap_commands import attack_command, load_config, train_command


def find_dataset(config: dict, name: str) -> dict:
    for dataset in config["datasets"]:
        if dataset["tdap_name"] == name or dataset["paper_name"] == name:
            return dataset
    raise SystemExit(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one generated TDAP train/attack job.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_tdap_matrix.json"))
    parser.add_argument("--stage", choices=["train", "attack"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epsilon", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = find_dataset(config, args.dataset)
    command = (
        train_command(config, dataset, args.model, args.seed)
        if args.stage == "train"
        else attack_command(config, dataset, args.model, args.seed, float(args.epsilon))
    )
    env = os.environ.copy()
    env.setdefault("DEVICE", "cuda:0")
    subprocess.run(command, shell=True, check=True, env=env)


if __name__ == "__main__":
    main()

