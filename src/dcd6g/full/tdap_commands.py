from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def _flag(enabled: bool, name: str) -> list[str]:
    return [name] if enabled else []


def _bash_path(path: Path) -> str:
    return path.as_posix()


def _shell_arg(part: object) -> str:
    text = str(part)
    if text.startswith("${") or text.startswith("$"):
        return text
    return shlex.quote(text)


def _join_command(parts: list[str]) -> str:
    return " ".join(_shell_arg(part) for part in parts if str(part) != "")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_command(config: dict[str, Any], dataset: dict[str, Any], model: str, seed: int) -> str:
    train = config["training"]
    tdap_root = Path(config.get("tdap_root", "third_party/TDAP"))
    file_name = "train_models_nc.py" if dataset["task"] == "node_classification" else "train_models.py"
    parts = [
        "python3",
        file_name,
        "-task",
        dataset["task"],
        "-dataset",
        dataset["tdap_name"],
        "-num_graphs",
        dataset["num_graphs"],
        "-context",
        dataset["context"],
        "-target_ts",
        dataset["target_ts"],
        "-model_name",
        model,
        "-emb_size",
        train["emb_size"],
        "-decoder_sizes",
        *train["decoder_sizes"],
        "-chebyK",
        train["chebyK"],
        "-dys_struc_head",
        16,
        8,
        8,
        "-dys_struc_layer",
        train["emb_size"],
        "-dys_temp_head",
        16,
        "-dys_temp_layer",
        train["emb_size"],
        "-dys_spa_drop",
        0.1,
        "-dys_temp_drop",
        0.5,
        "-nepochs",
        train["nepochs"],
        "-learning_rate",
        train["learning_rate"],
        "-neg_weight",
        train["neg_weight"],
        "-neg_sample_size",
        train["neg_sample_size"],
        "-batch_size",
        train["batch_size"],
        "-min_time_perc",
        train["min_time_perc"],
        "-device",
        "${DEVICE}",
        "-seed",
        seed,
        "-to_save",
        "-logging",
    ]
    parts += _flag(dataset.get("featureless", False), "-featureless")
    parts += _flag(dataset.get("dyn_feats", False), "-dyn_feats")
    return f"(cd {shlex.quote(_bash_path(tdap_root / 'models'))} && {_join_command(parts)})"


def attack_command(config: dict[str, Any], dataset: dict[str, Any], model: str, seed: int, epsilon: float) -> str:
    attack = config["attack"]
    tdap_root = Path(config.get("tdap_root", "third_party/TDAP"))
    result_dir = (
        Path(f"results_{model.lower()}")
        / dataset["tdap_name"]
        / "multi_targets"
        / attack["method"]
        / attack["constraint"]
    )
    result_file = (
        f"results_td_tg{attack['ntargets']}_n{dataset['num_graphs']}"
        f"_c{dataset['context']}t{dataset['target_ts']}_e{epsilon}"
        f"_eb{attack['epsilon1']}_l{attack['lambda1']}_seed{seed}.txt"
    )
    file_name = "src/main_feat.py" if attack["constraint"] == "noise_feat" else "src/main.py"
    parts = [
        "python3",
        "run.py",
        "-file",
        file_name,
        "-constraint",
        attack["constraint"],
        "-epsilon",
        epsilon,
        "-epsilon1",
        attack["epsilon1"],
        "-model_name",
        model,
        "-saved_model",
        f"models/{model}/{dataset['tdap_name']}",
        "-dataset",
        dataset["tdap_name"],
        "-task",
        dataset["task"],
        "-num_graphs",
        dataset["num_graphs"],
        "-ntargets",
        attack["ntargets"],
        "-khop",
        attack["khop"],
        "-num_steps",
        attack["num_steps"],
        "-lambda1",
        attack["lambda1"],
        "-context",
        dataset["context"],
        "-target_ts",
        dataset["target_ts"],
        "-method",
        attack["method"],
        "-num_samples",
        attack["num_samples"],
        "-device",
        "${DEVICE}",
        "-seed",
        seed,
        "-lr_init",
        attack["lr_init"],
        "-nprocs",
        1,
        "-sampling",
        attack["sampling"],
        "-inits",
        attack["inits"],
    ]
    parts += _flag(dataset.get("featureless", False), "-featureless")
    parts += _flag(dataset.get("dyn_feats", False), "-dyn_feats")
    parts += _flag(attack.get("neg_sampling", False), "-neg_sampling")
    parts += _flag(attack.get("use_optim", False), "-use_optim")
    cmd = _join_command(parts)
    return (
        f"mkdir -p {shlex.quote(_bash_path(tdap_root / result_dir))}\n"
        f"(cd {shlex.quote(_bash_path(tdap_root))} && {cmd}) > "
        f"{shlex.quote(_bash_path(tdap_root / result_dir / result_file))}"
    )


def build_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            for seed in config["seeds"]:
                jobs.append({"stage": "train", "dataset": dataset, "model": model, "seed": seed})
                for epsilon in config["epsilons"]:
                    jobs.append(
                        {
                            "stage": "attack",
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "epsilon": epsilon,
                        }
                    )
    return jobs


def write_train_script(config: dict[str, Any], out: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'TDAP_ROOT="${TDAP_ROOT:-third_party/TDAP}"',
        'GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"',
        "i=0",
        "mapfile -t GPUS < <(printf '%s\\n' ${GPU_IDS})",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for job in build_jobs(config):
        if job["stage"] != "train":
            continue
        key = (job["dataset"]["tdap_name"], job["model"])
        if key in seen:
            continue
        seen.add(key)
        lines.extend(
            [
                'DEVICE="cuda:${GPUS[$((i % ${#GPUS[@]}))]}"',
                f"echo '[train] {key[0]} {key[1]} seed={job['seed']} on ' $DEVICE",
                train_command(config, job["dataset"], job["model"], job["seed"]),
                "i=$((i+1))",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_attack_script(config: dict[str, Any], out: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"',
        "i=0",
        "mapfile -t GPUS < <(printf '%s\\n' ${GPU_IDS})",
        "",
    ]
    for job in build_jobs(config):
        if job["stage"] != "attack":
            continue
        lines.extend(
            [
                'DEVICE="cuda:${GPUS[$((i % ${#GPUS[@]}))]}"',
                (
                    f"echo '[attack] {job['dataset']['tdap_name']} {job['model']} "
                    f"eps={job['epsilon']} seed={job['seed']} on ' $DEVICE"
                ),
                attack_command(
                    config,
                    job["dataset"],
                    job["model"],
                    job["seed"],
                    job["epsilon"],
                ),
                "i=$((i+1))",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_jobs_tsv(config: dict[str, Any], out: Path) -> int:
    lines = ["stage\tdataset\tmodel\tseed\tepsilon"]
    count = 0
    for job in build_jobs(config):
        if job["stage"] != "attack":
            continue
        count += 1
        lines.append(
            "\t".join(
                [
                    job["stage"],
                    job["dataset"]["tdap_name"],
                    job["model"],
                    str(job["seed"]),
                    "" if job["stage"] == "train" else str(job["epsilon"]),
                ]
            )
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return count


def write_slurm_array(out: Path, job_count: int) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=dcd6g-tdap",
        f"#SBATCH --array=0-{max(job_count - 1, 0)}",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=48G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "#SBATCH --error=logs/%x_%A_%a.err",
        "",
        "set -euo pipefail",
        "mkdir -p logs",
        "JOB_FILE=${JOB_FILE:-scripts/generated/jobs.tsv}",
        "LINE=$(awk -v id=$((SLURM_ARRAY_TASK_ID+2)) 'NR==id {print}' \"$JOB_FILE\")",
        "IFS=$'\\t' read -r STAGE DATASET MODEL SEED EPSILON <<< \"$LINE\"",
        "export DEVICE=cuda:0",
        "export PYTHONPATH=${PYTHONPATH:-}:src",
        "python -m dcd6g.full.tdap_single_job --config configs/full_tdap_matrix.json "
        "--stage \"$STAGE\" --dataset \"$DATASET\" --model \"$MODEL\" --seed \"$SEED\" --epsilon \"$EPSILON\"",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TDAP full-reproduction commands.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_tdap_matrix.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/generated"))
    parser.add_argument("--mode", choices=["train", "attack", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "all"):
        write_train_script(config, args.out_dir / "train_models.sh")
    if args.mode in ("attack", "all"):
        write_attack_script(config, args.out_dir / "run_tdpgd_attacks.sh")
    job_count = write_jobs_tsv(config, args.out_dir / "jobs.tsv")
    write_slurm_array(args.out_dir / "slurm_array.sbatch", job_count)
    print(f"Wrote generated scripts to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
