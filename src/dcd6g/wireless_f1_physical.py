from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .td_pgd import TD_PGD_Attacker, TDPGDConfig
from .trained_models import (
    FEATURE_NAMES,
    benign_sequences,
    binary_f1,
    build_temporal_edge_data,
    graph_dict,
    predict_model,
    stable_seed,
    train_model,
)


MODEL_NAMES = ("Vanilla TGNN", "DG-Mamba", "DCD")


def _torch():
    import torch

    return torch


def attack_sequences(model, data, normalized_sequences, graph, epsilon, device, seed):
    torch = _torch()
    attacked = normalized_sequences.copy()
    model.eval()
    presence_index = FEATURE_NAMES.index("edge_present")
    mean = float(data.mean[0, 0, presence_index])
    std = float(data.std[0, 0, presence_index])
    test_times = sorted(int(value) for value in np.unique(data.times[data.test_mask]) if int(value) > 0)
    for target_time in test_times:
        row_indices = np.where(data.test_mask & (data.times == target_time))[0]
        if len(row_indices) == 0:
            continue
        batch = torch.as_tensor(attacked[row_indices], dtype=torch.float32, device=device)
        labels = torch.as_tensor(data.labels[row_indices], dtype=torch.float32, device=device)
        src = torch.as_tensor(data.src[row_indices], dtype=torch.long, device=device)
        dst = torch.as_tensor(data.dst[row_indices], dtype=torch.long, device=device)

        def loss_fn(state):
            values = batch.clone()
            presence = state["adjacency"][src, dst]
            values[:, -1, presence_index] = (presence - mean) / std
            return torch.nn.functional.binary_cross_entropy_with_logits(model(values), labels)

        attacker = TD_PGD_Attacker(
            loss_fn=loss_fn,
            config=TDPGDConfig(num_steps=10, learning_rate=0.2, num_rounds=10, device=device),
        )
        result = attacker.attack(graph, epsilon, target_time=target_time)
        adjacency = result["attacked_adjacency"]
        presence = adjacency[data.src[row_indices], data.dst[row_indices]]
        attacked[row_indices, -1, presence_index] = (presence - mean) / std
    return attacked


def write_f1_outputs(rows, out_dir):
    csv_path = out_dir / "wireless_f1_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "epsilon", "condition", "f1"))
        writer.writeheader()
        writer.writerows(rows)

    conditions = ("Clean", "Benign Fluctuation", "TDAP 5%", "TDAP 10%")
    colors = {"Vanilla TGNN": "#6b7280", "DG-Mamba": "#e9a23b", "DCD": "#4361ee"}
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for name in MODEL_NAMES:
        values = [next(row["f1"] for row in rows if row["model"] == name and row["condition"] == condition) for condition in conditions]
        axis.plot(conditions, values, marker="o", linewidth=2.2, label=name, color=colors[name])
    axis.set_ylabel("F1-score")
    axis.set_ylim(0.0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "Evaluation_a.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Evaluation_a.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate three temporal graph models under TD-PGD")
    parser.add_argument("--graph", type=Path, default=Path("outputs/disac_temporal_graph_sample.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/trained_results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if _torch().cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.graph, allow_pickle=False) as archive:
        graph = graph_dict(archive)
    data = build_temporal_edge_data(graph, context=args.context)
    clean = data.normalized()
    benign = data.normalized(benign_sequences(data, args.seed))
    rows = []
    embedding_output = {}

    for model_name in MODEL_NAMES:
        checkpoint = args.out_dir / "checkpoints" / f"{model_name.lower().replace(' ', '_').replace('-', '_')}.pt"
        model, threshold = train_model(
            model_name,
            data,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            checkpoint_path=checkpoint,
        )
        conditions = {
            "Clean": clean,
            "Benign Fluctuation": benign,
            "TDAP 5%": attack_sequences(model, data, clean, graph, 0.05, args.device, stable_seed(model_name, args.seed)),
            "TDAP 10%": attack_sequences(model, data, clean, graph, 0.10, args.device, stable_seed(model_name, args.seed)),
        }
        for condition, sequences in conditions.items():
            prediction, _, embedding = predict_model(model, sequences[data.test_mask], threshold, args.device)
            score = binary_f1(data.labels[data.test_mask], prediction)
            epsilon = "clean" if condition == "Clean" else "benign" if condition == "Benign Fluctuation" else "0.05" if "5%" in condition else "0.10"
            rows.append({"model": model_name, "epsilon": epsilon, "condition": condition, "f1": score})
            print(f"{model_name},{condition},F1={score:.4f}")
            if model_name in {"DG-Mamba", "DCD"} and condition in {"Clean", "Benign Fluctuation", "TDAP 10%"}:
                prefix = "dg_mamba" if model_name == "DG-Mamba" else "dcd"
                key = {
                    "Clean": "clean",
                    "Benign Fluctuation": "benign_fluctuation",
                    "TDAP 10%": "tdap_attack",
                }[condition]
                embedding_output[f"{prefix}_{key}"] = embedding.astype(np.float32)

    embedding_output["test_labels"] = data.labels[data.test_mask].astype(np.int64)
    np.savez_compressed(args.out_dir / "model_embeddings.npz", **embedding_output)
    write_f1_outputs(rows, args.out_dir)
    print("wrote", args.out_dir / "wireless_f1_scores.csv")
    print("wrote", args.out_dir / "model_embeddings.npz")


if __name__ == "__main__":
    main()
