from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


CONDITIONS = (
    ("clean", "Clean"),
    ("benign_fluctuation", "Benign Fluctuation"),
    ("tdap_attack", "TDAP Attack"),
)


def embed_jointly(groups: list[np.ndarray], seed: int) -> list[np.ndarray]:
    lengths = [len(group) for group in groups]
    features = np.vstack(groups).astype(np.float32)
    if len(features) < 3:
        padded = np.zeros((len(features), 2), dtype=np.float32)
        padded[:, : min(features.shape[1], 2)] = features[:, : min(features.shape[1], 2)]
        points = padded
    else:
        perplexity = min(18, max(2, (len(features) - 1) // 3))
        points = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=550,
        ).fit_transform(features)
    offsets = np.cumsum([0] + lengths)
    return [points[offsets[index] : offsets[index + 1]] for index in range(len(groups))]


def write_points(path: Path, groups: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "condition"))
        for points, (_, label) in zip(groups, CONDITIONS):
            for x_value, y_value in points:
                writer.writerow((float(x_value), float(y_value), label))


def plot(groups_by_model: list[tuple[str, list[np.ndarray]]], out_dir: Path) -> None:
    colors = {"Clean": "#2f80ed", "Benign Fluctuation": "#2a9d8f", "TDAP Attack": "#e76f51"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for axis, (title, groups) in zip(axes, groups_by_model):
        for points, (_, label) in zip(groups, CONDITIONS):
            axis.scatter(points[:, 0], points[:, 1], s=19, alpha=0.75, color=colors[label], label=label)
        axis.set_title(title)
        axis.set_xlabel("t-SNE Dimension 1")
        axis.set_ylabel("t-SNE Dimension 2")
        axis.grid(alpha=0.18)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "Evaluation_b.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Evaluation_b.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project trained DG-Mamba and DCD embeddings with t-SNE")
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/trained_results/model_embeddings.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/trained_results"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.embeddings, allow_pickle=False) as archive:
        dg_groups = [archive["dg_mamba_clean"], archive["dg_mamba_benign_fluctuation"], archive["dg_mamba_tdap_attack"]]
        dcd_groups = [archive["dcd_clean"], archive["dcd_benign_fluctuation"], archive["dcd_tdap_attack"]]
    dg_points = embed_jointly(dg_groups, args.seed)
    dcd_points = embed_jointly(dcd_groups, args.seed)
    write_points(args.out_dir / "dg_mamba_embeddings_tsne.csv", dg_points)
    write_points(args.out_dir / "dcd_consistency_embeddings_tsne.csv", dcd_points)
    plot([("DG-Mamba Latent Embeddings", dg_points), ("DCD Consistency Embeddings", dcd_points)], args.out_dir)
    print("wrote", args.out_dir / "Evaluation_b.pdf")
    print("wrote", args.out_dir / "Evaluation_b.png")
    print("wrote", args.out_dir / "dg_mamba_embeddings_tsne.csv")
    print("wrote", args.out_dir / "dcd_consistency_embeddings_tsne.csv")


if __name__ == "__main__":
    main()
