from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = ("snr_db", "distance_m", "path_loss_db", "shadowing_db", "rate_mbps", "edge_present")


def _torch():
    import torch

    return torch


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [binary_f1(y_true, probabilities >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


@dataclass
class TemporalEdgeData:
    raw_sequences: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def normalized(self, sequences: np.ndarray | None = None) -> np.ndarray:
        values = self.raw_sequences if sequences is None else sequences
        return ((values - self.mean) / self.std).astype(np.float32)


def _time_masks(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(times)
    if len(unique) < 3:
        raise ValueError("At least three temporal snapshots are required for train/validation/test splits")
    train_count = max(1, int(len(unique) * 0.6))
    val_count = max(1, int(len(unique) * 0.2))
    if train_count + val_count >= len(unique):
        train_count = len(unique) - 2
        val_count = 1
    train_times = unique[:train_count]
    val_times = unique[train_count : train_count + val_count]
    test_times = unique[train_count + val_count :]
    return np.isin(times, train_times), np.isin(times, val_times), np.isin(times, test_times)


def build_temporal_edge_data(graph: dict[str, np.ndarray], context: int = 8) -> TemporalEdgeData:
    required = {
        "src",
        "dst",
        "t",
        "y",
        "X_snr",
        "S_distance",
        "E_path_loss",
        "E_shadowing",
        "rate_mbps",
    }
    missing = sorted(required.difference(graph))
    if missing:
        raise KeyError(f"graph archive is missing fields: {', '.join(missing)}")

    src = graph["src"].astype(np.int64)
    dst = graph["dst"].astype(np.int64)
    times = graph["t"].astype(np.int64)
    labels = graph["y"].astype(np.int64)
    features = np.column_stack(
        [
            graph["X_snr"],
            graph["S_distance"],
            graph["E_path_loss"],
            graph["E_shadowing"],
            graph["rate_mbps"],
            np.ones(len(src), dtype=np.float32),
        ]
    ).astype(np.float32)

    sequences = np.empty((len(src), context, features.shape[1]), dtype=np.float32)
    history: dict[tuple[int, int], list[np.ndarray]] = {}
    order = np.lexsort((dst, src, times))
    for index in order:
        key = (int(src[index]), int(dst[index]))
        previous = history.setdefault(key, [])
        window = (previous + [features[index]])[-context:]
        pad = [window[0]] * (context - len(window))
        sequences[index] = np.asarray(pad + window, dtype=np.float32)
        previous.append(features[index])

    train_mask, val_mask, test_mask = _time_masks(times)
    train_values = sequences[train_mask].reshape(-1, features.shape[1])
    mean = train_values.mean(axis=0, keepdims=True).reshape(1, 1, -1)
    std = train_values.std(axis=0, keepdims=True).reshape(1, 1, -1)
    std = np.where(std < 1e-6, 1.0, std)
    return TemporalEdgeData(
        sequences,
        labels,
        times,
        src,
        dst,
        train_mask,
        val_mask,
        test_mask,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _model_classes():
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class VanillaTGNN(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, sequence: Any, return_embedding: bool = False):
            values = functional.gelu(self.input_projection(sequence))
            hidden, _ = self.temporal(values)
            embedding = hidden[:, -1]
            logits = self.classifier(embedding).squeeze(-1)
            return (logits, embedding) if return_embedding else logits

    class SelectiveStateSpaceBlock(nn.Module):
        def __init__(self, hidden_dim: int, state_dim: int = 16):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.state_dim = state_dim
            self.input_projection = nn.Linear(hidden_dim, hidden_dim * 2)
            self.delta_projection = nn.Linear(hidden_dim, hidden_dim)
            self.b_projection = nn.Linear(hidden_dim, state_dim)
            self.c_projection = nn.Linear(hidden_dim, state_dim)
            self.log_a = nn.Parameter(torch.zeros(hidden_dim, state_dim))
            self.skip = nn.Parameter(torch.ones(hidden_dim))
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, sequence: Any):
            values, gate = self.input_projection(sequence).chunk(2, dim=-1)
            delta = functional.softplus(self.delta_projection(values))
            b_values = self.b_projection(values)
            c_values = self.c_projection(values)
            state = torch.zeros(
                sequence.shape[0], self.hidden_dim, self.state_dim, device=sequence.device, dtype=sequence.dtype
            )
            outputs = []
            a_values = -torch.exp(self.log_a)
            for step in range(sequence.shape[1]):
                dt = delta[:, step].unsqueeze(-1)
                transition = torch.exp(dt * a_values.unsqueeze(0))
                state = transition * state + dt * values[:, step].unsqueeze(-1) * b_values[:, step].unsqueeze(1)
                scanned = (state * c_values[:, step].unsqueeze(1)).sum(dim=-1)
                outputs.append(scanned + self.skip * values[:, step])
            scanned_sequence = torch.stack(outputs, dim=1) * torch.sigmoid(gate)
            return self.norm(sequence + self.output_projection(scanned_sequence))

    class DGMamba(nn.Module):
        """Trainable wireless adaptation of DG-Mamba's selective snapshot scan."""

        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList([SelectiveStateSpaceBlock(hidden_dim), SelectiveStateSpaceBlock(hidden_dim)])
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, sequence: Any, return_embedding: bool = False):
            hidden = functional.gelu(self.input_projection(sequence))
            for block in self.blocks:
                hidden = block(hidden)
            embedding = hidden[:, -1]
            logits = self.classifier(embedding).squeeze(-1)
            return (logits, embedding) if return_embedding else logits

    class DCDModel(nn.Module):
        """Trainable dynamic causal-consistency defense model."""

        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.expected_state = nn.Linear(hidden_dim, hidden_dim)
            self.consistency_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.classifier = nn.Linear(hidden_dim * 2, 1)

        def forward(self, sequence: Any, return_embedding: bool = False):
            values = functional.gelu(self.input_projection(sequence))
            hidden, _ = self.temporal(values)
            current = hidden[:, -1]
            history = hidden[:, -2] if hidden.shape[1] > 1 else hidden[:, -1]
            expected = self.expected_state(history)
            residual = current - expected
            gate = self.consistency_gate(residual.abs())
            defended = gate * expected + (1.0 - gate) * current
            embedding = torch.cat([defended, residual], dim=-1)
            logits = self.classifier(embedding).squeeze(-1)
            return (logits, embedding) if return_embedding else logits

    return {"Vanilla TGNN": VanillaTGNN, "DG-Mamba": DGMamba, "DCD": DCDModel}


def create_model(name: str, input_dim: int, hidden_dim: int = 32):
    classes = _model_classes()
    if name not in classes:
        raise ValueError(f"unknown model: {name}")
    return classes[name](input_dim, hidden_dim)


def train_model(
    name: str,
    data: TemporalEdgeData,
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    device: str,
    checkpoint_path: Path,
):
    torch = _torch()
    model_seed = stable_seed(name, seed)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    model = create_model(name, data.raw_sequences.shape[-1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_x = torch.as_tensor(data.normalized()[data.train_mask], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(data.labels[data.train_mask], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(data.normalized()[data.val_mask], dtype=torch.float32, device=device)
    val_y = data.labels[data.val_mask]
    positive = max(float(train_y.sum().item()), 1.0)
    negative = max(float(train_y.numel() - train_y.sum().item()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    best_state = None
    best_f1 = -1.0
    best_threshold = 0.5
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_x)).cpu().numpy()
        threshold = choose_threshold(val_y, probabilities)
        score = binary_f1(val_y, probabilities >= threshold)
        if score >= best_f1:
            best_f1 = score
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "state_dict": model.state_dict(),
            "input_dim": data.raw_sequences.shape[-1],
            "hidden_dim": hidden_dim,
            "threshold": best_threshold,
            "feature_names": FEATURE_NAMES,
            "mean": data.mean,
            "std": data.std,
        },
        checkpoint_path,
    )
    return model, best_threshold


def predict_model(model: Any, sequences: np.ndarray, threshold: float, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(sequences, dtype=torch.float32, device=device)
        logits, embedding = model(tensor, return_embedding=True)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return (probabilities >= threshold).astype(np.int64), probabilities, embedding.cpu().numpy()


def benign_sequences(data: TemporalEdgeData, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequences = data.raw_sequences.copy()
    shadow_noise = rng.normal(0.0, 1.0, size=sequences.shape[:-1]).astype(np.float32)
    sequences[..., 0] += shadow_noise
    sequences[..., 3] += shadow_noise
    snr_linear = np.power(10.0, sequences[..., 0] / 10.0)
    sequences[..., 4] = 100.0 * np.log2(1.0 + snr_linear)
    return sequences


def graph_dict(archive: Any) -> dict[str, np.ndarray]:
    return {key: archive[key] for key in archive.files}

