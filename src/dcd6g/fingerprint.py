from __future__ import annotations

import numpy as np


def structural_hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing directed edges in two binary adjacency matrices."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return int(np.count_nonzero(a.astype(bool) != b.astype(bool)))


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return float(np.linalg.norm(a - b, ord="fro"))


def causal_fingerprint_distance(
    adjacency: np.ndarray,
    baseline_adjacency: np.ndarray,
    weights: np.ndarray,
    baseline_weights: np.ndarray,
    *,
    shd_weight: float = 0.55,
    fro_weight: float = 0.45,
) -> float:
    """Equation from the supplement: alpha * SHD + beta * ||W-W_base||_F."""
    shd = structural_hamming_distance(adjacency, baseline_adjacency)
    fro = frobenius_distance(weights, baseline_weights)
    return shd_weight * shd + fro_weight * fro


def update_dynamic_baseline(
    previous_baseline: np.ndarray, current_weights: np.ndarray, learning_rate: float
) -> np.ndarray:
    """First-order baseline filter B_t=(1-alpha)B_{t-1}+alpha W_t."""
    if not 0.0 < learning_rate < 1.0:
        raise ValueError("learning_rate must be inside (0, 1)")
    return (1.0 - learning_rate) * previous_baseline + learning_rate * current_weights


def within_three_sigma(
    weights: np.ndarray, baseline: np.ndarray, sigma: np.ndarray, *, multiplier: float = 3.0
) -> bool:
    envelope = multiplier * np.maximum(sigma, 1e-8)
    return bool(np.all(np.abs(weights - baseline) <= envelope))

