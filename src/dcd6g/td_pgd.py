from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np


LossFn = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class TDPGDConfig:
    num_steps: int = 20
    learning_rate: float = 0.1
    num_rounds: int = 20
    xi: float = 1e-5
    device: str = "cpu"


class TD_PGD_Attacker:
    """TDAP/TD-PGD attacker with projection and ROUND from the KDD 2023 recipe.

    The attacker is model-agnostic. Pass a differentiable `loss_fn` that receives
    a state dictionary and returns a torch scalar loss. The same attacker can
    therefore be connected to Vanilla TGNN, DG-Mamba, or DCD.
    """

    def __init__(self, loss_fn: LossFn | None = None, config: TDPGDConfig | None = None):
        self.loss_fn = loss_fn
        self.config = config or TDPGDConfig()

    @staticmethod
    def _torch():
        import torch

        return torch

    def project(self, values: Any, budget: float) -> Any:
        """Project s onto {s in [0,1]^m, 1^T s <= budget} using bisection."""
        torch = self._torch()
        budget = max(float(budget), 0.0)
        clipped = values.clamp(0.0, 1.0)
        if clipped.sum() <= budget:
            return clipped
        low = torch.min(values - 1.0)
        high = torch.max(values)
        for _ in range(80):
            mid = (low + high) / 2.0
            projected = (values - mid).clamp(0.0, 1.0)
            total = projected.sum()
            if torch.abs(total - budget) <= self.config.xi:
                return projected
            if total > budget:
                low = mid
            else:
                high = mid
        return (values - high).clamp(0.0, 1.0)

    @staticmethod
    def _dense_adjacencies(data: dict[str, np.ndarray]) -> np.ndarray:
        src = np.asarray(data["src"], dtype=np.int64)
        dst = np.asarray(data["dst"], dtype=np.int64)
        times = np.asarray(data["t"], dtype=np.int64)
        num_t = int(times.max()) + 1 if len(times) else 0
        num_nodes = int(max(src.max(initial=0), dst.max(initial=0))) + 1 if len(src) else 0
        adjs = np.zeros((num_t, num_nodes, num_nodes), dtype=np.float32)
        for s, d, t in zip(src, dst, times):
            adjs[t, s, d] = 1.0
        return adjs

    @staticmethod
    def _candidate_pairs(adjacency: np.ndarray) -> np.ndarray:
        active = np.where((adjacency.sum(axis=0) + adjacency.sum(axis=1)) != 0)[0]
        if len(active) < 2:
            active = np.arange(adjacency.shape[0])
        pairs = [(int(i), int(j)) for i in active for j in active if i != j]
        return np.asarray(pairs, dtype=np.int64)

    @staticmethod
    def _apply_relaxed_flips_torch(base_adj: Any, pairs: np.ndarray, scores: Any) -> Any:
        torch = TD_PGD_Attacker._torch()
        adv = base_adj.clone()
        if len(pairs) == 0:
            return adv
        rows = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=base_adj.device)
        cols = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=base_adj.device)
        current = adv[rows, cols]
        replacement_weight = torch.median(base_adj[base_adj > 0]) if torch.any(base_adj > 0) else torch.tensor(1.0, device=base_adj.device)
        flipped = torch.where(current > 0, current * (1.0 - scores), replacement_weight * scores)
        adv = adv.clone()
        adv[rows, cols] = flipped
        return adv

    def _default_loss(self, state: dict[str, Any]) -> Any:
        # Fallback objective for pipeline tests: maximize the total perturbation
        # energy. Real experiments should pass a cross-entropy model loss.
        return state["scores"].sum()

    def _round(self, scores: Any, budget: float, state_factory: Callable[[Any], dict[str, Any]]) -> Any:
        torch = self._torch()
        k = int(max(0, min(scores.numel(), np.floor(budget))))
        best = torch.zeros_like(scores)
        if k > 0:
            top_idx = torch.topk(scores, k).indices
            best[top_idx] = 1.0
        best_loss = self._loss_for_mask(best, state_factory)

        probabilities = scores.detach().clamp(0.0, 1.0)
        for _ in range(max(self.config.num_rounds, 0)):
            sample = torch.bernoulli(probabilities)
            if sample.sum() > budget:
                active = torch.where(sample > 0)[0]
                keep = active[torch.topk(probabilities[active], k).indices] if k > 0 else active[:0]
                sample = torch.zeros_like(sample)
                sample[keep] = 1.0
            sample_loss = self._loss_for_mask(sample, state_factory)
            if sample_loss > best_loss:
                best = sample
                best_loss = sample_loss
        return best

    def _loss_for_mask(self, mask: Any, state_factory: Callable[[Any], dict[str, Any]]) -> float:
        with self._torch().no_grad():
            loss = (self.loss_fn or self._default_loss)(state_factory(mask))
        return float(loss.detach().cpu().item())

    def attack(
        self,
        pyg_temporal_data: dict[str, np.ndarray],
        epsilon: float,
        *,
        target_time: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Attack a temporal graph stored as src/dst/edge_attr/t/y numpy arrays."""
        torch = self._torch()
        adjs_np = self._dense_adjacencies(pyg_temporal_data)
        if adjs_np.shape[0] < 2:
            raise ValueError("TD-PGD requires at least two temporal snapshots")
        target_time = adjs_np.shape[0] - 1 if target_time is None else target_time
        if target_time <= 0 or target_time >= adjs_np.shape[0]:
            raise ValueError("target_time must be in [1, num_snapshots - 1]")

        previous = adjs_np[target_time - 1]
        current = adjs_np[target_time]
        d_a_t = float(np.abs(current - previous).sum())
        budget = float(epsilon) * d_a_t
        pairs = self._candidate_pairs(current)

        device = self.config.device
        base_adj = torch.as_tensor(current, dtype=torch.float32, device=device)
        scores = torch.zeros(len(pairs), dtype=torch.float32, device=device, requires_grad=True)

        def make_state(mask_or_scores: Any) -> dict[str, Any]:
            adv_adj = self._apply_relaxed_flips_torch(base_adj, pairs, mask_or_scores)
            return {
                "adjacency": adv_adj,
                "base_adjacency": base_adj,
                "pairs": pairs,
                "scores": mask_or_scores,
                "target_time": target_time,
                "labels": pyg_temporal_data.get("y"),
            }

        for _ in range(self.config.num_steps):
            loss = (self.loss_fn or self._default_loss)(make_state(scores))
            if scores.grad is not None:
                scores.grad.zero_()
            loss.backward()
            with torch.no_grad():
                grad = torch.zeros_like(scores) if scores.grad is None else scores.grad
                scores += self.config.learning_rate * grad
                scores[:] = self.project(scores, budget)
            scores.requires_grad_(True)

        discrete = self._round(scores.detach(), budget, make_state)
        attacked_adj = make_state(discrete)["adjacency"].detach().cpu().numpy()
        out = dict(pyg_temporal_data)
        rows = pairs[:, 0] if len(pairs) else np.array([], dtype=np.int64)
        cols = pairs[:, 1] if len(pairs) else np.array([], dtype=np.int64)
        active = discrete.detach().cpu().numpy() > 0
        out["attack_src"] = rows[active].astype(np.int64)
        out["attack_dst"] = cols[active].astype(np.int64)
        out["attack_budget"] = np.asarray([budget], dtype=np.float32)
        out["attack_target_time"] = np.asarray([target_time], dtype=np.int64)
        out["attacked_adjacency"] = attacked_adj.astype(np.float32)
        return out


def save_attack_npz(result: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
