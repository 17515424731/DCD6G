from __future__ import annotations

import unittest

import numpy as np

from dcd6g.ctm import TraceIndicators, trace
from dcd6g.fingerprint import (
    causal_fingerprint_distance,
    structural_hamming_distance,
    update_dynamic_baseline,
    within_three_sigma,
)


def base_causal_adjacency() -> np.ndarray:
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int64)


def base_causal_weights() -> np.ndarray:
    return np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]], dtype=float)


class FingerprintTests(unittest.TestCase):
    def test_structural_hamming_distance(self) -> None:
        a = base_causal_adjacency()
        b = a.copy()
        b[0, 2] = 1
        self.assertEqual(structural_hamming_distance(a, b), 1)

    def test_causal_fingerprint_distance_increases(self) -> None:
        a = base_causal_adjacency()
        w = base_causal_weights()
        changed = w.copy()
        changed[1, 2] += 0.5
        self.assertGreater(causal_fingerprint_distance(a, a, changed, w), 0.0)

    def test_baseline_update_and_sigma_rule(self) -> None:
        w = base_causal_weights()
        updated = update_dynamic_baseline(w, w + 1.0, 0.2)
        self.assertTrue(np.allclose(updated, w + 0.2))
        self.assertTrue(within_three_sigma(w + 0.01, w, np.full_like(w, 0.01)))


class CTMTests(unittest.TestCase):
    def test_trace_perception_poisoning(self) -> None:
        result = trace(
            "structural_fracture",
            TraceIndicators(
                semantic_dependency_anomaly=True,
                single_node_feature_anomaly=True,
                collaboration_link_normal=True,
            ),
        )
        self.assertTrue(result.alarm)
        self.assertEqual(result.fracture, "X->Y")

    def test_trace_benign(self) -> None:
        result = trace("benign", TraceIndicators())
        self.assertFalse(result.alarm)


if __name__ == "__main__":
    unittest.main()
