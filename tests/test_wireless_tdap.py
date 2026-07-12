from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from dcd6g.london_traces import LondonTraceConfig, london_routes_to_vehicle_traces
from dcd6g.wireless import WirelessGraphConfig, build_disac_dynamic_graph, path_loss_db, shannon_rate_mbps


class WirelessDatasetTests(unittest.TestCase):
    def test_path_loss_protects_zero_distance(self) -> None:
        values = path_loss_db(np.array([0.0, 1.0, 10.0]), 28.0)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_build_disac_dynamic_graph_from_vehicle_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vehicle_traces.csv"
            path.write_text(
                "timestamp,vehicle_id,x,y\n"
                "0,a,0,0\n"
                "0,b,10,0\n"
                "0,c,400,0\n"
                "1,a,0,0\n"
                "1,b,20,0\n",
                encoding="utf-8",
            )
            graph = build_disac_dynamic_graph(
                path,
                config=WirelessGraphConfig(
                    max_distance_m=250.0,
                    shadow_sigma_db=0.0,
                    target_rate_mbps=900.0,
                ),
                seed=1,
            )
            self.assertGreater(len(graph["src"]), 0)
            self.assertIn("S_distance", graph)
            self.assertIn("X_snr", graph)
            self.assertIn("E_shadowing", graph)
            self.assertIn("rate_mbps", graph)
            self.assertIn("target_rate_mbps", graph)
            self.assertEqual(graph["node_features"].shape[0], 2)
            np.testing.assert_allclose(
                graph["rate_mbps"],
                shannon_rate_mbps(graph["X_snr"], WirelessGraphConfig().bandwidth_hz),
                rtol=1e-5,
            )
            self.assertTrue(np.all(graph["y"] == (graph["rate_mbps"] >= 900.0).astype(np.int64)))
            self.assertEqual(float(graph["target_rate_mbps"][0]), 900.0)

    def test_shannon_rate_increases_with_snr(self) -> None:
        rates = shannon_rate_mbps(np.array([0.0, 10.0, 20.0]), 100e6)
        self.assertTrue(np.all(np.diff(rates) > 0.0))

    def test_london_routes_convert_to_vehicle_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "LondonTrajectoriesDataset.csv"
            source.write_text(
                "ID,setID,coordinates\n"
                "1,42,-0.100000:51.500000;-0.100100:51.500100;-0.100200:51.500200\n"
                "2,42,-0.100010:51.500010;-0.100110:51.500110;-0.100210:51.500210\n"
                "3,99,-0.200000:51.600000;-0.200100:51.600100\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "vehicle_traces.csv"
            info = london_routes_to_vehicle_traces(
                source,
                out,
                config=LondonTraceConfig(num_steps=4, max_routes=2),
            )
            self.assertEqual(info["set_id"], "42")
            self.assertEqual(info["rows"], 8)

            graph = build_disac_dynamic_graph(
                out,
                config=WirelessGraphConfig(max_distance_m=250.0, shadow_sigma_db=0.0),
            )
            self.assertGreater(len(graph["src"]), 0)


class TDPGDTests(unittest.TestCase):
    def test_projection_respects_budget(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local environment")

        from dcd6g.td_pgd import TD_PGD_Attacker, TDPGDConfig

        attacker = TD_PGD_Attacker(config=TDPGDConfig())
        projected = attacker.project(torch.tensor([0.7, 0.8, 0.9]), budget=1.0)
        self.assertLessEqual(float(projected.sum()), 1.0001)
        self.assertTrue(torch.all(projected >= 0.0))
        self.assertTrue(torch.all(projected <= 1.0))


class TrainedModelTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_all_models_produce_learned_embeddings(self) -> None:
        import torch

        from dcd6g.trained_models import create_model

        sequence = torch.randn(4, 5, 6)
        for name in ("Vanilla TGNN", "DG-Mamba", "DCD"):
            model = create_model(name, input_dim=6, hidden_dim=8)
            logits, embedding = model(sequence, return_embedding=True)
            self.assertEqual(tuple(logits.shape), (4,))
            self.assertEqual(embedding.shape[0], 4)
            self.assertTrue(embedding.requires_grad)


if __name__ == "__main__":
    unittest.main()
