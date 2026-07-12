from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

from dcd6g.full.dcd_overlay_metrics import aggregate_method
from dcd6g.full.tdap_commands import attack_command, load_config
from dcd6g.full.tdap_results import parse_tdap_result


class FullPipelineTests(unittest.TestCase):
    def test_attack_command_is_bash_friendly(self) -> None:
        config = load_config(Path("configs/full_tdap_matrix.json"))
        dataset = config["datasets"][0]
        command = attack_command(config, dataset, "DySAT", 123, 0.05)
        self.assertIn("-device ${DEVICE}", command)
        self.assertIn("third_party/TDAP", command)
        self.assertNotIn("third_party\\TDAP", command)

    def test_parse_complete_tdap_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "results_dysat" / "bitcoin_otc" / "multi_targets" / "pgd" / "noise"
            path.mkdir(parents=True)
            result = path / "results_td_tg1_n20_c19t19_e0.05_eb100_l0_seed123.txt"
            result.write_text(
                "\n".join(
                    [
                        "Orig AUCROC: 0.91",
                        "Target_id: 0, Perturbation size: 12",
                        "dz': 0.4, dz'/dz: 1.2, K: 0.3, E: 0.2, dz'-dz: 0.1",
                        "AUCROC after perturbation: 0.84",
                        "Total time taken: 2.5",
                    ]
                ),
                encoding="utf-8",
            )
            row = parse_tdap_result(result, root)
            self.assertEqual(row["dataset"], "bitcoin_otc")
            self.assertEqual(row["epsilon"], "0.05")
            self.assertEqual(row["attacked_metric"], "0.84")
            self.assertEqual(row["avg_perturbations"], 12.0)

    def test_overlay_metrics_require_real_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("dataset", "epsilon", "y_true", "y_pred"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"dataset": "sample", "epsilon": "0.05", "y_true": 1, "y_pred": 1},
                        {"dataset": "sample", "epsilon": "0.05", "y_true": 0, "y_pred": 0},
                    ]
                )
            rows = aggregate_method("DCD", path)
            self.assertEqual(rows[0]["method"], "DCD")
            self.assertEqual(rows[0]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
