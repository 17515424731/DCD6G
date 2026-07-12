from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from dcd6g.trained_models import binary_f1


REQUIRED_COLUMNS = {"dataset", "epsilon", "y_true", "y_pred"}


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def aggregate_method(method: str, path: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_predictions(path):
        groups[(row["dataset"], row["epsilon"])].append((int(row["y_true"]), int(row["y_pred"])))
    output = []
    for (dataset, epsilon), values in sorted(groups.items()):
        y_true = np.asarray([value[0] for value in values], dtype=np.int64)
        y_pred = np.asarray([value[1] for value in values], dtype=np.int64)
        output.append(
            {
                "dataset": dataset,
                "epsilon": epsilon,
                "method": method,
                "n": len(values),
                "f1": binary_f1(y_true, y_pred),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate real per-sample CASR, GRACIE, and DCD predictions; no scores are synthesized"
    )
    parser.add_argument("--casr-predictions", type=Path, required=True)
    parser.add_argument("--gracie-predictions", type=Path, required=True)
    parser.add_argument("--dcd-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    rows.extend(aggregate_method("CASR", args.casr_predictions))
    rows.extend(aggregate_method("GRACIE", args.gracie_predictions))
    rows.extend(aggregate_method("DCD", args.dcd_predictions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "epsilon", "method", "n", "f1"))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
