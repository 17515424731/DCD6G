from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def summarize(path: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            key = (row.get("dataset", ""), row.get("epsilon", ""), row.get("paper_method", ""))
            groups[key].append(score)

    rows: list[dict[str, object]] = []
    for (dataset, epsilon, method), values in sorted(groups.items()):
        rows.append(
            {
                "dataset": dataset,
                "epsilon": epsilon,
                "paper_method": method,
                "n": len(values),
                "mean_score": mean(values),
                "std_score": pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full reproduction method scores.")
    parser.add_argument("--scores", type=Path, default=Path("outputs_full/paper_method_scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs_full/summary_scores.csv"))
    args = parser.parse_args()

    rows = summarize(args.scores)
    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} summary rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()

