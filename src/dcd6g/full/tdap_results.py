from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def parse_array(text: str) -> list[float]:
    text = text.strip()
    text = re.sub(r"tensor\((.*?)\)", r"\1", text)
    text = text.replace("array(", "").replace(")", "")
    try:
        value = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    out: list[float] = []

    def flatten(obj: object) -> None:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                flatten(item)
        elif isinstance(obj, (int, float)):
            out.append(float(obj))

    flatten(value)
    return out


def parse_tdap_result(path: Path, tdap_root: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    row: dict[str, object] = {
        "path": str(path),
        "model": "",
        "dataset": "",
        "method": "",
        "constraint": "",
        "epsilon": "",
        "seed": "",
        "orig_metric": "",
        "attacked_metric": "",
        "metric_name": "",
        "time_taken": "",
        "num_targets": 0,
        "avg_perturbations": "",
        "avg_k": "",
        "avg_e": "",
        "avg_dz_frac": "",
        "avg_del_dz": "",
        "f1_from_probs": "",
    }
    rel = path.relative_to(tdap_root)
    parts = rel.parts
    if len(parts) >= 5:
        row["model"] = parts[0].replace("results_", "")
        row["dataset"] = parts[1]
        row["method"] = parts[3]
        row["constraint"] = parts[4]

    fname = path.name
    eps_match = re.search(r"_e(" + FLOAT + r")_", fname)
    seed_match = re.search(r"_seed(\d+)", fname)
    if eps_match:
        row["epsilon"] = eps_match.group(1)
    if seed_match:
        row["seed"] = seed_match.group(1)

    orig = re.search(r"Orig (AUCROC|Accuracy): (" + FLOAT + r")", text)
    attacked = re.search(r"(AUCROC|Accuracy) after perturbation: (" + FLOAT + r")", text)
    if orig:
        row["metric_name"] = orig.group(1)
        row["orig_metric"] = orig.group(2)
    if attacked:
        row["metric_name"] = attacked.group(1)
        row["attacked_metric"] = attacked.group(2)
    time_match = re.search(r"Total time taken:\s*(" + FLOAT + r")", text)
    if time_match:
        row["time_taken"] = time_match.group(1)

    perturb_counts: list[float] = []
    for match in re.finditer(r"Perturbation size:\s*(\d+)", text):
        perturb_counts.append(float(match.group(1)))
    for match in re.finditer(r"Perturbations:\s*(.*?)\n", text):
        payload = match.group(1)
        perturb_counts.append(float(payload.count("tensor(")))

    values = {"k": [], "e": [], "dz_frac": [], "del_dz": []}
    pattern = (
        r"dz':\s*(" + FLOAT + r"),\s*dz'/dz:\s*(" + FLOAT + r"),\s*K:\s*("
        + FLOAT + r"),\s*E:\s*(" + FLOAT + r"),\s*dz'-dz:\s*(" + FLOAT + r")"
    )
    for match in re.finditer(pattern, text):
        values["dz_frac"].append(float(match.group(2)))
        values["k"].append(float(match.group(3)))
        values["e"].append(float(match.group(4)))
        values["del_dz"].append(float(match.group(5)))

    row["num_targets"] = max(len(values["k"]), len(perturb_counts))
    if perturb_counts:
        row["avg_perturbations"] = sum(perturb_counts) / len(perturb_counts)
    for key, vals in values.items():
        if vals:
            row[f"avg_{key}"] = sum(vals) / len(vals)

    labels_match = re.search(r"Target Labels:\s*(.*?)\n", text)
    attacked_probs_match = re.search(r"Attacked Probs:\s*(.*?)\n", text, re.S)
    if labels_match and attacked_probs_match:
        labels = [int(round(x)) for x in parse_array(labels_match.group(1))]
        probs = parse_array(attacked_probs_match.group(1))
        if labels and len(labels) == len(probs):
            preds = [1 if prob >= 0.5 else 0 for prob in probs]
            tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
            fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
            fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
            denom = 2 * tp + fp + fn
            row["f1_from_probs"] = "" if denom == 0 else (2 * tp) / denom
            row["metric_name"] = "F1"
    return row


def collect_results(tdap_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in tdap_root.glob("results_*/*/multi_targets/*/*/*.txt"):
        rows.append(parse_tdap_result(path, tdap_root))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse TDAP result text files.")
    parser.add_argument("--tdap-root", type=Path, default=Path("third_party/TDAP"))
    parser.add_argument("--out", type=Path, default=Path("outputs_full/tdap_results.csv"))
    args = parser.parse_args()

    rows = collect_results(args.tdap_root)
    write_csv(args.out, rows)
    print(f"Parsed {len(rows)} TDAP result files into {args.out.resolve()}")


if __name__ == "__main__":
    main()
