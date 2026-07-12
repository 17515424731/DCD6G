from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_col(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def _split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == "auto":
        return line.replace(",", " ").split()
    return next(csv.reader([line], delimiter=delimiter))


def read_edges(
    path: Path,
    *,
    src_col: int,
    dst_col: int,
    weight_col: int | None,
    time_col: int | None,
    delimiter: str,
    one_indexed: bool,
    skip_prefixes: tuple[str, ...],
) -> tuple[list[tuple[int, int, float, float]], int]:
    edges: list[tuple[int, int, float, float]] = []
    node_map: dict[int, int] = {}

    def map_node(raw: int) -> int:
        raw = raw - 1 if one_indexed else raw
        if raw not in node_map:
            node_map[raw] = len(node_map)
        return node_map[raw]

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for row_id, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line or any(line.startswith(prefix) for prefix in skip_prefixes):
                continue
            parts = _split_row(line, delimiter)
            try:
                src = map_node(int(float(parts[src_col])))
                dst = map_node(int(float(parts[dst_col])))
                weight = 1.0 if weight_col is None else float(parts[weight_col])
                ts = float(row_id) if time_col is None else float(parts[time_col])
            except (IndexError, ValueError):
                continue
            edges.append((src, dst, weight, ts))
    return edges, len(node_map)


def dynamic_features(
    snapshots: list[list[tuple[int, int, float]]], n_nodes: int
) -> np.ndarray:
    features = np.zeros((len(snapshots), n_nodes, 16), dtype=np.float32)
    cumulative_out_w = np.zeros(n_nodes, dtype=np.float32)
    cumulative_in_w = np.zeros(n_nodes, dtype=np.float32)
    cumulative_out_d = np.zeros(n_nodes, dtype=np.float32)
    cumulative_in_d = np.zeros(n_nodes, dtype=np.float32)

    for t, edges in enumerate(snapshots):
        out_w = np.zeros(n_nodes, dtype=np.float32)
        in_w = np.zeros(n_nodes, dtype=np.float32)
        out_d = np.zeros(n_nodes, dtype=np.float32)
        in_d = np.zeros(n_nodes, dtype=np.float32)
        successors: dict[int, list[int]] = defaultdict(list)
        predecessors: dict[int, list[int]] = defaultdict(list)

        for src, dst, weight in edges:
            out_w[src] += weight
            in_w[dst] += weight
            out_d[src] += 1.0
            in_d[dst] += 1.0
            successors[src].append(dst)
            predecessors[dst].append(src)

        features[t, :, 0] = out_w
        features[t, :, 1] = in_w
        features[t, :, 2] = out_d
        features[t, :, 3] = in_d
        features[t, :, 4] = cumulative_out_w
        features[t, :, 5] = cumulative_in_w
        features[t, :, 6] = cumulative_out_d
        features[t, :, 7] = cumulative_in_d
        for node in range(n_nodes):
            succ = successors.get(node, [])
            pred = predecessors.get(node, [])
            if succ:
                features[t, node, 8] = float(np.mean(out_w[succ]))
                features[t, node, 9] = float(np.mean(in_w[succ]))
                features[t, node, 12] = float(np.mean(out_d[succ]))
                features[t, node, 13] = float(np.mean(in_d[succ]))
            if pred:
                features[t, node, 10] = float(np.mean(out_w[pred]))
                features[t, node, 11] = float(np.mean(in_w[pred]))
                features[t, node, 14] = float(np.mean(out_d[pred]))
                features[t, node, 15] = float(np.mean(in_d[pred]))

        cumulative_out_w += out_w
        cumulative_in_w += in_w
        cumulative_out_d += out_d
        cumulative_in_d += in_d

    return features


def write_tdap_dataset(
    edges: list[tuple[int, int, float, float]],
    n_nodes: int,
    *,
    out_dir: Path,
    num_graphs: int,
    undirected: bool,
    dyn_feats: bool,
    bucket_strategy: str,
) -> None:
    import scipy.sparse as sp

    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[tuple[int, int, float]]] = [[] for _ in range(num_graphs)]
    if bucket_strategy == "equal_edges":
        sorted_edges = sorted(edges, key=lambda edge: edge[3])
        for edge_id, (src, dst, weight, _) in enumerate(sorted_edges):
            bucket = min(edge_id * num_graphs // len(sorted_edges), num_graphs - 1)
            buckets[bucket].append((src, dst, weight))
            if undirected and src != dst:
                buckets[bucket].append((dst, src, weight))
    else:
        min_ts = min(edge[3] for edge in edges)
        max_ts = max(edge[3] for edge in edges)
        span = max(max_ts - min_ts, 1.0)
        for src, dst, weight, ts in edges:
            bucket = min(int(math.floor((ts - min_ts) / span * num_graphs)), num_graphs - 1)
            buckets[bucket].append((src, dst, weight))
            if undirected and src != dst:
                buckets[bucket].append((dst, src, weight))

    adjs = []
    for bucket in buckets:
        rows = [src for src, _, _ in bucket]
        cols = [dst for _, dst, _ in bucket]
        data = [weight for _, _, weight in bucket]
        adjs.append(sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)))

    with (out_dir / f"graphs_{num_graphs}.pkl").open("wb") as handle:
        pickle.dump(adjs, handle)
    np.save(out_dir / "features.npy", np.eye(n_nodes, dtype=np.float32))
    if dyn_feats:
        np.save(out_dir / "dyn_features.npy", dynamic_features(buckets, n_nodes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a raw edge list to TDAP snapshots.")
    parser.add_argument("--raw-edge-list", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-graphs", type=int, default=20)
    parser.add_argument("--out-root", type=Path, default=Path("third_party/TDAP/data"))
    parser.add_argument("--src-col", type=int, default=0)
    parser.add_argument("--dst-col", type=int, default=1)
    parser.add_argument("--weight-col", default="none")
    parser.add_argument("--time-col", default="none")
    parser.add_argument("--delimiter", default="auto")
    parser.add_argument("--one-indexed", default="false", choices=["true", "false"])
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--dyn-feats", action="store_true")
    parser.add_argument("--bucket-strategy", default="equal_time", choices=["equal_time", "equal_edges"])
    parser.add_argument("--skip-prefix", action="append", default=["#", "%"])
    args = parser.parse_args()

    edges, n_nodes = read_edges(
        args.raw_edge_list,
        src_col=args.src_col,
        dst_col=args.dst_col,
        weight_col=_parse_col(args.weight_col),
        time_col=_parse_col(args.time_col),
        delimiter=args.delimiter,
        one_indexed=args.one_indexed == "true",
        skip_prefixes=tuple(args.skip_prefix),
    )
    if not edges:
        raise SystemExit("No edges were parsed from the raw edge list.")
    write_tdap_dataset(
        edges,
        n_nodes,
        out_dir=args.out_root / args.dataset,
        num_graphs=args.num_graphs,
        undirected=args.undirected,
        dyn_feats=args.dyn_feats,
        bucket_strategy=args.bucket_strategy,
    )
    print(
        f"Wrote {args.dataset}: {len(edges)} edges, {n_nodes} nodes, "
        f"{args.num_graphs} snapshots to {(args.out_root / args.dataset).resolve()}"
    )


if __name__ == "__main__":
    main()

