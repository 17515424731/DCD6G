from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WirelessGraphConfig:
    max_distance_m: float = 250.0
    carrier_frequency_ghz: float = 28.0
    transmit_power_dbm: float = 23.0
    noise_density_dbm_hz: float = -174.0
    bandwidth_hz: float = 100e6
    shadow_sigma_db: float = 3.0
    min_distance_m: float = 1.0
    target_rate_mbps: float = 700.0


def path_loss_db(distance_m: np.ndarray, carrier_frequency_ghz: float) -> np.ndarray:
    """3GPP TR 38.901 path-loss model with a guard against log10(0)."""
    safe_distance = np.maximum(distance_m.astype(float), 1e-6)
    return 28.0 + 22.0 * np.log10(safe_distance) + 20.0 * math.log10(carrier_frequency_ghz)


def thermal_noise_power_dbm(noise_density_dbm_hz: float, bandwidth_hz: float) -> float:
    return noise_density_dbm_hz + 10.0 * math.log10(bandwidth_hz)


def snr_db(distance_m: np.ndarray, config: WirelessGraphConfig) -> np.ndarray:
    distance = np.maximum(distance_m.astype(float), config.min_distance_m)
    noise_power = thermal_noise_power_dbm(config.noise_density_dbm_hz, config.bandwidth_hz)
    return config.transmit_power_dbm - path_loss_db(distance, config.carrier_frequency_ghz) - noise_power


def shannon_rate_mbps(snr_db_value: np.ndarray, bandwidth_hz: float) -> np.ndarray:
    snr_linear = np.power(10.0, snr_db_value.astype(float) / 10.0)
    return bandwidth_hz * np.log2(1.0 + snr_linear) / 1e6


def read_vehicle_traces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    vehicle_ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "vehicle_id", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"vehicle trace CSV missing columns: {sorted(missing)}")
        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                vehicle_ids.append(row["vehicle_id"])
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (TypeError, ValueError):
                continue
    if not timestamps:
        raise ValueError("no valid trace rows were parsed")
    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(vehicle_ids, dtype=str),
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def build_disac_dynamic_graph(
    trace_csv: Path,
    *,
    config: WirelessGraphConfig = WirelessGraphConfig(),
    seed: int = 123,
) -> dict[str, np.ndarray]:
    timestamps, raw_vehicle_ids, xs, ys = read_vehicle_traces(trace_csv)
    unique_times = np.asarray(sorted(set(timestamps.tolist())), dtype=float)
    vehicle_vocab = {vehicle_id: idx for idx, vehicle_id in enumerate(sorted(set(raw_vehicle_ids.tolist())))}
    rng = np.random.default_rng(seed)

    src_all: list[int] = []
    dst_all: list[int] = []
    edge_attr_all: list[float] = []
    distance_all: list[float] = []
    path_loss_all: list[float] = []
    shadowing_all: list[float] = []
    rate_all: list[float] = []
    edge_time_all: list[int] = []
    edge_label_all: list[int] = []
    node_features = np.zeros((len(unique_times), len(vehicle_vocab), 4), dtype=np.float32)

    for t_idx, timestamp in enumerate(unique_times):
        mask = timestamps == timestamp
        ids = raw_vehicle_ids[mask]
        coords = np.column_stack([xs[mask], ys[mask]]).astype(np.float32)
        node_ids = np.asarray([vehicle_vocab[vehicle_id] for vehicle_id in ids], dtype=int)
        if len(node_ids) == 0:
            continue
        node_features[t_idx, node_ids, 0] = coords[:, 0]
        node_features[t_idx, node_ids, 1] = coords[:, 1]

        edge_snrs: list[float] = []
        edge_pairs: list[tuple[int, int]] = []
        edge_distances: list[float] = []
        edge_path_losses: list[float] = []
        edge_shadowing: list[float] = []
        for local_i in range(len(node_ids)):
            diffs = coords - coords[local_i]
            distances = np.sqrt((diffs * diffs).sum(axis=1))
            valid = (distances <= config.max_distance_m) & (distances > 0.0)
            for local_j in np.where(valid)[0]:
                # X is the link observation derived from distance, path loss, and SNR.
                distance = float(distances[local_j])
                pl = float(path_loss_db(np.asarray([distance]), config.carrier_frequency_ghz)[0])
                base_snr = snr_db(np.asarray([distance]), config)[0]
                shadow = float(rng.normal(0.0, config.shadow_sigma_db))
                faded_snr = base_snr + shadow
                edge_pairs.append((int(node_ids[local_i]), int(node_ids[local_j])))
                edge_snrs.append(float(faded_snr))
                edge_distances.append(distance)
                edge_path_losses.append(pl)
                edge_shadowing.append(shadow)

        if not edge_pairs:
            continue
        degree = np.zeros(len(vehicle_vocab), dtype=np.float32)
        snr_sum = np.zeros(len(vehicle_vocab), dtype=np.float32)
        for idx, ((src, dst), value) in enumerate(zip(edge_pairs, edge_snrs)):
            rate_mbps = float(shannon_rate_mbps(np.asarray([value]), config.bandwidth_hz)[0])
            src_all.append(src)
            dst_all.append(dst)
            edge_attr_all.append(value)
            distance_all.append(edge_distances[idx])
            path_loss_all.append(edge_path_losses[idx])
            shadowing_all.append(edge_shadowing[idx])
            rate_all.append(rate_mbps)
            edge_time_all.append(t_idx)
            edge_label_all.append(int(rate_mbps >= config.target_rate_mbps))
            degree[src] += 1.0
            snr_sum[src] += value

        active = degree > 0
        node_features[t_idx, :, 2] = degree
        node_features[t_idx, active, 3] = snr_sum[active] / degree[active]

    return {
        "src": np.asarray(src_all, dtype=np.int64),
        "dst": np.asarray(dst_all, dtype=np.int64),
        "edge_attr": np.asarray(edge_attr_all, dtype=np.float32),
        "S_distance": np.asarray(distance_all, dtype=np.float32),
        "X_snr": np.asarray(edge_attr_all, dtype=np.float32),
        "E_path_loss": np.asarray(path_loss_all, dtype=np.float32),
        "E_shadowing": np.asarray(shadowing_all, dtype=np.float32),
        "rate_mbps": np.asarray(rate_all, dtype=np.float32),
        "target_rate_mbps": np.asarray([config.target_rate_mbps], dtype=np.float32),
        "t": np.asarray(edge_time_all, dtype=np.int64),
        "y": np.asarray(edge_label_all, dtype=np.int64),
        "node_features": node_features,
        "timestamps": unique_times,
        "vehicle_ids": np.asarray(sorted(vehicle_vocab, key=vehicle_vocab.get),dtype=str),
    }


def save_npz(graph: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 6G-DISAC wireless dynamic graph from vehicle traces.")
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-distance-m", type=float, default=250.0)
    parser.add_argument("--target-rate-mbps", type=float, default=700.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    config = WirelessGraphConfig(
        max_distance_m=args.max_distance_m,
        target_rate_mbps=args.target_rate_mbps,
    )
    graph = build_disac_dynamic_graph(args.trace_csv, config=config, seed=args.seed)
    save_npz(graph, args.out)
    print(
        f"wrote {args.out}: {len(graph['src'])} directed edges, "
        f"{len(graph['vehicle_ids'])} vehicles, {len(graph['timestamps'])} timestamps"
    )


if __name__ == "__main__":
    main()
