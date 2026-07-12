from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LondonTraceConfig:
    num_steps: int = 120
    max_routes: int = 80
    set_id: str | None = None
    min_points: int = 2


def parse_coordinate_path(value: str) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        lon_text, lat_text = item.split(":", 1)
        try:
            coords.append((float(lon_text), float(lat_text)))
        except ValueError:
            continue
    return coords


def _lonlat_to_xy_m(
    coords: list[tuple[float, float]],
    *,
    lon0: float,
    lat0: float,
) -> np.ndarray:
    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    lon = np.asarray([p[0] for p in coords], dtype=np.float64)
    lat = np.asarray([p[1] for p in coords], dtype=np.float64)
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0
    y = (lat - lat0) * 110_540.0
    return np.column_stack([x, y]).astype(np.float32)


def _resample_path(points: np.ndarray, num_steps: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), num_steps, axis=0)
    old_idx = np.arange(len(points), dtype=np.float32)
    new_idx = np.linspace(0, len(points) - 1, num_steps, dtype=np.float32)
    xs = np.interp(new_idx, old_idx, points[:, 0])
    ys = np.interp(new_idx, old_idx, points[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


def _read_london_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ID", "setID", "coordinates"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"London CSV missing columns: {sorted(missing)}")
        for row in reader:
            if len(parse_coordinate_path(row.get("coordinates", ""))) >= 2:
                rows.append(row)
    if not rows:
        raise ValueError("no usable London trajectory rows found")
    return rows


def choose_route_rows(rows: list[dict[str, str]], config: LondonTraceConfig) -> list[dict[str, str]]:
    if config.set_id is None:
        counts = Counter(row["setID"] for row in rows)
        selected_set = counts.most_common(1)[0][0]
    else:
        selected_set = str(config.set_id)
    selected = [row for row in rows if row["setID"] == selected_set]
    if not selected:
        raise ValueError(f"no rows found for setID={selected_set!r}")
    return selected[: config.max_routes]


def london_routes_to_vehicle_traces(
    input_csv: Path,
    out_csv: Path,
    *,
    config: LondonTraceConfig = LondonTraceConfig(),
) -> dict[str, int | str]:
    rows = choose_route_rows(_read_london_rows(input_csv), config)
    paths = [parse_coordinate_path(row["coordinates"]) for row in rows]
    all_lon = [lon for path in paths for lon, _ in path]
    all_lat = [lat for path in paths for _, lat in path]
    lon0 = float(np.mean(all_lon))
    lat0 = float(np.mean(all_lat))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "vehicle_id", "x", "y"])
        for route_idx, (row, path) in enumerate(zip(rows, paths)):
            points = _lonlat_to_xy_m(path, lon0=lon0, lat0=lat0)
            sampled = _resample_path(points, config.num_steps)
            vehicle_id = f"route_{row.get('ID', route_idx)}"
            for timestamp, (x, y) in enumerate(sampled):
                writer.writerow([timestamp, vehicle_id, f"{float(x):.3f}", f"{float(y):.3f}"])
                row_count += 1

    return {
        "set_id": rows[0]["setID"],
        "routes": len(rows),
        "steps": config.num_steps,
        "rows": row_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LondonTrajectories routes into timestamp,vehicle_id,x,y traces."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--max-routes", type=int, default=80)
    parser.add_argument("--set-id", type=str, default=None)
    args = parser.parse_args()

    info = london_routes_to_vehicle_traces(
        args.input,
        args.out,
        config=LondonTraceConfig(
            num_steps=args.num_steps,
            max_routes=args.max_routes,
            set_id=args.set_id,
        ),
    )
    print(
        f"wrote {args.out}: setID={info['set_id']}, "
        f"routes={info['routes']}, steps={info['steps']}, rows={info['rows']}"
    )


if __name__ == "__main__":
    main()
