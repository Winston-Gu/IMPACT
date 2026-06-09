#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class BinResult:
    bin_label: str
    bin_lo: float
    bin_hi: float
    samples: List[int]
    successes: int
    total: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total > 0 else 0.0


def _parse_bin_label(label: str) -> Tuple[float, float]:
    lo_str, hi_str = label.split("-", 1)
    return float(lo_str), float(hi_str)


def _read_successes(csv_path: Path) -> Tuple[List[int], int, int]:
    samples: List[int] = []
    successes = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "success" not in reader.fieldnames:
            raise ValueError(f"Missing 'success' column in {csv_path}")
        for row in reader:
            value = int(row["success"])
            samples.append(value)
            successes += value
    total = len(samples)
    return samples, successes, total


def _collect_results(root: Path) -> Dict[str, List[BinResult]]:
    results: Dict[str, List[BinResult]] = {}
    for csv_path in root.rglob("benchmark.csv"):
        bin_dir = csv_path.parent
        combo_dir = bin_dir.parent
        bin_label = bin_dir.name
        combo_label = combo_dir.name
        try:
            lo, hi = _parse_bin_label(bin_label)
        except ValueError as exc:
            raise ValueError(f"Invalid bin label '{bin_label}' in {bin_dir}") from exc
        samples, successes, total = _read_successes(csv_path)
        results.setdefault(combo_label, []).append(
            BinResult(
                bin_label=bin_label,
                bin_lo=lo,
                bin_hi=hi,
                samples=samples,
                successes=successes,
                total=total,
            )
        )
    for combo_label in results:
        results[combo_label].sort(key=lambda r: r.bin_lo)
    return results


def _print_table(results: Dict[str, List[BinResult]]) -> None:
    combos = sorted(results.keys())
    bins = sorted(
        {r.bin_label for combo in results.values() for r in combo},
        key=lambda b: _parse_bin_label(b)[0],
    )
    header = ["mass_bin", *combos]
    widths = [max(len(h), 8) for h in header]
    for i, combo in enumerate(combos, start=1):
        widths[i] = max(widths[i], len(combo))
    for i, bin_label in enumerate(bins):
        widths[0] = max(widths[0], len(bin_label))
        for j, combo in enumerate(combos, start=1):
            rate = next(
                (r.success_rate for r in results[combo] if r.bin_label == bin_label),
                None,
            )
            text = "n/a" if rate is None else f"{rate:.3f}"
            widths[j] = max(widths[j], len(text))

    def fmt_row(values: List[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt_row(header))
    print(fmt_row(["-" * w for w in widths]))
    for bin_label in bins:
        row = [bin_label]
        for combo in combos:
            rate = next(
                (r.success_rate for r in results[combo] if r.bin_label == bin_label),
                None,
            )
            row.append("n/a" if rate is None else f"{rate:.3f}")
        print(fmt_row(row))


def _write_csv(results: Dict[str, List[BinResult]], output_path: Path) -> None:
    combos = sorted(results.keys())
    bins = sorted(
        {r.bin_label for combo in results.values() for r in combo},
        key=lambda b: _parse_bin_label(b)[0],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mass_bin", *combos])
        for bin_label in bins:
            row = [bin_label]
            for combo in combos:
                rate = next(
                    (
                        r.success_rate
                        for r in results[combo]
                        if r.bin_label == bin_label
                    ),
                    None,
                )
                row.append("" if rate is None else f"{rate:.6f}")
            writer.writerow(row)


def _write_plot(results: Dict[str, List[BinResult]], output_path: Path) -> None:
    combos = sorted(results.keys())
    bins = sorted(
        {r.bin_label for combo in results.values() for r in combo},
        key=lambda b: _parse_bin_label(b)[0],
    )
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    x = list(range(len(bins)))
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]
    y_positions = [i * 2 for i in range(len(combos))]
    for y_idx, combo in enumerate(combos):
        z = []
        for bin_label in bins:
            result = next(
                (r for r in results[combo] if r.bin_label == bin_label),
                None,
            )
            if result is None:
                z.append(0.0)
                continue
            z.append(result.success_rate)
        verts = [(x[0], 0.0), *list(zip(x, z)), (x[-1], 0.0)]
        face_color = colors[y_idx % len(colors)]
        poly = PolyCollection(
            [verts],
            facecolors=[face_color],
            edgecolors="none",
            linewidths=0.3,
            alpha=0.2,
        )
        ax.add_collection3d(poly, zs=y_positions[y_idx], zdir="y")
        ax.plot(
            x,
            [y_positions[y_idx]] * len(x),
            z,
            "-o",
            color=face_color,
            markersize=4,
            linewidth=3.0,
            label=combo,
        )
        if z:
            peak_idx = max(range(len(z)), key=lambda i: z[i])
            last_idx = len(z) - 1
            for idx in {peak_idx, last_idx}:
                ax.text(
                    x[idx],
                    y_positions[y_idx],
                    z[idx] + 0.03,
                    f"{z[idx]:.2f}",
                    fontsize=9,
                    color=face_color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(combos)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("mass bin")
    ax.set_ylabel("controller")
    ax.set_zlabel("success rate")
    ax.set_box_aspect((len(bins), len(combos), 3))
    ax.view_init(elev=25, azim=-70)
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark success rates per mass bin."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Benchmark run directory, e.g. logs/benchmark/MassSweep/20260126_191423",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the summary table.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    results = _collect_results(root)
    if not results:
        raise SystemExit(f"No benchmark.csv files found under: {root}")

    _print_table(results)
    results_dir = root / "results"
    default_csv = results_dir / "summary.csv"
    _write_csv(results, default_csv)
    if args.csv is not None:
        _write_csv(results, args.csv)
    plot_path = results_dir / "success_rate.png"
    _write_plot(results, plot_path)


if __name__ == "__main__":
    main()
