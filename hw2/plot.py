import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "results/results.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/plots")
    args = parser.parse_args()
    runs = defaultdict(list)
    with args.csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            runs[row["impl"]].append(row)
    assert runs, "CSV contains no runs."
    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts = (
        ("vram", "Peak allocated VRAM (MiB)",
         lambda row: float(row["peak_train_allocated_mib"])),
        ("latency", "Forward + backward latency (ms)",
         lambda row: float(row["forward_mean_ms"]) + float(row["backward_mean_ms"])),
    )
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    colors = plt.get_cmap("tab10")
    for name, ylabel, metric in charts:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for index, (impl, rows) in enumerate(runs.items()):
            ax.scatter(
                [int(row["batch_size"]) for row in rows],
                [metric(row) for row in rows],
                label=impl, color=colors(index % 10),
                marker=markers[index % len(markers)], s=65, alpha=0.8,
            )
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({int(row["batch_size"]) for rows in runs.values() for row in rows}))
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(title="Implementation", loc="upper left", bbox_to_anchor=(1.02, 1))
        path = args.out_dir / f"{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
