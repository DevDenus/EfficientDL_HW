import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import latency, energy

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "measurements.csv"
THETA_PATH = ROOT / "results" / "theta.json"
FIGURES_DIR = ROOT / "results" / "figures"

EMB_DIM = 256
OUT_DIM = 100

ATTEMPTS = 1000

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def fit_theta(image_size, batch, y, predict_fn) -> np.ndarray:

    def diff(theta):
        pred = predict_fn(image_size, batch, theta, EMB_DIM, OUT_DIM)
        return pred - y

    best_theta = None
    lowest_cost = float('inf')
    for _ in range(ATTEMPTS):
        theta0 = np.random.uniform(-10.0, 10.0, size=3)
        res = least_squares(diff, theta0, method="trf")
        cur_theta = res.x
        cur_cost = res.cost
        if cur_cost < lowest_cost:
            lowest_cost = cur_cost
            best_theta = cur_theta

    return best_theta


def evaluate(image_size, batch, y, theta, predict_fn) -> dict:
    pred = np.asarray(predict_fn(image_size, batch, theta, EMB_DIM, OUT_DIM), dtype=np.float64).reshape(-1)
    err = pred - y
    return {
        "mse": mse(y, pred),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-12)) * 100.0),
        "pred": pred,
    }


def plot_measured_vs_predicted(
    s: np.ndarray,
    b: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    ylabel: str,
    path: Path,
    metrics: dict,
) -> None:
    x = np.log(b * s**2)
    order = np.argsort(x)
    x, y_true, y_pred = x[order], y_true[order], y_pred[order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y_true, s=32, alpha=0.8, label="measured", color="orange")
    ax.scatter(x, y_pred, s=32, alpha=0.8, label="predicted", color="blue")
    ax.set_xlabel("log input bytes")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title} MSE={metrics['mse']:.4g}  MAE={metrics['mae']:.4g}  MAPE={metrics['mape']:.1f}%"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["latency_ms", "energy_mJ"]).copy()
    df["latency_ms"] = df["latency_ms"].astype(float)
    df["energy_mJ"] = df["energy_mJ"].astype(float)

    train = df.loc[~df["is_validation"]]
    val = df.loc[df["is_validation"]]

    def pack(split: pd.DataFrame):
        s = split["S"].to_numpy(dtype=np.float64)
        b = split["B"].to_numpy(dtype=np.float64)
        return s, b, split["latency_ms"].to_numpy(dtype=np.float64), split["energy_mJ"].to_numpy(dtype=np.float64)

    s_tr, b_tr, lat_tr, en_tr = pack(train)
    s_va, b_va, lat_va, en_va = pack(val)

    print(f"Train samples: {len(train)}, val samples: {len(val)}")

    theta_lat = fit_theta(s_tr, b_tr, lat_tr, latency)
    theta_energy = fit_theta(s_tr, b_tr, en_tr, energy)

    metrics = {
        "latency": {
            "train": evaluate(s_tr, b_tr, lat_tr, theta_lat, latency),
            "val": evaluate(s_va, b_va, lat_va, theta_lat, latency),
        },
        "energy": {
            "train": evaluate(s_tr, b_tr, en_tr, theta_energy, energy),
            "val": evaluate(s_va, b_va, en_va, theta_energy, energy),
        },
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_specs_comparison = [
        ("latency", "train", s_tr, b_tr, lat_tr, "latency [ms]", metrics["latency"]["train"]),
        ("latency", "val", s_va, b_va, lat_va, "latency [ms]", metrics["latency"]["val"]),
        ("energy", "train", s_tr, b_tr, en_tr, "energy [mJ]", metrics["energy"]["train"]),
        ("energy", "val", s_va, b_va, en_va, "energy [mJ]", metrics["energy"]["val"]),
    ]
    for name, split, s, b, y_true, ylabel, m in plot_specs_comparison:
        out = FIGURES_DIR / f"{name}_{split}.png"
        plot_measured_vs_predicted(
            s,
            b,
            y_true,
            m["pred"],
            title=f"{name} {split}",
            ylabel=ylabel,
            path=out,
            metrics=m,
        )

    payload = {
        "theta_lat": theta_lat.tolist(),
        "theta_energy": theta_energy.tolist(),
        "emb_dim": EMB_DIM,
        "out_dim": OUT_DIM,
        "metrics": {
            name: {
                split: {k: v for k, v in m.items() if k != "pred"}
                for split, m in split_metrics.items()
            }
            for name, split_metrics in metrics.items()
        },
    }
    THETA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THETA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for name, split_metrics in metrics.items():
        print(f"\n{name}:")
        for split, m in split_metrics.items():
            print(f"  {split}: MSE={m['mse']:.4g}  MAE={m['mae']:.4g}  MAPE={m['mape']:.2f}%")


if __name__ == "__main__":
    main()
