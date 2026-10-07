"""Evaluate an equipment-based median rate-per-mile baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import prepare_splits
from plot_results import plot_baseline


def fit_baseline(train: pd.DataFrame) -> tuple[pd.Series, float]:
    """Learn equipment rates and an unknown-equipment fallback from training."""
    rate_per_mile = train["posted_rate"] / train["distance"]
    rates = rate_per_mile.groupby(train["equipment"]).median()
    return rates, float(rate_per_mile.median())


def predict_baseline(
    loads: pd.DataFrame, rates: pd.Series, fallback: float
) -> pd.Series:
    """Use the overall training median for missing or unseen equipment."""
    return loads["equipment"].map(rates).fillna(fallback) * loads["distance"]


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    errors = predicted.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def grouped_metrics(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in predictions.groupby(column, sort=True, dropna=False):
        rows.append({
            column: value,
            "rows": len(group),
            **regression_metrics(group["posted_rate"], group["predicted_rate"]),
        })
    return pd.DataFrame(rows)


def run_baseline(data_path: Path, cutoff: str, output_dir: Path) -> dict:
    train, holdout, medians = prepare_splits(data_path, cutoff)
    rates, fallback = fit_baseline(train)

    predictions = holdout[
        ["load_id", "date", "equipment", "distance", "posted_rate"]
    ].copy()
    predictions["predicted_rate"] = predict_baseline(holdout, rates, fallback)
    if not np.isfinite(predictions["predicted_rate"]).all():
        raise ValueError("Predictions must be finite.")
    if (predictions["predicted_rate"] <= 0).any():
        raise ValueError("Predictions must be positive.")
    predictions["error"] = predictions["predicted_rate"] - predictions["posted_rate"]
    predictions["month"] = predictions["date"].dt.strftime("%Y-%m")

    overall = regression_metrics(predictions["posted_rate"], predictions["predicted_rate"])
    by_month = grouped_metrics(predictions, "month")
    by_equipment = grouped_metrics(predictions, "equipment")
    summary = {
        "model": "equipment_median_rate_per_mile",
        "features": ["equipment", "distance"],
        "target": "posted_rate",
        "metric_units": "USD",
        "cutoff": cutoff,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_date_range": [train["date"].min().date().isoformat(), train["date"].max().date().isoformat()],
        "holdout_date_range": [holdout["date"].min().date().isoformat(), holdout["date"].max().date().isoformat()],
        "rates_per_mile": {str(key): float(value) for key, value in rates.items()},
        "fallback_rate_per_mile": fallback,
        "preprocessing_medians": medians,
        "overall": overall,
        "by_month": by_month.to_dict(orient="records"),
        "by_equipment": by_equipment.to_dict(orient="records"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.drop(columns="month").to_csv(
        output_dir / "holdout_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    by_month.to_csv(output_dir / "metrics_by_month.csv", index=False)
    by_equipment.to_csv(output_dir / "metrics_by_equipment.csv", index=False)
    figure_paths = plot_baseline(output_dir)

    print(f"Training rows: {len(train):,}")
    print(f"Holdout rows: {len(holdout):,}")
    print("Learned rates per mile:")
    for equipment, rate in rates.items():
        print(f"  {equipment}: ${rate:.4f}")
    print(f"MAE: ${overall['mae']:,.2f}")
    print(f"RMSE: ${overall['rmse']:,.2f}")
    print("Monthly holdout results:")
    print(by_month.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"Saved results to {output_dir}")
    for path in figure_paths:
        print(f"Created figure: {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train-test.csv"))
    parser.add_argument("--cutoff", default="2025-09-01")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    run_baseline(args.data, args.cutoff, args.output_dir)
