"""Prepare the labeled development data using a chronological split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_labeled_data(path: str | Path = "data/train-test.csv") -> pd.DataFrame:
    """Read and validate observed labels without fitting preprocessing values."""
    data = pd.read_csv(path)
    required = {
        "load_id", "date", "equipment", "distance", "weight",
        "market_index", "posted_rate",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise")
    if data["date"].isna().any():
        raise ValueError("Dates must be present.")
    if data["load_id"].isna().any() or data["load_id"].duplicated().any():
        raise ValueError("Load IDs must be present and unique.")

    for column in ("distance", "posted_rate", "weight", "market_index"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].dropna()).all():
            raise ValueError(f"Non-missing {column} values must be finite.")
    for column in ("distance", "posted_rate"):
        if data[column].isna().any() or (data[column] <= 0).any():
            raise ValueError(f"{column} must be present and positive.")

    return data


def prepare_splits(
    path: str | Path = "data/train-test.csv",
    cutoff: str = "2025-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return training rows, holdout rows, and training-only imputation values."""
    data = load_labeled_data(path)
    split_date = pd.Timestamp(cutoff)
    if pd.isna(split_date):
        raise ValueError("The cutoff must be a valid date.")
    train = data.loc[data["date"] < split_date].copy()
    holdout = data.loc[data["date"] >= split_date].copy()
    if train.empty or holdout.empty:
        raise ValueError("Both date partitions must contain rows.")

    for frame in (train, holdout):
        frame["weight_invalid"] = (frame["weight"] <= 0).astype("int8")
        frame.loc[frame["weight"] <= 0, "weight"] = float("nan")

    medians: dict[str, float] = {}
    for column in ("weight", "market_index"):
        median = train[column].median()
        if pd.isna(median):
            raise ValueError(f"No valid training values for {column}.")
        medians[column] = float(median)
        for frame in (train, holdout):
            frame[f"{column}_missing"] = frame[column].isna().astype("int8")
            frame[column] = frame[column].fillna(median)

    return train, holdout, medians


if __name__ == "__main__":
    from plot_results import plot_preparation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train-test.csv"))
    parser.add_argument("--cutoff", default="2025-09-01")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preparation"))
    args = parser.parse_args()
    train, holdout, medians = prepare_splits(args.data, args.cutoff)
    print("Training rows:", len(train))
    print("Holdout rows:", len(holdout))
    print("Training medians:", medians)
    for path in plot_preparation(train, holdout, args.output_dir):
        print(f"Created figure: {path}")
