"""Select a CatBoost model on earlier dates, then evaluate the fixed holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import catboost
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from baseline import fit_baseline, predict_baseline, regression_metrics, grouped_metrics
from freight_features import CATEGORICAL_FEATURES, FreightFeatures
from prepare_data import load_labeled_data
from plot_results import plot_catboost


HOLDOUT_START = pd.Timestamp("2025-09-01")
FOLDS = [
    ("2025-03-01", "2025-05-01"),
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
]
DEPTHS = [6, 8]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_model(depth: int, iterations: int, seed: int, threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=seed,
        task_type="CPU",
        thread_count=threads,
        allow_writing_files=False,
        verbose=False,
    )


def positive_predictions(model: CatBoostRegressor, features: pd.DataFrame) -> pd.Series:
    values = np.asarray(model.predict(features), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite predictions.")
    return pd.Series(np.maximum(values, 0.01), index=features.index)


def select_model(
    development: pd.DataFrame, max_iterations: int, seed: int, threads: int, output: Path
) -> dict:
    """This function receives January-August rows only."""
    if development["date"].max() >= HOLDOUT_START:
        raise ValueError("Model selection must exclude the September-October holdout.")
    fold_results = []
    for start, end in FOLDS:
        train = development.loc[development["date"] < start].copy()
        valid = development.loc[
            (development["date"] >= start) & (development["date"] < end)
        ].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Empty chronological fold: {start} to {end}.")
        transformer = FreightFeatures.fit(train)
        x_train = transformer.transform(train)
        x_valid = transformer.transform(valid)
        rates, fallback = fit_baseline(train)
        baseline_metrics = regression_metrics(
            valid["posted_rate"], predict_baseline(valid, rates, fallback)
        )
        for depth in DEPTHS:
            print(f"CV {start} to {end} (end exclusive): depth {depth}", flush=True)
            model = make_model(depth, max_iterations, seed, threads)
            began = time.perf_counter()
            model.fit(
                x_train,
                train["posted_rate"],
                cat_features=CATEGORICAL_FEATURES,
                eval_set=(x_valid, valid["posted_rate"]),
                early_stopping_rounds=80,
                use_best_model=True,
            )
            predicted = positive_predictions(model, x_valid)
            metrics = regression_metrics(valid["posted_rate"], predicted)
            row = {
                "validation_start": start,
                "validation_end_exclusive": end,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "depth": depth,
                "best_iterations": int(model.tree_count_),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "fit_seconds": time.perf_counter() - began,
            }
            fold_results.append(row)
            pd.DataFrame(fold_results).to_csv(output / "cv_results.csv", index=False)
            print(
                f"  trees={model.tree_count_}; MAE=${metrics['mae']:.2f}; "
                f"RMSE=${metrics['rmse']:.2f}; baseline MAE=${baseline_metrics['mae']:.2f}",
                flush=True,
            )

    candidates = []
    for depth in DEPTHS:
        rows = [row for row in fold_results if row["depth"] == depth]
        weights = np.asarray([row["validation_rows"] for row in rows])
        candidates.append({
            "depth": depth,
            "cv_mae": float(np.average([row["mae"] for row in rows], weights=weights)),
            "cv_rmse": float(np.sqrt(np.average(
                [row["rmse"] ** 2 for row in rows], weights=weights
            ))),
            "iterations": int(np.median([row["best_iterations"] for row in rows])),
        })
    selected = min(candidates, key=lambda item: (item["cv_mae"], item["depth"]))
    selection = {
        "criterion": "lowest validation-row-weighted chronological CV MAE",
        "iteration_rule": "median of the selected depth's best tree counts across folds",
        "candidates": candidates,
        "selected": selected,
        "seed": seed,
        "threads": threads,
        "max_iterations": max_iterations,
        "early_stopping_rounds": 80,
    }
    write_json(output / "selection.json", selection)
    return selection


def run(data_path: Path, output: Path, max_iterations: int, seed: int, threads: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = load_labeled_data(data_path)
    development = data.loc[data["date"] < HOLDOUT_START].copy()
    holdout = data.loc[data["date"] >= HOLDOUT_START].copy()
    if development.empty or holdout.empty:
        raise ValueError("Both development and holdout data are required.")
    print(f"Development rows: {len(development):,}; holdout rows: {len(holdout):,}", flush=True)
    selection = select_model(development, max_iterations, seed, threads, output)
    chosen = selection["selected"]
    print(
        f"Selected depth {chosen['depth']}, {chosen['iterations']} trees; "
        f"CV MAE=${chosen['cv_mae']:.2f}. Refitting on January-August.",
        flush=True,
    )

    transformer = FreightFeatures.fit(development)
    x_train = transformer.transform(development)
    model = make_model(chosen["depth"], chosen["iterations"], seed, threads)
    began = time.perf_counter()
    # The outer holdout is not passed as an eval_set or used for early stopping.
    model.fit(x_train, development["posted_rate"], cat_features=CATEGORICAL_FEATURES)
    fit_seconds = time.perf_counter() - began
    model.save_model(str(output / "model.cbm"))
    transformer.save(output / "feature_state.json")

    x_holdout = transformer.transform(holdout)
    predicted = positive_predictions(model, x_holdout)
    rates, fallback = fit_baseline(development)
    baseline_predicted = predict_baseline(holdout, rates, fallback)
    catboost_metrics = regression_metrics(holdout["posted_rate"], predicted)
    baseline_metrics = regression_metrics(holdout["posted_rate"], baseline_predicted)

    predictions = holdout[
        ["load_id", "date", "pickup", "delivery", "equipment", "distance", "posted_rate"]
    ].copy()
    predictions["predicted_rate"] = predicted
    predictions["baseline_predicted_rate"] = baseline_predicted
    predictions["error"] = predicted - predictions["posted_rate"]
    predictions["month"] = predictions["date"].dt.strftime("%Y-%m")
    predictions.drop(columns="month").to_csv(
        output / "holdout_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    by_month = grouped_metrics(predictions, "month")
    by_equipment = grouped_metrics(predictions, "equipment")
    by_month.to_csv(output / "metrics_by_month.csv", index=False)
    by_equipment.to_csv(output / "metrics_by_equipment.csv", index=False)
    importance = pd.DataFrame({
        "feature": x_train.columns,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)

    summary = {
        "model": "CatBoostRegressor",
        "target": "posted_rate",
        "loss": "MAE",
        "metric_units": "USD",
        "features": list(x_train.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_inputs": ["load_id", "market_index", "quote_signal"],
        "prediction_floor_usd": 0.01,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "development_date_range": [development["date"].min().date().isoformat(), development["date"].max().date().isoformat()],
        "holdout_date_range": [holdout["date"].min().date().isoformat(), holdout["date"].max().date().isoformat()],
        "selection": selection,
        "fitted_parameters": model.get_all_params(),
        "catboost": catboost_metrics,
        "baseline": baseline_metrics,
        "mae_reduction_percent": 100.0 * (1.0 - catboost_metrics["mae"] / baseline_metrics["mae"]),
        "rmse_reduction_percent": 100.0 * (1.0 - catboost_metrics["rmse"] / baseline_metrics["rmse"]),
        "by_month": by_month.to_dict(orient="records"),
        "by_equipment": by_equipment.to_dict(orient="records"),
        "final_fit_seconds": fit_seconds,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(), "catboost": catboost.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    write_json(output / "metrics.json", summary)
    figure_paths = plot_catboost(output)
    print("\nSeptember-October holdout results:", flush=True)
    print(f"Baseline MAE=${baseline_metrics['mae']:.2f}; RMSE=${baseline_metrics['rmse']:.2f}")
    print(f"CatBoost MAE=${catboost_metrics['mae']:.2f}; RMSE=${catboost_metrics['rmse']:.2f}")
    print(f"MAE reduction: {summary['mae_reduction_percent']:.2f}%")
    print(by_month.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Saved model, preprocessing state, selection results, and holdout results to {output}")
    for path in figure_paths:
        print(f"Created figure: {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train-test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/catboost"))
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.threads < 1:
        parser.error("--max-iterations and --threads must be positive")
    run(args.data, args.output_dir, args.max_iterations, args.seed, args.threads)
