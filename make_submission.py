"""Refit the selected CatBoost configuration and generate the final submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import catboost
import numpy as np
import pandas as pd

from freight_features import CATEGORICAL_FEATURES, FreightFeatures
from prepare_data import load_labeled_data
from score import EXPECTED_IDS, EXPECTED_ROWS, validate_december, validate_predictions
from train_catboost import make_model, positive_predictions, write_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_configuration(evaluation_dir: Path, train_path: Path) -> tuple[dict, dict]:
    metrics_path = evaluation_dir / "metrics.json"
    selection_path = evaluation_dir / "selection.json"
    if not metrics_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Run python train_catboost.py before making the submission.")
    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if evaluation["data_sha256"] != sha256(train_path):
        raise ValueError("Training data differs from the evaluated file. Rerun train_catboost.py.")
    if evaluation["selection"] != selection:
        raise ValueError("The saved selection and evaluation results do not match.")
    for key in ("depth", "iterations"):
        value = selection["selected"][key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid selected {key}.")
    if evaluation["versions"]["catboost"] != catboost.__version__:
        raise ValueError("Use the same CatBoost version as the evaluated model.")
    return selection, evaluation


def load_prediction_inputs(validation_path: Path, template_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    loads = pd.read_csv(validation_path, dtype={"load_id": str})
    template = pd.read_csv(template_path, dtype={"load_id": str})
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("The prediction template must have load_id,predicted_rate columns.")
    for label, frame in (("validation", loads), ("template", template)):
        if "load_id" not in frame or len(frame) != EXPECTED_ROWS:
            raise ValueError(f"{label} must contain {EXPECTED_ROWS:,} load IDs.")
        if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
            raise ValueError(f"{label} contains missing or duplicate load IDs.")
        if set(frame["load_id"]) != EXPECTED_IDS:
            raise ValueError(f"{label} IDs do not match the assessment's final loads.")
    return loads, template


def run(args: argparse.Namespace) -> dict:
    inputs = [args.train, args.validation, args.template, args.december_inputs]
    input_paths = {path.resolve() for path in inputs}
    output_paths = [args.predictions, args.december_predictions, args.summary]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Prediction and summary output paths must be distinct.")
    if any(path.resolve() in input_paths for path in output_paths):
        raise ValueError("Output paths must not overwrite original input files.")
    if args.model_dir.resolve() == args.evaluation_dir.resolve():
        raise ValueError("Keep the final model separate from the evaluation model.")

    selection, evaluation = load_selected_configuration(args.evaluation_dir, args.train)
    train = load_labeled_data(args.train)
    loads, template = load_prediction_inputs(args.validation, args.template)
    december = pd.read_csv(args.december_inputs)
    if set(train["load_id"]) & set(loads["load_id"]):
        raise ValueError("Training and final prediction IDs overlap.")
    validation_dates = pd.to_datetime(loads["date"], format="%Y-%m-%d", errors="raise")
    if validation_dates.isna().any() or train["date"].max() >= validation_dates.min():
        raise ValueError("Final prediction dates must follow all labeled training dates.")

    chosen = selection["selected"]
    print(f"Final training rows: {len(train):,}", flush=True)
    print(f"Selected configuration: depth {chosen['depth']}, {chosen['iterations']} trees", flush=True)
    transformer = FreightFeatures.fit(train)
    x_train = transformer.transform(train)
    model = make_model(chosen["depth"], chosen["iterations"], selection["seed"], selection["threads"])
    # Use every labeled row with the already selected settings, without a new eval_set.
    model.fit(x_train, train["posted_rate"], cat_features=CATEGORICAL_FEATURES)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_dir / "model.cbm"))
    transformer.save(args.model_dir / "feature_state.json")

    values = positive_predictions(model, transformer.transform(loads)).round(2)
    by_id = pd.Series(values.to_numpy(), index=loads["load_id"])
    predictions = pd.DataFrame({
        "load_id": template["load_id"],
        "predicted_rate": template["load_id"].map(by_id),
    })
    december_predictions = december.copy()
    december_predictions["predicted_rate"] = positive_predictions(
        model, transformer.transform(december)
    ).round(2)
    validate_predictions(predictions)
    validate_december(december_predictions)

    for path in (args.predictions, args.december_predictions, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False, float_format="%.2f")
    december_predictions.to_csv(args.december_predictions, index=False, float_format="%.2f")
    print(f"Saved {len(predictions):,} final predictions to {args.predictions}", flush=True)
    print(f"Saved {len(december_predictions)} December predictions to {args.december_predictions}", flush=True)

    # Run the supplied script on the actual files, including its original chart code.
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("score.py")),
        "--predictions", str(args.predictions),
        "--december-predictions", str(args.december_predictions),
        "--output-dir", str(args.scorer_dir),
    ], check=True)

    known_cities = set(transformer.city_coordinates)
    unseen_city_rows = ~loads["pickup"].isin(known_cities) | ~loads["delivery"].isin(known_cities)
    summary = {
        "stage": "final_fit_and_prediction",
        "training_rows": len(train),
        "training_date_range": [train["date"].min().date().isoformat(), train["date"].max().date().isoformat()],
        "selected_depth": chosen["depth"],
        "selected_iterations": chosen["iterations"],
        "random_seed": selection["seed"],
        "threads": selection["threads"],
        "fitted_parameters": model.get_all_params(),
        "feature_names": list(x_train.columns),
        "final_weight_median": transformer.weight_median,
        "final_prediction_rows": len(predictions),
        "december_prediction_rows": len(december_predictions),
        "final_loads_with_unseen_city": int(unseen_city_rows.sum()),
        "prices_rounded_to_decimal_places": 2,
        "final_predicted_rate_min": float(predictions["predicted_rate"].min()),
        "final_predicted_rate_max": float(predictions["predicted_rate"].max()),
        "december_predicted_rate_min": float(december_predictions["predicted_rate"].min()),
        "december_predicted_rate_max": float(december_predictions["predicted_rate"].max()),
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "evaluation_metrics_source": str(args.evaluation_dir / "metrics.json"),
        "evaluation_note": "The earlier September-October metrics describe the January-August evaluation model, not this final refit.",
        "final_accuracy_metrics": None,
        "final_accuracy_note": "True prices for the 12,000 final loads are not supplied. The provided scorer checks output format and generates the chart.",
        "scorer_passed": True,
        "outputs": {
            "predictions": str(args.predictions),
            "december_predictions": str(args.december_predictions),
            "chart": str(args.scorer_dir / "candidate_december.png"),
            "model": str(args.model_dir / "model.cbm"),
            "feature_state": str(args.model_dir / "feature_state.json"),
        },
        "versions": {
            "python": platform.python_version(), "catboost": catboost.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    write_json(args.summary, summary)
    print(f"Saved final-fit summary to {args.summary}")
    print("Earlier holdout evaluation remains in outputs/catboost/metrics.json.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train-test.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.csv"))
    parser.add_argument("--template", type=Path, default=Path("data/validation-predictions-template.csv"))
    parser.add_argument("--december-inputs", type=Path, default=Path("data/december-chart-inputs.csv"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/catboost"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--predictions", type=Path, default=Path("validation_predictions.csv"))
    parser.add_argument("--december-predictions", type=Path, default=Path("submission/december_predictions.csv"))
    parser.add_argument("--scorer-dir", type=Path, default=Path("scorer_results"))
    parser.add_argument("--summary", type=Path, default=Path("submission/run_summary.json"))
    run(parser.parse_args())
