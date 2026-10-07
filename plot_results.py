"""Create PNG figures from prepared data or saved evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on laptops and headless machines alike.
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


BLUE = "#3066A1"
TEAL = "#168477"
ORANGE = "#C87824"
GRAY = "#737D8C"
STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _save(fig, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bars(ax, labels, series: dict, ylabel: str = "Error (USD)") -> None:
    """Zero-based grouped bars; each series supplies values and a color."""
    positions = np.arange(len(labels))
    width = 0.76 / len(series)
    maximum = 0.0
    for index, (name, (values, color)) in enumerate(series.items()):
        bars = ax.bar(
            positions + (index - (len(series) - 1) / 2) * width,
            values, width, label=name, color=color,
        )
        ax.bar_label(bars, labels=[f"{v:,.2f}" for v in values], padding=3, fontsize=9)
        maximum = max(maximum, max(values, default=0))
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(maximum * 1.28, 1))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9)


def plot_preparation(
    train: pd.DataFrame, holdout: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Use prepared flags for quality counts and training labels only for EDA."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        fig.suptitle(
            f"Data preparation | {len(train):,} training / {len(holdout):,} holdout loads\n"
            "Only monthly row counts include the holdout; other panels use training rows.",
            fontsize=14,
        )
        monthly = pd.concat({
            "Training": train.groupby(train["date"].dt.strftime("%Y-%m")).size(),
            "Holdout": holdout.groupby(holdout["date"].dt.strftime("%Y-%m")).size(),
        }, axis=1).fillna(0).sort_index()
        monthly.plot.bar(stacked=True, color=[BLUE, ORANGE], ax=axes[0, 0], rot=45)
        axes[0, 0].set(title="Chronological split", xlabel="Month", ylabel="Loads")
        axes[0, 0].set_ylim(0, monthly.sum(axis=1).max() * 1.2)
        axes[0, 0].legend(frameon=False, loc="upper center", ncols=2)

        equipment = train["equipment"].fillna("Missing").value_counts().sort_index()
        bars = axes[0, 1].bar(equipment.index, equipment.values, color=BLUE)
        axes[0, 1].bar_label(bars, labels=[f"{v:,}" for v in equipment], padding=3)
        axes[0, 1].set(title="Equipment mix | training", ylabel="Loads")
        axes[0, 1].set_ylim(0, equipment.max() * 1.18)

        # weight_missing includes invalid weights after cleaning. Subtract the
        # invalid flag so the original missing and invalid counts stay disjoint.
        issues = pd.Series({
            "Originally missing weight": int((train["weight_missing"] - train["weight_invalid"]).sum()),
            "Non-positive weight": int(train["weight_invalid"].sum()),
            "Missing market index": int(train["market_index_missing"].sum()),
        })
        bars = axes[1, 0].barh(issues.index, issues.values, color=ORANGE)
        axes[1, 0].bar_label(
            bars, labels=[f"{v:,} ({100 * v / len(train):.2f}%)" for v in issues], padding=4,
        )
        axes[1, 0].set_xlim(0, max(issues.max() * 1.6, 1))
        axes[1, 0].invert_yaxis()
        axes[1, 0].set(title="Before imputation | training", xlabel="Affected loads")

        density = axes[1, 1].hexbin(
            train["distance"], train["posted_rate"], gridsize=45,
            mincnt=1, bins="log", cmap="viridis",
        )
        axes[1, 1].set(
            title="Distance and price | training, full range",
            xlabel="Distance (miles)", ylabel="Posted rate (USD)",
        )
        fig.colorbar(density, ax=axes[1, 1], label="Loads per hexagon (log scale)")
        return [_save(fig, output_dir / "figures", "data_overview.png")]


def plot_baseline(output_dir: Path) -> list[Path]:
    """Plot the existing baseline summary without fitting or changing results."""
    summary = _read_json(output_dir / "metrics.json")
    monthly = pd.read_csv(output_dir / "metrics_by_month.csv")
    equipment = pd.read_csv(output_dir / "metrics_by_equipment.csv")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        dates = " to ".join(summary["holdout_date_range"])
        fig.suptitle(f"Equipment baseline | holdout {dates}\nLower MAE and RMSE are better.", fontsize=14)
        rates = summary["rates_per_mile"]
        _bars(axes[0, 0], list(rates), {"Rate": (list(rates.values()), BLUE)}, "Rate (USD / mile)")
        axes[0, 0].yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        axes[0, 0].axhline(summary["fallback_rate_per_mile"], color=GRAY, ls="--", label="Fallback")
        for label in axes[0, 0].texts:
            label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1})
        axes[0, 0].legend(frameon=False)
        axes[0, 0].set_title("Rates learned from training rows")
        _bars(axes[0, 1], ["MAE", "RMSE"], {"Baseline": ([summary["overall"][k] for k in ("mae", "rmse")], BLUE)})
        axes[0, 1].set_title(f"Overall holdout | {summary['holdout_rows']:,} loads")
        for ax, frame, column, title in (
            (axes[1, 0], monthly, "month", "Holdout by month"),
            (axes[1, 1], equipment, "equipment", "Holdout by equipment"),
        ):
            _bars(ax, frame[column], {"MAE": (frame["mae"], BLUE), "RMSE": (frame["rmse"], ORANGE)})
            ax.set_title(title)
        return [_save(fig, output_dir / "figures", "baseline_summary.png")]


def _group_mae(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    errors = pd.DataFrame({
        column: predictions[column].fillna("Missing"),
        "Baseline": (predictions["baseline_predicted_rate"] - predictions["posted_rate"]).abs(),
        "CatBoost": (predictions["predicted_rate"] - predictions["posted_rate"]).abs(),
    })
    return errors.groupby(column, sort=True)[["Baseline", "CatBoost"]].mean()


def plot_catboost(output_dir: Path) -> list[Path]:
    """All charts derive from saved evaluation files; CatBoost is not imported."""
    summary = _read_json(output_dir / "metrics.json")
    selection = _read_json(output_dir / "selection.json")
    cv = pd.read_csv(output_dir / "cv_results.csv")
    predictions = pd.read_csv(output_dir / "holdout_predictions.csv", parse_dates=["date"])
    importance = pd.read_csv(output_dir / "feature_importance.csv")
    directory = output_dir / "figures"
    paths = []
    chosen = selection["selected"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        folds = cv.drop_duplicates("validation_start").sort_values("validation_start")
        labels = [
            f"{row.validation_start}\nto {row.validation_end_exclusive} (exclusive)"
            for row in folds.itertuples()
        ]
        series = {"Baseline": (folds["baseline_mae"], GRAY)}
        for depth, color in zip(sorted(cv["depth"].unique()), (TEAL, BLUE)):
            values = cv.loc[cv["depth"] == depth].set_index("validation_start").loc[folds["validation_start"], "mae"]
            series[f"Depth {depth}"] = (values, color)
        _bars(ax, labels, series, "Validation MAE (USD)")
        ax.set_xlabel("Chronological validation window")
        ax.set_title(
            f"Model selection | selected depth {chosen['depth']}, {chosen['iterations']} trees\n"
            f"Selected validation-row-weighted MAE: ${chosen['cv_mae']:.2f} | Lower is better"
        )
        paths.append(_save(fig, directory, "cv_mae.png"))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        dates = " to ".join(summary["holdout_date_range"])
        fig.suptitle(
            f"Baseline vs CatBoost | holdout {dates}\n"
            f"MAE reduction: {summary['mae_reduction_percent']:.2f}% | "
            f"RMSE reduction: {summary['rmse_reduction_percent']:.2f}%",
            fontsize=14,
        )
        for ax, metric in zip(axes[0], ("mae", "rmse")):
            _bars(ax, ["All holdout loads"], {
                "Baseline": ([summary["baseline"][metric]], GRAY),
                "CatBoost": ([summary["catboost"][metric]], TEAL),
            })
            ax.set_title(f"Overall {metric.upper()} | lower is better")
        predictions["month"] = predictions["date"].dt.strftime("%Y-%m")
        for ax, column in zip(axes[1], ("month", "equipment")):
            grouped = _group_mae(predictions, column)
            _bars(ax, grouped.index, {"Baseline": (grouped["Baseline"], GRAY), "CatBoost": (grouped["CatBoost"], TEAL)}, "MAE (USD)")
            ax.set_title(f"Holdout MAE by {column}")
        paths.append(_save(fig, directory, "holdout_comparison.png"))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        fig.suptitle(f"Prediction diagnostics | {len(predictions):,} labeled holdout loads", fontsize=14)
        density = axes[0].hexbin(
            predictions["posted_rate"], predictions["predicted_rate"],
            gridsize=50, mincnt=1, bins="log", cmap="viridis",
        )
        limit = predictions[["posted_rate", "predicted_rate"]].to_numpy().max() * 1.04
        axes[0].plot([0, limit], [0, limit], "--", color=GRAY, label="Perfect prediction")
        axes[0].set(
            title="CatBoost predicted vs actual | full range",
            xlabel="Actual rate (USD)", ylabel="Predicted rate (USD)",
            xlim=(0, limit), ylim=(0, limit), aspect="equal",
        )
        axes[0].legend(frameon=False, fontsize=9)
        fig.colorbar(density, ax=axes[0], label="Loads per hexagon (log scale)", shrink=0.8)
        for column, name, color in (
            ("baseline_predicted_rate", "Baseline", GRAY),
            ("predicted_rate", "CatBoost", TEAL),
        ):
            errors = (predictions[column] - predictions["posted_rate"]).abs().to_numpy()
            thresholds, counts = np.unique(errors, return_counts=True)
            percentages = 100 * (len(errors) - np.r_[0, np.cumsum(counts)[:-1]]) / len(errors)
            axes[1].step(thresholds, percentages, where="pre", label=name, color=color)
        axes[1].set_xscale("symlog", linthresh=10)
        axes[1].set_yscale("log")
        axes[1].set(
            title="Absolute-error tail | all errors retained",
            xlabel="Absolute error (USD; linear to $10, then log)",
            ylabel="Loads with error at least this large (%)",
            ylim=(50 / len(predictions), 110),
        )
        axes[1].yaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        axes[1].grid(alpha=0.2)
        axes[1].legend(frameon=False)
        paths.append(_save(fig, directory, "prediction_diagnostics.png"))

        fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
        importance = importance.sort_values("importance", ascending=True)
        bars = ax.barh(importance["feature"], importance["importance"], color=TEAL)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in importance["importance"]], padding=3, fontsize=9)
        ax.set_xlim(0, max(importance["importance"].max() * 1.15, 1))
        ax.set(
            title="CatBoost feature importance | January-August fitted model\n"
            "PredictionValuesChange scores; correlated features can share importance.",
            xlabel="Importance score (normalized to sum to 100; not causal)",
        )
        paths.append(_save(fig, directory, "feature_importance.png"))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "baseline", "catboost"), default="all")
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--catboost-dir", type=Path, default=Path("outputs/catboost"))
    args = parser.parse_args()
    try:
        for stage, directory, plotter in (
            ("baseline", args.baseline_dir, plot_baseline),
            ("catboost", args.catboost_dir, plot_catboost),
        ):
            if args.stage in ("all", stage):
                for path in plotter(directory):
                    print(f"Created figure: {path}")
    except FileNotFoundError as exc:
        parser.exit(1, f"Missing saved result: {exc.filename}. Run the corresponding evaluation script first.\n")
