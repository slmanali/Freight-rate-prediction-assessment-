"""Training-fitted freight features shared by evaluation and later prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "lane"]
REQUIRED_INPUTS = {"pickup", "delivery", "equipment", "distance", "weight", "date"}


@dataclass
class FreightFeatures:
    weight_median: float
    city_coordinates: dict[str, list[float]]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "FreightFeatures":
        """Learn the weight median and city coordinates from this training fold."""
        weights = pd.to_numeric(train["weight"], errors="raise")
        weights = weights.where((weights > 0) & np.isfinite(weights))
        median = float(weights.median())
        if not np.isfinite(median):
            raise ValueError("Training data has no valid positive weights.")

        locations = pd.concat([
            train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(
                ["city", "lat", "lon"], axis=1
            ),
            train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
                ["city", "lat", "lon"], axis=1
            ),
        ], ignore_index=True)
        locations = locations.dropna(subset=["city", "lat", "lon"])
        locations["city"] = locations["city"].astype(str)
        if (locations.groupby("city")[["lat", "lon"]].nunique() > 1).any().any():
            raise ValueError("Training data contains conflicting coordinates for a city.")
        coordinates = {
            city: [float(row["lat"]), float(row["lon"])]
            for city, row in locations.drop_duplicates("city").set_index("city").iterrows()
        }
        return cls(weight_median=median, city_coordinates=coordinates)

    def transform(self, loads: pd.DataFrame) -> pd.DataFrame:
        """Build features without reading load IDs, labels, or auxiliary signals."""
        missing = REQUIRED_INPUTS - set(loads.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {sorted(missing)}")
        result = pd.DataFrame(index=loads.index)
        for name in ("pickup", "delivery", "equipment"):
            result[name] = loads[name].fillna("__MISSING__").astype(str)
        result["lane"] = result["pickup"] + " -> " + result["delivery"]

        distance = pd.to_numeric(loads["distance"], errors="raise")
        if not np.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError("Distance must be finite and positive.")
        result["distance"] = distance.astype(float)
        result["log_distance"] = np.log1p(distance)

        weights = pd.to_numeric(loads["weight"], errors="raise")
        invalid = (weights <= 0) | (~np.isfinite(weights) & weights.notna())
        cleaned_weights = weights.mask(invalid)
        result["weight"] = cleaned_weights.fillna(self.weight_median).astype(float)
        result["weight_missing"] = cleaned_weights.isna().astype("int8")
        result["weight_invalid"] = invalid.astype("int8")

        for role in ("pickup", "delivery"):
            for offset, axis in enumerate(("lat", "lon")):
                name = f"{role}_{axis}"
                supplied = pd.to_numeric(
                    loads[name] if name in loads else pd.Series(np.nan, index=loads.index),
                    errors="raise",
                )
                supplied = supplied.where(np.isfinite(supplied))
                lookup = {city: pair[offset] for city, pair in self.city_coordinates.items()}
                result[name] = supplied.fillna(result[role].map(lookup)).astype(float)
        result["latitude_change"] = result["delivery_lat"] - result["pickup_lat"]
        result["longitude_change"] = result["delivery_lon"] - result["pickup_lon"]

        dates = pd.to_datetime(loads["date"], format="%Y-%m-%d", errors="raise")
        if dates.isna().any():
            raise ValueError("Dates must be present.")
        result["month"] = dates.dt.month
        result["day_of_month"] = dates.dt.day
        result["day_of_week"] = dates.dt.dayofweek
        result["weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
        period = np.where(dates.dt.is_leap_year, 366.0, 365.0)
        angle = 2.0 * np.pi * (dates.dt.dayofyear - 1) / period
        result["day_of_year_sin"] = np.sin(angle)
        result["day_of_year_cos"] = np.cos(angle)
        return result

    def save(self, path: Path) -> None:
        state = {
            "weight_median": self.weight_median,
            "city_coordinates": self.city_coordinates,
        }
        path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FreightFeatures":
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls(**state)
