"""Feature engineering helpers shared across ML components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import load_paths


LOGGER = logging.getLogger(__name__)

BASE_FEATURE_MAP: Dict[str, str] = {
    "protein_g": "protein_g_per100g",
    "fiber_g": "fiber_g_per100g",
    "potassium_mg": "potassium_mg_per100g",
    "added_sugar_g": "added_sugar_g_per100g",
    "sugar_g": "sugar_g_per100g",
    "sodium_mg": "sodium_mg_per100g",
    "sat_fat_g": "sat_fat_g_per100g",
    "total_fat_g": "total_fat_g_per100g",
    "carbs_g": "carbs_g_per100g",
    "energy_kcal": "energy_kcal_per100g",
    "mono_fat_g": "mono_fat_g_per100g",
    "poly_fat_g": "poly_fat_g_per100g",
}


DERIVED_FEATURES = ["unsat_sat_ratio", "energy_density"]

REQUIRED_COLUMNS = {"fdc_id", "nutrition_score", "grade"}


def feature_columns(include_derived: bool = True) -> List[str]:
    columns = list(BASE_FEATURE_MAP.keys())
    if include_derived:
        columns.extend(DERIVED_FEATURES)
    return columns


def build_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    feature_frame = dataset[["fdc_id"]].copy()
    for feature, source in BASE_FEATURE_MAP.items():
        if source not in dataset.columns:
            feature_frame[feature] = np.nan
        else:
            feature_frame[feature] = dataset[source]

    mono = feature_frame["mono_fat_g"].fillna(0.0)
    poly = feature_frame["poly_fat_g"].fillna(0.0)
    sat = feature_frame["sat_fat_g"].fillna(0.0)
    feature_frame["unsat_sat_ratio"] = (mono + poly) / (sat + 1e-6)
    feature_frame["energy_density"] = feature_frame["energy_kcal"]

    feature_frame["score"] = dataset["nutrition_score"]
    feature_frame["grade"] = dataset["grade"]
    feature_frame["is_healthy"] = dataset["grade"].isin(["A", "B"]).astype(int)
    feature_frame["description"] = dataset.get("description")
    feature_frame["form"] = dataset.get("form")
    feature_frame["food_category"] = dataset.get("food_category")
    feature_frame["wweia_category"] = dataset.get("wweia_category")
    feature_frame["kcal_per_serv"] = dataset.get("kcal_per_serv")
    feature_frame["sodium_mg_perserving"] = dataset.get("sodium_mg_perserving")
    feature_frame["fiber_g_perserving"] = dataset.get("fiber_g_perserving")
    return feature_frame


def load_feature_frame(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    paths = load_paths()
    processed = Path(processed_dir) if processed_dir else Path(paths["processed_dir"])
    dataset_path = processed / "foods_nutrients.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset at {dataset_path}. Run make build first.")
    dataset = pd.read_parquet(dataset_path)
    missing = REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        LOGGER.warning(
            "Dataset at %s is missing required columns %s; running scoring step to regenerate.",
            dataset_path,
            ", ".join(sorted(missing)),
        )
        from ..scoring import run_scoring

        run_scoring({"processed_dir": str(processed)})
        dataset = pd.read_parquet(dataset_path)
        missing_after = REQUIRED_COLUMNS - set(dataset.columns)
        if missing_after:
            raise ValueError(
                f"Dataset is missing required columns after scoring: {missing_after}. "
                "Run `make score` to refresh the processed dataset."
            )
    return build_feature_frame(dataset)
