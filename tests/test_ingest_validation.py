from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest_usda import normalize_and_validate, RENAME, REQUIRED

DATA_DIR = Path("DATA")


CSV_FILES = [
    "food.csv",
    "foundation_food.csv",
    "sr_legacy_food.csv",
    "branded_food.csv",
    "food_nutrient.csv",
    "nutrient.csv",
    "food_portion.csv",
    "measure_unit.csv",
    "food_category.csv",
    "wweia_food_category.csv",
]


@pytest.mark.parametrize("filename", CSV_FILES)
def test_required_columns_present(filename: str) -> None:
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in DATA directory")
    frame = pd.read_csv(path, low_memory=False)
    normalized = normalize_and_validate(frame, filename)
    assert not normalized.empty or filename == "branded_food.csv"


def test_nutrient_join_unit_name_coverage() -> None:
    nutrient_path = DATA_DIR / "nutrient.csv"
    food_nutrient_path = DATA_DIR / "food_nutrient.csv"
    if not nutrient_path.exists() or not food_nutrient_path.exists():
        pytest.skip("nutrient.csv or food_nutrient.csv missing")
    nutrients = normalize_and_validate(pd.read_csv(nutrient_path, low_memory=False), "nutrient.csv")
    food_nutrient = normalize_and_validate(
        pd.read_csv(food_nutrient_path, low_memory=False),
        "food_nutrient.csv",
    )
    merged = food_nutrient.merge(
        nutrients[["id", "unit_name"]],
        left_on="nutrient_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )
    missing_fraction = merged["unit_name"].isna().mean()
    assert missing_fraction <= 0.02, f"unit_name missing for {missing_fraction:.2%} of rows"
