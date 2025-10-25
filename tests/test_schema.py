from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_paths


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    paths = load_paths()
    dataset_path = Path(paths["processed_dir"]) / "foods_nutrients.parquet"
    if not dataset_path.exists():
        pytest.skip("Run make build before executing schema tests.")
    return pd.read_parquet(dataset_path)


def test_expected_columns(dataset: pd.DataFrame) -> None:
    expected = {
        "fdc_id",
        "description",
        "data_type",
        "food_category",
        "wweia_category",
        "form",
        "nutrition_score",
        "grade",
        "energy_kcal_per100g",
        "protein_g_per100g",
        "fiber_g_per100g",
        "sodium_mg_per100g",
        "sugar_g_perserving",
        "serving_gram_weight",
        "serving_desc",
    }
    missing = expected - set(dataset.columns)
    assert not missing, f"Dataset is missing expected columns: {missing}"


def test_fdc_id_unique(dataset: pd.DataFrame) -> None:
    assert dataset["fdc_id"].is_unique


def test_added_sugar_flag(dataset: pd.DataFrame) -> None:
    assert "added_sugar_missing" in dataset.columns
    missing_flag = dataset["added_sugar_missing"]
    assert set(missing_flag.unique()) <= {0, 1}
