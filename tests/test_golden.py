from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.config import load_paths
from src.reco import get_substitutions, load_reco_config


def _load_dataset() -> pd.DataFrame:
    paths = load_paths()
    processed = Path(paths["processed_dir"])
    dataset_path = processed / "foods_nutrients.parquet"
    if not dataset_path.exists():
        pytest.skip("Run make build + make score before golden tests.")
    return pd.read_parquet(dataset_path)


def _pick_items(dataset: pd.DataFrame, keyword: str, n: int = 2, highest: bool = True) -> List[int]:
    mask = (
        dataset["description"].str.contains(keyword, case=False, na=False)
        | dataset["food_category"].fillna("").str.contains(keyword, case=False)
        | dataset["wweia_category"].fillna("").str.contains(keyword, case=False)
    )
    subset = dataset.loc[mask].dropna(subset=["nutrition_score"])
    if subset.empty:
        return []
    ordered = subset.nlargest(n, "nutrition_score") if highest else subset.nsmallest(n, "nutrition_score")
    return ordered["fdc_id"].astype(int).tolist()


def test_group_score_expectations():
    dataset = _load_dataset()
    groups = {
        "fruit": _pick_items(dataset, "fruit", highest=True),
        "vegetable": _pick_items(dataset, "vegetable", highest=True),
        "soda": _pick_items(dataset, "soda", highest=False),
        "snack": _pick_items(dataset, "snack", highest=False),
    }
    score_lookup = dataset.set_index("fdc_id")["nutrition_score"]
    grade_lookup = dataset.set_index("fdc_id")["grade"]

    if len(groups["fruit"]) >= 2:
        fruit_scores = score_lookup.loc[groups["fruit"]]
        assert fruit_scores.mean() >= 60
        assert (grade_lookup.loc[groups["fruit"]].isin(["A", "B"])).any()

    if len(groups["vegetable"]) >= 2:
        veg_scores = score_lookup.loc[groups["vegetable"]]
        assert veg_scores.mean() >= 65
        assert (grade_lookup.loc[groups["vegetable"]].isin(["A", "B"])).any()

    if len(groups["soda"]) >= 2:
        soda_scores = score_lookup.loc[groups["soda"]]
        assert soda_scores.mean() <= 55
        assert (grade_lookup.loc[groups["soda"]].isin(["D", "E"])).any()

    if len(groups["snack"]) >= 2:
        snack_scores = score_lookup.loc[groups["snack"]]
        assert snack_scores.mean() <= 65


def test_de_items_have_swaps():
    dataset = _load_dataset()
    paths = load_paths()
    processed = Path(paths["processed_dir"])
    config = load_reco_config(Path("configs/reco.yml"))
    min_gain = config.get("constraints", {}).get("min_score_gain", 10)

    target = dataset[dataset["grade"].isin(["D", "E"])].head(10)
    if target.empty:
        pytest.skip("No grade D/E items to evaluate coverage.")

    for fdc_id in target["fdc_id"].astype(int):
        swaps = get_substitutions(fdc_id, processed_dir=processed)
        assert not swaps.empty, f"No swaps for D/E item {fdc_id}"
        assert (swaps["score_gain"].fillna(0) >= min_gain).any()
