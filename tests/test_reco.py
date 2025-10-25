from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_paths
from src.reco import get_substitutions, load_reco_config


@pytest.fixture(scope="session")
def assets():
    paths = load_paths()
    processed = Path(paths["processed_dir"])
    if not (processed / "foods_nutrients.parquet").exists():
        pytest.skip("Run make build + make score before recommender tests.")
    if not (processed / "nn_index.parquet").exists():
        pytest.skip("Run make reco before recommender tests.")
    dataset = pd.read_parquet(processed / "foods_nutrients.parquet")
    config = load_reco_config(Path("configs/reco.yml"))
    return {"processed": processed, "dataset": dataset, "config": config}


def test_constraints_enforced(assets) -> None:
    dataset = assets["dataset"]
    target = dataset[dataset["grade"].isin(["D", "E"])]
    if target.empty:
        pytest.skip("No grade D/E items for constraint test.")
    fdc_id = int(target["fdc_id"].iloc[0])
    swaps = get_substitutions(fdc_id, k=5, processed_dir=assets["processed"])
    if swaps.empty:
        pytest.skip("No swaps generated to verify constraints.")
    min_gain = assets["config"]["constraints"]["min_score_gain"]
    assert (swaps["score_gain"].fillna(0) >= min_gain).all()


def test_deterministic_ranking(assets) -> None:
    dataset = assets["dataset"]
    fdc_id = int(dataset["fdc_id"].iloc[0])
    swaps_a = get_substitutions(fdc_id, k=5, processed_dir=assets["processed"])
    swaps_b = get_substitutions(fdc_id, k=5, processed_dir=assets["processed"])
    pd.testing.assert_frame_equal(swaps_a.reset_index(drop=True), swaps_b.reset_index(drop=True))
