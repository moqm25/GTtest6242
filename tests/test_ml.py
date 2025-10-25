from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_paths
from src.ml.feature_view import load_feature_frame
from src.ml.predict import grade_probabilities, predict_for_fdc_id


@pytest.fixture(scope="session")
def assets():
    paths = load_paths()
    processed_dir = Path(paths["processed_dir"])
    models_dir = Path(paths["models_dir"])
    if not (processed_dir / "foods_nutrients.parquet").exists():
        pytest.skip("Run make build + make score before ML tests.")
    if not (models_dir / "calibration.joblib").exists():
        pytest.skip("Run make ml before ML tests.")
    return {"processed": processed_dir, "models": models_dir}


def test_grade_probability_shapes(assets) -> None:
    frame = load_feature_frame(assets["processed"])
    probs = grade_probabilities(frame["score"], assets["models"])
    assert len(probs) == len(frame)
    prob_cols = [col for col in probs.columns if col.startswith("prob_") and col != "prob_AB"]
    assert np.allclose(probs[prob_cols].sum(axis=1), 1.0, atol=1e-4)
    assert (probs[prob_cols] >= 0).all().all()


def test_monotonic_ab_probability(assets) -> None:
    frame = load_feature_frame(assets["processed"])
    probs = grade_probabilities(frame["score"], assets["models"])
    ordered = probs.assign(score=frame["score"]).sort_values("score")
    diffs = ordered["prob_AB"].diff().fillna(0)
    assert (diffs >= -1e-4).all(), "AB probability should increase with score"


def test_predict_single_item(assets) -> None:
    frame = load_feature_frame(assets["processed"])
    fdc_id = int(frame["fdc_id"].iloc[0])
    result = predict_for_fdc_id(fdc_id, assets["processed"], assets["models"])
    for key in {"fdc_id", "score", "probabilities", "drivers", "grade"}:
        assert key in result
