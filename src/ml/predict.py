"""Serve predictions from trained calibration and interpretation models."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..config import load_paths
from .feature_view import feature_columns, load_feature_frame

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _paths() -> Dict[str, str]:
    return load_paths()


@lru_cache(maxsize=None)
def _load_calibration(models_dir: Optional[Path] = None) -> Dict:
    paths = _paths()
    models_path = Path(models_dir) if models_dir else Path(paths["models_dir"])
    artifact_path = models_path / "calibration.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing calibration model at {artifact_path}")
    artifacts = joblib.load(artifact_path)
    return artifacts


@lru_cache(maxsize=None)
def _load_elastic(models_dir: Optional[Path] = None):
    paths = _paths()
    models_path = Path(models_dir) if models_dir else Path(paths["models_dir"])
    artifact_path = models_path / "elastic_net.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing elastic net model at {artifact_path}")
    return joblib.load(artifact_path)


@lru_cache(maxsize=None)
def _feature_frame(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    paths = _paths()
    processed = Path(processed_dir) if processed_dir else Path(paths["processed_dir"])
    return load_feature_frame(processed)


@lru_cache(maxsize=None)
def _feature_list(models_dir: Optional[Path] = None) -> List[str]:
    paths = _paths()
    models_path = Path(models_dir) if models_dir else Path(paths["models_dir"])
    feat_path = models_path / "feature_list.json"
    if feat_path.exists():
        try:
            loaded = json.loads(feat_path.read_text())
            if isinstance(loaded, list) and all(isinstance(item, str) for item in loaded):
                return loaded
            LOGGER.warning("feature_list.json did not contain a list of feature names; falling back to defaults.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load feature_list.json (%s); falling back to defaults.", exc)
    return feature_columns(include_derived=True)


def grade_probabilities(
    scores: pd.Series, models_dir: Optional[Path] = None
) -> pd.DataFrame:
    artifacts = _load_calibration(models_dir)
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    probabilities = model.predict_proba(scores.to_numpy().reshape(-1, 1))
    grade_labels = encoder.classes_
    prob_df = pd.DataFrame(probabilities, columns=[f"prob_{g}" for g in grade_labels])
    ab_columns = [col for col in prob_df.columns if col in {"prob_A", "prob_B"}]
    if ab_columns:
        prob_ab = prob_df[ab_columns].sum(axis=1).to_numpy()
        order = np.argsort(scores.to_numpy())
        monotonic = np.maximum.accumulate(prob_ab[order])
        adjusted = prob_ab.copy()
        adjusted[order] = monotonic
        scale = np.ones_like(adjusted, dtype=float)
        nonzero = prob_ab > 0
        scale[nonzero] = adjusted[nonzero] / prob_ab[nonzero]
        for column in ab_columns:
            prob_df[column] = prob_df[column].to_numpy(dtype=float) * scale
        grade_cols = [col for col in prob_df.columns if col.startswith("prob_") and col != "prob_AB"]
        totals = prob_df[grade_cols].sum(axis=1).replace(0.0, 1.0)
        prob_df[grade_cols] = prob_df[grade_cols].div(totals, axis=0)
        prob_df["prob_AB"] = prob_df[ab_columns].sum(axis=1)
    else:
        prob_df["prob_AB"] = 0.0
    return prob_df


def _elastic_contributions(
    feature_values: pd.DataFrame, models_dir: Optional[Path] = None
) -> pd.DataFrame:
    pipeline = _load_elastic(models_dir)
    cols = _feature_list(models_dir)
    missing_cols = [col for col in cols if col not in feature_values.columns]
    if missing_cols:
        LOGGER.warning("Missing expected feature columns: %s", ", ".join(missing_cols))
    used = feature_values.reindex(columns=cols)
    used_values = used.to_numpy(dtype=float, copy=False)
    transformed = pipeline.named_steps["imputer"].transform(used_values)
    scaled = pipeline.named_steps["scaler"].transform(transformed)
    coefs = pipeline.named_steps["model"].coef_.ravel()
    contributions = scaled * coefs
    contrib_df = pd.DataFrame(contributions, columns=cols, index=feature_values.index)
    contrib_df["intercept"] = pipeline.named_steps["model"].intercept_
    return contrib_df


def explain_drivers(
    feature_values: pd.DataFrame, models_dir: Optional[Path] = None
) -> List[Dict[str, List[str]]]:
    contributions = _elastic_contributions(feature_values, models_dir)
    explanations = []
    for idx, row in contributions.iterrows():
        contrib_series = row.drop(labels=["intercept"], errors="ignore")
        positives = [
            (feat, val) for feat, val in contrib_series.items() if val > 0
        ]
        negatives = [
            (feat, val) for feat, val in contrib_series.items() if val < 0
        ]
        positives.sort(key=lambda item: item[1], reverse=True)
        negatives.sort(key=lambda item: item[1])

        explanations.append(
            {
                "positive": [feat for feat, _ in positives[:3]],
                "negative": [feat for feat, _ in negatives[:3]],
            }
        )
    return explanations


def predict_for_fdc_id(
    fdc_id: int,
    processed_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> Dict:
    features = _feature_frame(processed_dir)
    if fdc_id not in features["fdc_id"].values:
        raise KeyError(f"FDC id {fdc_id} not found in feature frame.")
    row = features.loc[features["fdc_id"] == fdc_id].set_index("fdc_id")
    score = row["score"]
    probabilities = grade_probabilities(score, models_dir).iloc[0].to_dict()
    drivers = explain_drivers(row, models_dir)[0]
    result = {
        "fdc_id": fdc_id,
        "score": float(score.iloc[0]),
        "probabilities": probabilities,
        "drivers": drivers,
        "grade": row["grade"].iloc[0],
    }
    return result


def predict_from_features(
    feature_values: pd.DataFrame, models_dir: Optional[Path] = None
) -> pd.DataFrame:
    probabilities = grade_probabilities(feature_values["score"], models_dir)
    drivers = explain_drivers(feature_values, models_dir)
    results = feature_values[["fdc_id", "score", "grade"]].copy()
    for column in probabilities.columns:
        results[column] = probabilities[column].values
    results["drivers"] = drivers
    return results
