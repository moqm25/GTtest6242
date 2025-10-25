"""Train calibration and interpretation models for nutrition scoring."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config import load_paths
from ..progress import progress_for_iterable, timed_step
from .feature_view import feature_columns, load_feature_frame


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def train_calibration_model(features: pd.DataFrame) -> Dict:
    usable = features.dropna(subset=["score", "grade"])
    X = usable[["score"]].to_numpy()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(usable["grade"])

    logistic = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    splits = cv.split(X, y_encoded)
    n_splits = cv.get_n_splits(X, y_encoded)
    for train_idx, val_idx in progress_for_iterable(splits, "Calibration CV folds", length=n_splits):
        model = clone(logistic)
        model.fit(X[train_idx], y_encoded[train_idx])
        probs = model.predict_proba(X[val_idx])
        scores.append(log_loss(y_encoded[val_idx], probs, labels=model.classes_))
    mean_log_loss = float(np.mean(scores))
    LOGGER.info("Calibration log-loss (5-fold): %s", mean_log_loss)
    logistic.fit(X, y_encoded)
    return {
        "model": logistic,
        "encoder": encoder,
        "cv_log_loss": mean_log_loss,
    }


def train_elastic_net(features: pd.DataFrame) -> Dict:
    cols = feature_columns(include_derived=True)
    usable = features.dropna(subset=["score"])
    usable = usable.copy()
    valid_cols = [col for col in cols if usable[col].notna().any()]
    nunique = usable[valid_cols].nunique(dropna=True)
    valid_cols = [col for col in valid_cols if nunique[col] > 1]
    dropped = sorted(set(cols) - set(valid_cols))
    if dropped:
        LOGGER.info("Dropping constant/empty features: %s", ", ".join(dropped))
    X = usable[valid_cols].to_numpy()
    y = usable["is_healthy"].to_numpy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    splits = cv.split(X, y)
    n_splits = cv.get_n_splits(X, y)
    for train_idx, val_idx in progress_for_iterable(splits, "ElasticNet CV folds", length=n_splits):
        pipe = clone(pipeline)
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[val_idx])
        mse_scores.append(mean_squared_error(y[val_idx], preds))
    rmse = float(np.sqrt(np.mean(mse_scores)))
    LOGGER.info("ElasticNet RMSE (5-fold): %s", rmse)
    pipeline.fit(X, y)
    return {
        "pipeline": pipeline,
        "features": valid_cols,
        "cv_rmse": rmse,
    }


def write_coef_json(pipeline: Pipeline, feature_names: pd.Index, output_path: Path) -> None:
    model: ElasticNet = pipeline.named_steps["model"]
    coefs = dict(zip(feature_names, model.coef_.tolist()))
    payload = {
        "intercept": float(model.intercept_),
        "coefficients": coefs,
        "note": "Coefficients are with respect to the standardized features used in the pipeline.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def train_models(paths_override: Optional[Dict[str, str]] = None) -> None:
    paths = load_paths().copy()
    if paths_override:
        paths.update(
            {key: str(Path(value)) for key, value in paths_override.items() if value is not None}
        )
    processed_dir = Path(paths["processed_dir"])
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    with timed_step("Loading feature frame"):
        features = load_feature_frame(processed_dir)

    with timed_step("Training calibration model"):
        calibration_artifacts = train_calibration_model(features)
    calibration_path = models_dir / "calibration.joblib"
    joblib.dump(
        {"model": calibration_artifacts["model"], "encoder": calibration_artifacts["encoder"]},
        calibration_path,
    )

    with timed_step("Training ElasticNet model"):
        elastic_artifacts = train_elastic_net(features)
    elastic_path = models_dir / "elastic_net.joblib"
    joblib.dump(elastic_artifacts["pipeline"], elastic_path)

    with timed_step("Writing model artifacts"):
        write_coef_json(
            elastic_artifacts["pipeline"],
            pd.Index(elastic_artifacts["features"]),
            models_dir / "coef.json",
        )
        feature_list_path = models_dir / "feature_list.json"
        feature_list_path.write_text(json.dumps(elastic_artifacts["features"]))

        metrics_path = models_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "calibration_log_loss": calibration_artifacts["cv_log_loss"],
                    "elastic_net_rmse": elastic_artifacts["cv_rmse"],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    LOGGER.info("Artifacts written to %s", models_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.processed_dir:
        overrides["processed_dir"] = str(args.processed_dir)
    if args.models_dir:
        overrides["models_dir"] = str(args.models_dir)
    train_models(overrides if overrides else None)


if __name__ == "__main__":
    main()
