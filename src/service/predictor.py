"""High-level service wrapper for the Streamlit app."""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import load_paths
from ..ml import predict as ml_predict
from ..ml.feature_view import load_feature_frame
from ..reco import get_substitutions


class PredictorService:
    """Expose reusable helpers for the app without repeated I/O."""

    def __init__(self, processed_dir: Optional[Path] = None):
        paths = load_paths()
        self.processed_dir = Path(processed_dir) if processed_dir else Path(paths["processed_dir"])
        dataset_path = self.processed_dir / "foods_nutrients.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError("Run make build before starting the app.")
        self._dataset = pd.read_parquet(dataset_path)
        index_path = self.processed_dir / "nn_index.parquet"
        if index_path.exists():
            self._nn_index = pd.read_parquet(index_path)
        else:
            self._nn_index = pd.DataFrame()
        self._features = load_feature_frame(self.processed_dir)

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    @property
    def neighbors(self) -> pd.DataFrame:
        return self._nn_index

    def get_food(self, fdc_id: int) -> Dict:
        match = self._dataset[self._dataset["fdc_id"] == fdc_id]
        if match.empty:
            raise KeyError(f"FDC ID {fdc_id} not found.")
        return match.iloc[0].to_dict()

    @lru_cache(maxsize=128)
    def predict(self, fdc_id: int) -> Dict:
        return ml_predict.predict_for_fdc_id(fdc_id, self.processed_dir)

    def recommend(self, fdc_id: int, k: int = 5, constraints: Optional[Dict] = None) -> pd.DataFrame:
        if self._nn_index.empty:
            raise FileNotFoundError("Run make reco before requesting substitutions.")
        return get_substitutions(
            food_id=fdc_id,
            k=k,
            constraints=constraints,
            foods=self.dataset,
            nn_index=self.neighbors,
        )

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        if len(query) < 2:
            return self._dataset.nlargest(limit, "nutrition_score")[
                [
                    "fdc_id",
                    "description",
                    "grade",
                    "nutrition_score",
                    "kcal_per_serv",
                    "sodium_mg_perserving",
                    "sugar_g_perserving",
                    "food_category",
                    "wweia_category",
                ]
            ]
        query_lower = query.lower()
        # coarse filter using string containment across multiple text columns
        candidates = self._dataset[
            self._dataset["description"].str.lower().str.contains(query_lower, na=False)
            | self._dataset["food_category"].fillna("").str.lower().str.contains(query_lower)
            | self._dataset["wweia_category"].fillna("").str.lower().str.contains(query_lower)
        ].copy()
        if candidates.empty:
            candidates = self._dataset.copy()

        def similarity(text: str) -> float:
            text = (text or "").lower()
            return SequenceMatcher(None, query_lower, text).ratio()

        candidates["match_score"] = candidates["description"].map(similarity)
        candidates.sort_values(
            by=["match_score", "nutrition_score"], ascending=[False, False], inplace=True
        )
        results = candidates.head(limit).copy()
        return results[
            [
                "fdc_id",
                "description",
                "grade",
                "nutrition_score",
                "kcal_per_serv",
                "sodium_mg_perserving",
                "sugar_g_perserving",
                "food_category",
                "wweia_category",
            ]
        ]

    def compare(self, ids: list[int]) -> pd.DataFrame:
        return self._dataset[self._dataset["fdc_id"].isin(ids)]
