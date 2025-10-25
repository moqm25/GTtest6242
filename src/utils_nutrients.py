"""Utilities for mapping nutrients and handling unit conversions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import load_paths


CANONICAL_NUTRIENTS: Dict[str, Dict[str, Iterable[str]]] = {
    "energy_kcal": {
        "patterns": [r"\benergy\b", r"\bcalories?\b", r"kilocalorie"],
        "target_unit": "kcal",
    },
    "protein_g": {"patterns": [r"\bprotein\b"], "target_unit": "g"},
    "total_fat_g": {
        "patterns": [
            r"\btotal\b.*\bfat\b",
            r"^fat$",
            r"\btotal\b.*fatty acids\b",
        ],
        "target_unit": "g",
    },
    "sat_fat_g": {
        "patterns": [
            r"\bsaturated\b.*\bfat\b",
            r"sat\.?\s*fat",
            r"fatty.*saturated",
        ],
        "target_unit": "g",
    },
    "mono_fat_g": {
        "patterns": [r"\bmono.*fat\b", r"monounsaturated"],
        "target_unit": "g",
    },
    "poly_fat_g": {
        "patterns": [r"\bpoly.*fat\b", r"polyunsaturated"],
        "target_unit": "g",
    },
    "carbs_g": {
        "patterns": [r"\bcarbo(?:hydrate)?s?\b"],
        "target_unit": "g",
    },
    "fiber_g": {"patterns": [r"\bfiber\b", r"\bfibre\b"], "target_unit": "g"},
    "sugar_g": {
        "patterns": [r"\bsugars?\b(?!.*added)"],
        "target_unit": "g",
    },
    "added_sugar_g": {
        "patterns": [
            r"\badded\s+sugars?\b",
            r"sugars?,\s*added",
            r"added.*sugar",
        ],
        "target_unit": "g",
    },
    "sodium_mg": {"patterns": [r"\bsodium\b"], "target_unit": "mg"},
    "potassium_mg": {"patterns": [r"\bpotassium\b"], "target_unit": "mg"},
}

CANONICAL_ORDER: List[str] = list(CANONICAL_NUTRIENTS.keys())

UNIT_NORMALIZATION = {
    "µg": "ug",
    "μg": "ug",
    "mcg": "ug",
    "ug": "ug",
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "kcal": "kcal",
    "calorie": "kcal",
    "kilocalorie": "kcal",
    "kj": "kj",
    "unit": "unit",
}


CONVERSION_FACTORS = {
    ("ug", "mg"): 1 / 1000.0,
    ("ug", "g"): 1 / 1_000_000.0,
    ("mg", "g"): 1 / 1000.0,
    ("mg", "ug"): 1000.0,
    ("g", "mg"): 1000.0,
    ("g", "ug"): 1_000_000.0,
    ("kj", "kcal"): 0.239005736,
    ("kcal", "kj"): 4.184,
}


def normalize_unit(unit_name: Optional[str]) -> Optional[str]:
    if unit_name is None or pd.isna(unit_name):
        return None
    key = unit_name.strip().lower()
    return UNIT_NORMALIZATION.get(key, key)


def convert_value(value: float, from_unit: Optional[str], to_unit: str) -> float:
    """Convert a value between supported units, raising if unsupported."""
    normalized_from = normalize_unit(from_unit)
    normalized_to = normalize_unit(to_unit)
    if normalized_to is None:
        raise ValueError("Target unit cannot be None.")
    if normalized_from == normalized_to or normalized_from is None:
        return value
    pair = (normalized_from, normalized_to)
    if pair not in CONVERSION_FACTORS:
        raise ValueError(
            f"Unsupported conversion from '{from_unit}' to '{to_unit}'. "
            f"Known conversions: {sorted(CONVERSION_FACTORS.keys())}"
        )
    return value * CONVERSION_FACTORS[pair]


def robust_zscore(values: pd.Series, clip: Optional[float] = 2.5) -> pd.Series:
    """Compute robust z-score (median/MAD) with optional symmetric clipping."""
    series = values.astype(float)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or np.isnan(mad):
        zscores = pd.Series(np.zeros(len(series)), index=series.index)
    else:
        scale = 1.4826 * mad
        zscores = (series - median) / scale
    if clip is not None:
        zscores = zscores.clip(lower=-clip, upper=clip)
    return zscores


def sanitize_string(value: Optional[str]) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class NutrientSelection:
    canonical_name: str
    nutrient_id: int
    nutrient_name: str
    unit_name: str
    matched_pattern: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "canonical_name": self.canonical_name,
            "nutrient_id": self.nutrient_id,
            "nutrient_name": self.nutrient_name,
            "unit_name": self.unit_name,
            "matched_pattern": self.matched_pattern,
        }


class NutrientMatcher:
    """Automates mapping canonical nutrient names to FoodData Central ids."""

    def __init__(self, nutrient_table: pd.DataFrame):
        expected = {"id", "name", "unit_name"}
        missing = expected - set(nutrient_table.columns)
        if missing:
            raise ValueError(f"nutrient_table missing columns: {missing}")
        table = nutrient_table.copy()
        table["normalized_name"] = table["name"].astype(str).str.strip()
        table["normalized_unit"] = table["unit_name"].map(normalize_unit)
        self.table = table
        self._mapping: Dict[str, NutrientSelection] = {}

    @classmethod
    def from_data_root(cls, data_root: Path) -> "NutrientMatcher":
        nutrient_path = data_root / "nutrient.csv"
        usecols = ["id", "name", "unit_name", "nutrient_nbr"]
        nutrient_df = pd.read_csv(nutrient_path, usecols=usecols)
        return cls(nutrient_df)

    @classmethod
    def from_config(cls) -> "NutrientMatcher":
        paths = load_paths()
        return cls.from_data_root(Path(paths["data_root"]))

    def _score_candidate(self, row: pd.Series, target_unit: str) -> Tuple[int, int]:
        unit = row.get("normalized_unit")
        score = 0
        if unit == normalize_unit(target_unit):
            score += 10
        elif (unit, normalize_unit(target_unit)) in CONVERSION_FACTORS:
            score += 5
        # prefer smaller nutrient numbers if present
        nutrient_nbr = row.get("nutrient_nbr")
        if not pd.isna(nutrient_nbr):
            try:
                return score, -int(nutrient_nbr)
            except (TypeError, ValueError):
                pass
        return score, 0

    def _match_pattern(
        self, pattern: str, target_unit: str
    ) -> Optional[NutrientSelection]:
        compiled = re.compile(pattern, flags=re.IGNORECASE)
        mask = self.table["normalized_name"].str.contains(compiled, na=False)
        candidates = self.table.loc[mask].copy()
        if candidates.empty:
            return None
        if "nutrient_nbr" not in candidates.columns or candidates["nutrient_nbr"].isna().all():
            candidates["nutrient_nbr"] = np.nan
        score_df = candidates.apply(
            lambda row: pd.Series(self._score_candidate(row, target_unit)),
            axis=1,
        )
        score_df.columns = ["_unit_score", "_nbr_score"]
        candidates = pd.concat([candidates, score_df], axis=1)
        candidates.sort_values(
            by=["_unit_score", "_nbr_score"], ascending=[False, False], inplace=True
        )
        top = candidates.iloc[0]
        return NutrientSelection(
            canonical_name="",
            nutrient_id=int(top["id"]),
            nutrient_name=str(top["name"]),
            unit_name=str(top["normalized_unit"] or top["unit_name"]),
            matched_pattern=pattern,
        )

    def build_mapping(self) -> Dict[str, NutrientSelection]:
        if self._mapping:
            return self._mapping
        mapping: Dict[str, NutrientSelection] = {}
        for canonical, spec in CANONICAL_NUTRIENTS.items():
            target_unit = spec["target_unit"]
            chosen: Optional[NutrientSelection] = None
            for pattern in spec["patterns"]:
                match = self._match_pattern(pattern, target_unit)
                if match:
                    chosen = NutrientSelection(
                        canonical_name=canonical,
                        nutrient_id=match.nutrient_id,
                        nutrient_name=match.nutrient_name,
                        unit_name=match.unit_name,
                        matched_pattern=match.matched_pattern,
                    )
                    break
            if chosen is None:
                raise LookupError(
                    f"Unable to resolve nutrient id for '{canonical}'. "
                    f"Patterns tried: {spec['patterns']}"
                )
            mapping[canonical] = chosen
        self._mapping = mapping
        return mapping

    def to_markdown(self) -> str:
        mapping = self.build_mapping()
        header = "| canonical | nutrient_id | nutrient_name | unit |\n|---|---|---|---|"
        rows = [
            f"| {key} | {sel.nutrient_id} | {sel.nutrient_name} | {sel.unit_name} |"
            for key, sel in mapping.items()
        ]
        return "\n".join([header, *rows])

    def to_dataframe(self) -> pd.DataFrame:
        mapping = self.build_mapping()
        records = [sel.to_dict() for sel in mapping.values()]
        return pd.DataFrame(records).sort_values("canonical_name").reset_index(drop=True)


@lru_cache(maxsize=None)
def get_nutrient_mapping() -> Dict[str, NutrientSelection]:
    matcher = NutrientMatcher.from_config()
    return matcher.build_mapping()


def canonical_nutrient_ids() -> Dict[str, int]:
    return {name: selection.nutrient_id for name, selection in get_nutrient_mapping().items()}
