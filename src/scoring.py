"""Compute rule-based nutrition scores and grades."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from .config import load_paths, project_root
from .progress import timed_step


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_scoring_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def capped_percent(value: pd.Series, daily_value: float) -> pd.Series:
    pct = (value / daily_value) * 100.0
    return pct.clip(lower=0.0, upper=100.0)


def safe_mean(values: pd.DataFrame) -> pd.Series:
    return values.mean(axis=1, skipna=True).fillna(0.0)


def compute_scores(dataset: pd.DataFrame, config: Dict) -> pd.DataFrame:
    dv = config["daily_values"]
    weights = config["weights"]

    protein_pct = capped_percent(dataset["protein_g_perserving"], dv["protein_g"])
    fiber_pct = capped_percent(dataset["fiber_g_perserving"], dv["fiber_g"])
    potassium_pct = capped_percent(dataset["potassium_mg_perserving"], dv["potassium_mg"])

    added_sugar = dataset["added_sugar_g_perserving"]
    added_sugar_missing = dataset["added_sugar_missing"] == 1
    sugar_estimate = dataset["sugar_g_perserving"] * config.get("penalty_estimate_factor", 0.75)
    added_sugar = added_sugar.where(~added_sugar_missing, sugar_estimate)
    dataset["penalty_estimated"] = added_sugar_missing.astype(int)

    added_sugar_penalty = capped_percent(added_sugar, dv["added_sugar_g"])
    sodium_penalty = capped_percent(dataset["sodium_mg_perserving"], dv["sodium_mg"])
    sat_fat_penalty = capped_percent(dataset["sat_fat_g_perserving"], dv["sat_fat_g"])
    energy_density_penalty = (
        (dataset["energy_kcal_per100g"] / config["energy_density_reference"]) * 100.0
    ).clip(lower=0.0, upper=100.0)

    macro_score = safe_mean(pd.concat([protein_pct, fiber_pct], axis=1))
    micro_score = potassium_pct.fillna(0.0)
    penalty_score = safe_mean(
        pd.concat(
            [
                added_sugar_penalty,
                sodium_penalty,
                sat_fat_penalty,
                energy_density_penalty,
            ],
            axis=1,
        )
    )

    score = (
        weights["macros"] * macro_score
        + weights["micros"] * micro_score
        - weights["penalties"] * penalty_score
    )
    score = score.clip(lower=0.0, upper=100.0)
    dataset["macro_score"] = macro_score
    dataset["micro_score"] = micro_score
    dataset["penalty_score"] = penalty_score
    dataset["nutrition_score"] = score
    dataset["nutrition_score"] = dataset["nutrition_score"].fillna(0.0)

    cutpoints = config["grade_cutpoints"]
    dataset["grade"] = pd.cut(
        dataset["nutrition_score"],
        bins=[-np.inf, cutpoints["D"], cutpoints["C"], cutpoints["B"], cutpoints["A"], np.inf],
        labels=["E", "D", "C", "B", "A"],
        right=False,
    )
    dataset["grade"] = dataset["grade"].astype(str).replace({"nan": "E"})
    dataset["score"] = dataset["nutrition_score"]
    return dataset


def write_sanity_checks(dataset: pd.DataFrame, processed_dir: Path) -> None:
    if dataset.empty:
        return
    top = dataset.nlargest(50, "nutrition_score")
    bottom = dataset.nsmallest(50, "nutrition_score")
    sanity = pd.concat([top.assign(rank_type="top"), bottom.assign(rank_type="bottom")])
    sanity_cols = [
        "rank_type",
        "fdc_id",
        "description",
        "grade",
        "nutrition_score",
        "kcal_per_serv",
        "sodium_mg_perserving",
        "sugar_g_perserving",
    ]
    sanity.to_csv(processed_dir / "score_sanity_sample.csv", index=False, columns=sanity_cols)


def run_scoring(paths_override: Optional[Dict[str, str]] = None) -> None:
    paths = load_paths().copy()
    if paths_override:
        paths.update(
            {key: str(Path(value)) for key, value in paths_override.items() if value is not None}
        )
    processed_dir = Path(paths["processed_dir"])
    configs_dir = project_root() / "configs"
    dataset_path = processed_dir / "foods_nutrients.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cannot find {dataset_path}. Run build_dataset.py before scoring."
        )
    with timed_step("Loading dataset"):
        dataset = pd.read_parquet(dataset_path)
    with timed_step("Loading scoring config"):
        config = load_scoring_config(configs_dir / "scoring.yml")
    with timed_step("Computing nutrition scores"):
        dataset = compute_scores(dataset, config)
    with timed_step("Saving scored dataset"):
        dataset.to_parquet(dataset_path, index=False)
    with timed_step("Writing sanity checks"):
        write_sanity_checks(dataset, processed_dir)
    LOGGER.info("Scoring complete; updated dataset saved to %s", dataset_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.processed_dir:
        overrides["processed_dir"] = str(args.processed_dir)
    run_scoring(overrides if overrides else None)


if __name__ == "__main__":
    main()
