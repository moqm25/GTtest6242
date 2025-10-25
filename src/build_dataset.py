"""Construct the canonical foods_nutrients parquet dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, List

import numpy as np
import pandas as pd

from .config import load_paths
from .progress import timed_step
from .utils_nutrients import (
    CANONICAL_ORDER,
    NutrientSelection,
    convert_value,
    get_nutrient_mapping,
    robust_zscore,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

LIQUID_KEYWORDS = [
    "milk",
    "juice",
    "smoothie",
    "water",
    "soda",
    "tea",
    "coffee",
    "soup",
    "broth",
    "drink",
    "yogurt drink",
    "oil",
]


def _resolve_csv_usecols(csv_path: Path, requested: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not requested:
        return None
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except FileNotFoundError:
        return list(requested)
    available = set(header.columns.tolist())
    resolved = [col for col in requested if col in available]
    missing = [col for col in requested if col not in available]
    if missing:
        LOGGER.warning(
            "Skipping missing columns in %s: %s", csv_path.name, ", ".join(missing)
        )
    return resolved or None


def load_table(
    data_root: Path,
    staging_dir: Path,
    filename: str,
    *,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    parquet_path = staging_dir / filename.replace(".csv", ".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to read staging parquet %s (%s); falling back to CSV.",
                parquet_path,
                exc,
            )
    csv_path = data_root / filename
    if not csv_path.exists():
        LOGGER.warning("Missing %s; returning empty frame.", filename)
        return pd.DataFrame(columns=usecols if usecols else [])
    resolved = _resolve_csv_usecols(csv_path, usecols)
    return pd.read_csv(csv_path, usecols=resolved, low_memory=False)


def infer_form(
    description: str, food_category: str, wweia_category: str, overrides: Dict[int, str], fdc_id: int
) -> str:
    if fdc_id in overrides:
        return overrides[fdc_id]
    parts = []
    for value in (description, food_category, wweia_category):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    text = " ".join(parts).lower()
    for keyword in LIQUID_KEYWORDS:
        if keyword in text:
            return "liquid"
    return "solid"


def load_form_overrides(dictionary_dir: Path) -> Dict[int, str]:
    overrides_path = dictionary_dir / "form_overrides.csv"
    if not overrides_path.exists():
        return {}
    frame = pd.read_csv(overrides_path)
    frame["form"] = frame["form"].str.lower()
    return {int(row["fdc_id"]): str(row["form"]) for _, row in frame.iterrows()}


def select_primary_portions(
    food_portion: pd.DataFrame, measure_unit: pd.DataFrame
) -> pd.DataFrame:
    if food_portion.empty:
        return pd.DataFrame(columns=["fdc_id", "serving_desc", "serving_gram_weight"])
    unit_lookup = measure_unit.set_index("id")["name"].to_dict()
    portions = food_portion.copy()
    if "modifier" not in portions.columns and "gram_weight_modifier" in portions.columns:
        portions = portions.rename(columns={"gram_weight_modifier": "modifier"})
    # --- Diagnostics: show incoming columns ---
    LOGGER.info("food_portion columns: %s", list(portions.columns))
    expected_cols = ["fdc_id", "portion_description", "modifier", "measure_unit_id", "gram_weight"]
    missing_cols = [c for c in expected_cols if c not in portions.columns]
    if missing_cols:
        LOGGER.warning("food_portion missing columns: %s; synthesizing defaults.", ", ".join(missing_cols))
    # ---- Guard against missing columns from source/staged files ----
    # Some USDA releases or staged parquet may omit these columns.
    if "portion_description" not in portions.columns:
        portions["portion_description"] = ""
    if "modifier" not in portions.columns:
        portions["modifier"] = ""
    if "measure_unit_id" not in portions.columns:
        portions["measure_unit_id"] = np.nan
    if "gram_weight" not in portions.columns:
        portions["gram_weight"] = np.nan
    portions["unit_name"] = portions["measure_unit_id"].map(unit_lookup) if "measure_unit_id" in portions.columns else ""
    # Prefer portion_description; fall back to modifier; finally unit_name
    _portion_desc = portions["portion_description"] if "portion_description" in portions.columns else ""
    _modifier = portions["modifier"] if "modifier" in portions.columns else ""
    portions["serving_desc"] = (
        pd.Series(_portion_desc, index=portions.index).fillna("")
        .where(pd.Series(_portion_desc, index=portions.index).notna(),
               pd.Series(_modifier, index=portions.index).fillna(""))
    )
    portions.loc[portions["serving_desc"] == "", "serving_desc"] = portions.loc[
        portions["serving_desc"] == "", "unit_name"
    ].fillna("")
    portions["has_text"] = portions["serving_desc"].astype(str).str.len() > 0
    portions["gram_weight"] = pd.to_numeric(portions["gram_weight"], errors="coerce").fillna(0.0)

    # --- Diagnostics: post-normalization summary ---
    n_rows = len(portions)
    n_text = int(portions["has_text"].sum())
    n_zero_wt = int((portions["gram_weight"] == 0).sum())
    LOGGER.info(
        "food_portion normalized: rows=%d, with_text=%d (%.1f%%), zero_weight=%d",
        n_rows, n_text, (100.0 * n_text / max(n_rows, 1)), n_zero_wt
    )

    portions.sort_values(
        by=["fdc_id", "has_text", "gram_weight"], ascending=[True, False, False], inplace=True
    )
    primary = portions.drop_duplicates(subset=["fdc_id"], keep="first")
    LOGGER.info("primary serving rows: %d (unique fdc_id)", len(primary))
    return primary[["fdc_id", "serving_desc", "gram_weight"]].rename(
        columns={"gram_weight": "serving_gram_weight"}
    )


def compute_nutrient_matrix(
    food_nutrient: pd.DataFrame,
    mapping: Dict[str, NutrientSelection],
    nutrients: pd.DataFrame,
) -> pd.DataFrame:
    # -- normalize key dtypes from ingest (often stored as strings) --
    food_nutrient = food_nutrient.copy()
    food_nutrient["nutrient_id"] = pd.to_numeric(food_nutrient["nutrient_id"], errors="coerce")
    food_nutrient["fdc_id"] = pd.to_numeric(food_nutrient["fdc_id"], errors="coerce")
    food_nutrient["amount"] = pd.to_numeric(food_nutrient["amount"], errors="coerce")
    # drop rows with missing keys or amount after coercion
    food_nutrient = food_nutrient[
        food_nutrient["nutrient_id"].notna() & food_nutrient["fdc_id"].notna()
    ]
    food_nutrient["nutrient_id"] = food_nutrient["nutrient_id"].astype("int64")
    food_nutrient["fdc_id"] = food_nutrient["fdc_id"].astype("int64")

    if food_nutrient.empty:
        raise ValueError("food_nutrient table is empty; cannot build dataset.")
    id_to_canonical = {selection.nutrient_id: name for name, selection in mapping.items()}
    subset = food_nutrient[food_nutrient["nutrient_id"].isin(id_to_canonical.keys())].copy()
    if subset.empty:
        raise ValueError("No nutrient rows matched canonical nutrient ids.")
    subset["canonical_name"] = subset["nutrient_id"].map(id_to_canonical)
    # Coerce nutrients.id to int for a clean join
    _nutr = nutrients[["id", "unit_name"]].copy()
    _nutr["id"] = pd.to_numeric(_nutr["id"], errors="coerce")
    _nutr = _nutr[_nutr["id"].notna()]
    _nutr["id"] = _nutr["id"].astype("int64")
    subset = subset.merge(
        _nutr,
        left_on="nutrient_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )

    missing_unit = subset["unit_name"].isna().mean()
    if missing_unit > 0.02:
        raise ValueError(
            f"unit_name missing for {missing_unit:.2%} of nutrient rows; check coverage in nutrient.csv"
        )

    def _convert_row(row: pd.Series) -> float:
        target_unit = mapping[row["canonical_name"]].unit_name
        unit_name = row.get("unit_name")
        amount = row["amount"]
        if pd.isna(amount):
            return np.nan
        try:
            return convert_value(float(amount), unit_name, target_unit)
        except ValueError:
            # Conversion not required or unsupported; keep original value
            return float(amount)

    subset["amount_target_unit"] = subset.apply(_convert_row, axis=1)
    subset = subset.drop_duplicates(subset=["fdc_id", "canonical_name", "amount_target_unit"])
    aggregated = subset.groupby(["fdc_id", "canonical_name"])["amount_target_unit"].mean()
    aggregated.replace([np.inf, -np.inf], np.nan, inplace=True)
    aggregated = aggregated.reset_index()
    pivot = aggregated.pivot(index="fdc_id", columns="canonical_name", values="amount_target_unit")
    pivot = pivot.reindex(columns=CANONICAL_ORDER)
    pivot.columns = [f"{col}_per100g" for col in pivot.columns]
    # Ensure fdc_id is a column for downstream merges
    pivot = pivot.reset_index()
    # enforce stable dtype
    pivot["fdc_id"] = pd.to_numeric(pivot["fdc_id"], errors="raise").astype("int64")
    return pivot


def enrich_with_servings(dataset: pd.DataFrame, servings: pd.DataFrame) -> pd.DataFrame:
    merged = dataset.merge(servings, how="left", on="fdc_id")
    merged["serving_gram_weight"] = merged["serving_gram_weight"].fillna(100.0)
    merged["serving_desc"] = merged["serving_desc"].fillna("100 g")
    branded_text = merged.get("household_serving_fulltext")
    if branded_text is not None:
        mask = merged["serving_desc"].eq("100 g") & branded_text.notna()
        merged.loc[mask, "serving_desc"] = branded_text[mask]
    return merged


def add_per_serving_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    gram_weight = dataset["serving_gram_weight"].replace({0: np.nan})
    for canonical in CANONICAL_ORDER:
        per100g_col = f"{canonical}_per100g"
        if per100g_col not in dataset.columns:
            continue
        per_serv_col = f"{canonical}_perserving"
        dataset[per_serv_col] = dataset[per100g_col] * (gram_weight / 100.0)
    dataset["kcal_per_serv"] = dataset["energy_kcal_perserving"]
    dataset["kcal_per100g"] = dataset["energy_kcal_per100g"]
    return dataset


def add_missingness_flags(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["added_sugar_missing"] = dataset["added_sugar_g_per100g"].isna().astype(int)
    return dataset


def attach_form(dataset: pd.DataFrame, overrides: Dict[int, str]) -> pd.DataFrame:
    dataset["form"] = dataset.apply(
        lambda row: infer_form(
            description=row.get("description", ""),
            food_category=row.get("food_category", ""),
            wweia_category=row.get("wweia_category", ""),
            overrides=overrides,
            fdc_id=row["fdc_id"],
        ),
        axis=1,
    )
    return dataset


def compute_robust_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    for canonical in CANONICAL_ORDER:
        per100g_col = f"{canonical}_per100g"
        if per100g_col not in dataset.columns:
            continue
        z_col = f"{canonical}_per100g_z"
        dataset[z_col] = robust_zscore(dataset[per100g_col], clip=None)
        clipped_col = f"{canonical}_per100g_clipped"
        dataset[clipped_col] = dataset[z_col].clip(-2.5, 2.5)
    return dataset


def compute_missingness_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in dataset.columns if col.endswith("_per100g")]
    records = []
    total = len(dataset)
    for col in columns:
        missing_fraction = float(dataset[col].isna().sum()) / max(total, 1)
        records.append({"column": col, "missing_fraction": missing_fraction})
    return pd.DataFrame(records)


def compute_nutrient_stats(dataset: pd.DataFrame) -> pd.DataFrame:
    records = []
    for canonical in CANONICAL_ORDER:
        col = f"{canonical}_per100g"
        if col not in dataset.columns:
            continue
        series = dataset[col].dropna()
        if series.empty:
            continue
        records.append(
            {
                "nutrient": canonical,
                "mean": series.mean(),
                "p10": series.quantile(0.10),
                "p90": series.quantile(0.90),
                "median": series.median(),
                "mad": (series - series.median()).abs().median(),
            }
        )
    return pd.DataFrame(records)


def generate_food_groups(dataset: pd.DataFrame, output_path: Path) -> None:
    food_col = (
        dataset["food_category"] if "food_category" in dataset.columns else pd.Series("Unknown", index=dataset.index)
    )
    wweia_col = (
        dataset["wweia_category"] if "wweia_category" in dataset.columns else pd.Series("Unknown", index=dataset.index)
    )
    frame = pd.DataFrame(
        {
            "food_category": food_col.fillna("Unknown"),
            "wweia_category": wweia_col.fillna("Unknown"),
        }
    )
    groups = frame.value_counts().reset_index(name="count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output_path, index=False)


def write_schema_doc(dataset: pd.DataFrame, mapping: Dict[str, NutrientSelection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# foods_nutrients.parquet schema", ""]
    lines.append("## Columns")
    lines.append("")
    lines.append("| column | dtype |")
    lines.append("| --- | --- |")
    for column, dtype in dataset.dtypes.items():
        lines.append(f"| {column} | {dtype} |")
    lines.append("")
    lines.append("## Nutrient mapping")
    lines.append("")
    lines.append("| canonical | nutrient_id | nutrient_name | unit |")
    lines.append("| --- | --- | --- | --- |")
    for canonical, selection in mapping.items():
        lines.append(
            f"| {canonical} | {selection.nutrient_id} | {selection.nutrient_name} | {selection.unit_name} |"
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_dataset(paths_override: Optional[Dict[str, str]] = None) -> None:
    paths = load_paths().copy()
    if paths_override:
        paths.update(
            {key: str(Path(value)) for key, value in paths_override.items() if value is not None}
        )
    data_root = Path(paths["data_root"])
    staging_dir = Path(paths["staging_dir"])
    processed_dir = Path(paths["processed_dir"])
    dictionary_dir = Path(paths["dictionaries_dir"])

    with timed_step("Loading nutrient mapping"):
        mapping = get_nutrient_mapping()
    with timed_step("Loading nutrient table"):
        nutrients = load_table(
            data_root,
            staging_dir,
            "nutrient.csv",
            usecols=["id", "name", "unit_name"],
        )
    # normalize nutrients id dtype early
    if "id" in nutrients.columns:
        nutrients["id"] = pd.to_numeric(nutrients["id"], errors="coerce")

    with timed_step("Loading source tables"):
        food = load_table(
            data_root,
            staging_dir,
            "food.csv",
            usecols=[
                "fdc_id",
                "data_type",
                "description",
                "food_category_id",
                "wweia_food_category_id",
                "publication_date",
            ],
        )
        food_category = load_table(data_root, staging_dir, "food_category.csv")
        wweia_food_category = load_table(
            data_root, staging_dir, "wweia_food_category.csv"
        )
        branded = load_table(
            data_root,
            staging_dir,
            "branded_food.csv",
            usecols=[
                "fdc_id",
                "brand_owner",
                "household_serving_fulltext",
                "serving_size",
                "serving_size_unit",
            ],
        )
        food_nutrient = load_table(
            data_root,
            staging_dir,
            "food_nutrient.csv",
            usecols=[
                "fdc_id",
                "nutrient_id",
                "amount",
            ],
        )
        food_portion = load_table(
            data_root,
            staging_dir,
            "food_portion.csv",
            usecols=[
                "fdc_id",
                "seq_num",
                "measure_unit_id",
                "portion_description",
                "modifier",
                "gram_weight",
            ],
        )
        measure_unit = load_table(
            data_root,
            staging_dir,
            "measure_unit.csv",
            usecols=["id", "name"],
        )

    with timed_step("Preparing metadata joins"):
        if "food_category_id" in food.columns and not pd.api.types.is_numeric_dtype(food["food_category_id"]):
            food = food.rename(columns={"food_category_id": "food_category"})
            food_category = pd.DataFrame()
        if not food_category.empty:
            food_category = food_category.rename(
                columns={"id": "food_category_id", "description": "food_category"}
            )
        if not wweia_food_category.empty:
            wweia_food_category = wweia_food_category.rename(
                columns={"wweia_food_category_description": "wweia_category"}
            )
        metadata = food
        if not food_category.empty and "food_category_id" in metadata.columns:
            metadata = metadata.merge(food_category, on="food_category_id", how="left")
        if "wweia_food_category_id" not in metadata.columns or wweia_food_category.empty:
            metadata = metadata
        else:
            metadata = metadata.merge(wweia_food_category, on="wweia_food_category_id", how="left")
        if "wweia_category" not in metadata.columns:
            metadata["wweia_category"] = pd.NA
        metadata = metadata.merge(branded, on="fdc_id", how="left")

    with timed_step("Building nutrient matrix"):
        nutrient_matrix = compute_nutrient_matrix(food_nutrient, mapping, nutrients)

    with timed_step("Selecting primary servings"):
        servings = select_primary_portions(food_portion, measure_unit)

    with timed_step("Combining tables"):
        metadata["fdc_id"] = pd.to_numeric(metadata["fdc_id"], errors="raise").astype("int64")
        nutrient_matrix["fdc_id"] = pd.to_numeric(nutrient_matrix["fdc_id"], errors="raise").astype("int64")
        if not servings.empty and "fdc_id" in servings.columns:
            servings["fdc_id"] = pd.to_numeric(servings["fdc_id"], errors="raise").astype("int64")
        dataset = metadata.merge(nutrient_matrix, on="fdc_id", how="left")
        dataset = enrich_with_servings(dataset, servings)

    with timed_step("Computing derived columns"):
        dataset = add_per_serving_columns(dataset)
        dataset = add_missingness_flags(dataset)
        overrides = load_form_overrides(dictionary_dir)
        dataset = attach_form(dataset, overrides)
        dataset = compute_robust_columns(dataset)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with timed_step("Writing foods_nutrients.parquet"):
        dataset.to_parquet(processed_dir / "foods_nutrients.parquet", index=False)

    with timed_step("Writing foods_nutrients.csv"):
        dataset.to_csv(processed_dir / "foods_nutrients.csv", index=False)
    LOGGER.info("Dataset rows: %s", f"{len(dataset):,}")

    with timed_step("Computing diagnostics"):
        missingness = compute_missingness_matrix(dataset)
        missingness.to_parquet(processed_dir / "missingness.parquet", index=False)

        stats = compute_nutrient_stats(dataset)
        stats.to_parquet(processed_dir / "nutrient_stats.parquet", index=False)

    with timed_step("Writing dictionaries and docs"):
        generate_food_groups(dataset, dictionary_dir / "food_groups.csv")
        write_schema_doc(dataset, mapping, Path(paths["docs_dir"]) / "schema.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.data_root:
        overrides["data_root"] = str(args.data_root)
    if args.staging_dir:
        overrides["staging_dir"] = str(args.staging_dir)
    if args.processed_dir:
        overrides["processed_dir"] = str(args.processed_dir)
    build_dataset(overrides if overrides else None)


if __name__ == "__main__":
    main()
