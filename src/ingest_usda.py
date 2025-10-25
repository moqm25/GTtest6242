"""Stage USDA FoodData Central CSVs into slimmer parquet tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import load_paths
from .progress import progress_for_iterable, timed_step


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

RENAME: Dict[str, Dict[str, str]] = {
    "foundation_food.csv": {"NDB_number": "ndb_number"},
    "sr_legacy_food.csv": {"NDB_number": "ndb_number"},
    "food_portion.csv": {"modifier": "gram_weight_modifier"},
    "wweia_food_category.csv": {"wweia_food_category": "wweia_food_category_id"},
}

REQUIRED: Dict[str, Iterable[str]] = {
    "food.csv": [
        "fdc_id",
        "data_type",
        "description",
        "food_category_id",
        "publication_date",
    ],
    "foundation_food.csv": ["fdc_id", "ndb_number", "footnote"],
    "sr_legacy_food.csv": ["fdc_id", "ndb_number"],
    "branded_food.csv": [
        "fdc_id",
        "brand_owner",
        "brand_name",
        "subbrand_name",
        "gtin_upc",
        "ingredients",
        "not_a_significant_source_of",
        "serving_size",
        "serving_size_unit",
        "household_serving_fulltext",
        "branded_food_category",
        "data_source",
        "package_weight",
        "modified_date",
        "available_date",
        "market_country",
        "discontinued_date",
        "preparation_state_code",
        "trade_channel",
        "short_description",
        "material_code",
    ],
    "food_nutrient.csv": [
        "id",
        "fdc_id",
        "nutrient_id",
        "amount",
        "data_points",
        "derivation_id",
        "min",
        "max",
        "median",
        "loq",
        "footnote",
        "min_year_acquired",
        "percent_daily_value",
    ],
    "nutrient.csv": ["id", "name", "unit_name", "nutrient_nbr", "rank"],
    "food_portion.csv": [
        "id",
        "fdc_id",
        "seq_num",
        "amount",
        "measure_unit_id",
        "portion_description",
        "gram_weight_modifier",
        "gram_weight",
        "data_points",
        "footnote",
        "min_year_acquired",
    ],
    "measure_unit.csv": ["id", "name"],
    "food_category.csv": ["id", "code", "description"],
    "wweia_food_category.csv": [
        "wweia_food_category_id",
        "wweia_food_category_description",
    ],
}

EXTRAS: Dict[str, Iterable[str]] = {}

PA_SCHEMAS: Dict[str, pa.Schema] = {
    "food_nutrient.csv": pa.schema(
        [
            ("id", pa.string()),
            ("fdc_id", pa.string()),
            ("nutrient_id", pa.string()),
            ("amount", pa.string()),
            ("data_points", pa.string()),
            ("derivation_id", pa.string()),
            ("min", pa.string()),
            ("max", pa.string()),
            ("median", pa.string()),
            ("loq", pa.string()),
            ("footnote", pa.string()),
            ("min_year_acquired", pa.string()),
            ("percent_daily_value", pa.string()),
        ]
    ),
}


def normalize_and_validate(frame: pd.DataFrame, file_name: str) -> pd.DataFrame:
    df = frame.rename(columns=RENAME.get(file_name, {}))
    required = set(REQUIRED[file_name])
    have = set(df.columns)
    missing = required - have
    if missing:
        raise ValueError(
            f"{file_name}: missing required columns after rename: {sorted(missing)}; "
            f"present={sorted(df.columns)}"
        )
    keep = list(required) + [c for c in EXTRAS.get(file_name, []) if c not in required]
    if keep:
        # If a declared Arrow schema exists for this file, honor its column order.
        schema = PA_SCHEMAS.get(file_name)
        if schema:
            ordered = [name for name in schema.names if name in df.columns]
            return df[ordered]
        # Otherwise, preserve the CSV header order while selecting only desired columns.
        present_keep = [c for c in df.columns if c in keep]
        return df[present_keep]
    return df


def _write_parquet_chunked(
    csv_path: Path,
    output_path: Path,
    dtype: dict[str, str] | None = None,
) -> None:
    del dtype  # chunked ingest enforces string schema regardless of requested dtypes
    file_name = csv_path.name
    output_path.unlink(missing_ok=True)

    reader = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        chunksize=200_000,
        low_memory=False,
    )

    target_schema = PA_SCHEMAS.get(file_name)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    chunk_count = 0
    try:
        for chunk in progress_for_iterable(reader, f"Reading {file_name} chunks"):
            chunk_count += 1
            chunk = normalize_and_validate(chunk, file_name)
            for column in REQUIRED[file_name]:
                if column not in chunk.columns:
                    chunk[column] = ""
            chunk = chunk.fillna("")
            chunk = chunk.astype({column: "string" for column in chunk.columns})
            # If we have a target schema, reorder columns to match it before creating the table.
            if target_schema is not None:
                ordered_cols = [name for name in target_schema.names if name in chunk.columns]
                chunk = chunk[ordered_cols]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if target_schema is None:
                # Build a schema that matches the incoming table (all strings)
                target_schema = pa.schema([(field.name, pa.string()) for field in table.schema])
            # Cast is safe now that names and order are aligned
            table = table.cast(target_schema)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), schema=target_schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    LOGGER.info("Wrote %s rows (%s chunks) to %s", total_rows, chunk_count, output_path)


def _write_parquet(
    csv_path: Path,
    output_path: Path,
    *,
    dtype: Optional[Dict[str, str]] = None,
    chunked: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if chunked:
        _write_parquet_chunked(csv_path, output_path, dtype=dtype)
        return
    read_dtype: object = dtype if dtype is not None else str
    frame = pd.read_csv(
        csv_path,
        dtype=read_dtype,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    frame = normalize_and_validate(frame, csv_path.name)
    for column in REQUIRED[csv_path.name]:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.fillna("")
    frame = frame.astype({column: "string" for column in frame.columns})
    # Align column order to the target schema if available before creating Arrow table
    schema = PA_SCHEMAS.get(csv_path.name)
    if schema is not None:
        ordered_cols = [n for n in schema.names if n in frame.columns]
        frame = frame[ordered_cols]
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if schema is not None:
        table = table.cast(schema)
    pq.write_table(table, str(output_path), compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(frame), output_path)


def ingest(paths_override: Optional[Dict[str, str]] = None) -> None:
    paths = load_paths().copy()
    if paths_override:
        paths.update(
            {key: str(Path(value)) for key, value in paths_override.items() if value}
        )
    data_root = Path(paths["data_root"])
    staging_dir = Path(paths["staging_dir"])
    staging_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "food.csv": {},
        "foundation_food.csv": {},
        "sr_legacy_food.csv": {},
        "branded_food.csv": {},
        "food_nutrient.csv": {"chunked": True},
        "nutrient.csv": {},
        "food_portion.csv": {},
        "measure_unit.csv": {},
        "food_category.csv": {},
        "wweia_food_category.csv": {},
    }

    cached_frames: Dict[str, pd.DataFrame] = {}
    for filename, options in specs.items():
        csv_path = data_root / filename
        if not csv_path.exists():
            LOGGER.warning("Skipping %s because it was not found.", filename)
            continue
        output_path = staging_dir / (filename.replace(".csv", ".parquet"))
        with timed_step(f"Staging {filename}"):
            _write_parquet(
                csv_path,
                output_path,
                dtype=options.get("dtype"),
                chunked=options.get("chunked", False),
            )
        cached_frames[filename] = pd.read_parquet(output_path)
    LOGGER.info("Ingest complete for %d tables.", len(cached_frames))

    # Perform nutrient join validation
    try:
        nutrients = cached_frames.get("nutrient.csv")
        food_nutrient = cached_frames.get("food_nutrient.csv")
        if nutrients is not None and food_nutrient is not None:
            enriched = food_nutrient.merge(
                nutrients[["id", "unit_name"]],
                left_on="nutrient_id",
                right_on="id",
                how="left",
                validate="many_to_one",
            )
            missing_unit = enriched["unit_name"].isna().mean()
            if missing_unit > 0.02:
                raise ValueError(
                    "food_nutrient.csv: unit_name missing for more than 2% of rows after join. "
                    "Check nutrient_id coverage."
                )
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Missing required nutrient columns during validation: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional override for the raw DATA directory.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Optional override for the staging parquet directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.data_root:
        overrides["data_root"] = str(args.data_root)
    if args.staging_dir:
        overrides["staging_dir"] = str(args.staging_dir)
    ingest(overrides if overrides else None)


if __name__ == "__main__":
    main()
