# Nutrition Scoring & Recommendation System

## Overview

This repository implements a complete, local-first nutrition analytics stack using the USDA FoodData Central CSV extracts. The pipeline ingests the raw CSVs, standardises nutrients, generates rule-based health scores, calibrates ML models for interpretability, builds substitution recommendations, and exposes the results via automated tests and a Streamlit explorer.

## What’s Included

-   Deterministic ETL from `./DATA` into curated parquet datasets.
-   Rule-based scoring with Daily Value caps and letter grades (v1.0).
-   Logistic calibration and elastic-net interpretability models.
-   Nutrient-embedding recommender with constraint-aware substitutions.
-   Multipage Streamlit UI for search, detail, comparison, and swaps.
-   Automated tests, linting, formatting, and reproducible Make targets.

---

Hi! Moiez here, if yall don't wanna read through the entirety of this readme (which i understand why), do the following:

-   First run this simple script: `./scripts/bootstrap.sh`
    -   Follow the instructions in there - it include file paths to make it simpler for yall
-   Step 4 says this but you need to run the following commands:
    -   `make ingest`
    -   `make build`
    -   `make score`
    -   `make ml`
    -   `make reco`
-   Step 5, run this `make app-smoke`

and that should (hopefully) open the page

---

## Setup

Python 3.13.1 is required. The repository ships with `scripts/bootstrap.sh`, which now prints the manual checklist below:

1. Install Python 3.13.1 (recommended via pyenv):
    ```bash
    pyenv install 3.13.1
    pyenv virtualenv 3.13.1 cse6242-3.13.1
    pyenv activate cse6242-3.13.1
    # or: python3.13 -m venv .venv && source .venv/bin/activate
    ```
2. Upgrade packaging tools and install dependencies:
    ```bash
    pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
    ```
3. Stage the USDA FoodData Central CSV bundle in `./DATA/`:
    ```bash
    USE_RICH=1 python scripts/setup_data.py
    # add --force to refresh existing CSVs
    ```
4. Run the pipeline targets once data is in place:
    ```bash
    make ingest
    make build
    make score
    make ml
    make reco
    make app-smoke
    ```

Helpful tips:

-   `make setup` still runs `env install data`; ensure your virtualenv is active before invoking it so the `pip` commands install into the right interpreter.
-   macOS users can install pyenv and plugins with Homebrew: `brew install pyenv pyenv-virtualenv`.
-   Troubleshooting parquet: if `pyarrow` is unavailable, install `fastparquet` manually, then rerun `make build`.

---

## Data Files

Place the USDA FoodData Central CSV exports inside `./DATA`. Required minimum files:

-   `food.csv`, `foundation_food.csv`, `sr_legacy_food.csv`, `branded_food.csv`
-   `food_nutrient.csv`, `nutrient.csv`
-   `food_portion.csv`, `measure_unit.csv`
-   `food_category.csv`, `wweia_food_category.csv`

Optional CSVs are ignored unless referenced by the pipeline.

## Quick Start

Run the complete workflow via Make:

```bash
make ingest
make build
make score
make ml
make reco
make app-smoke
make test
```

Each target is idempotent and may be re-run safely. `make app-smoke` launches Streamlit in headless mode for a quick import check.

## Health Score v1.0

-   Macros (45% weight): Protein & fibre percent DV capped at 100.
-   Micros (35% weight): Potassium percent DV capped at 100.
-   Penalties (20% weight): Added sugar (or estimated total sugar), sodium, saturated fat, and energy density.
-   Scores are clipped to `[0, 100]` and mapped to grades A–E using the cut points defined in `configs/scoring.yml`.

## Recommendations

-   Nutrient embeddings are built from per-100g nutrient features.
-   Cosine similarity combined with score gain forms the ranking objective.
-   Default constraints include minimum score gain (+10), sodium cap (≤500 mg per serving), kcal delta (≤100), and matching form/category.
-   Fallback swaps surface the highest-scoring items within the same food group.

## App Usage

Launch the Streamlit explorer with:

```bash
streamlit run app/app.py
```

Pages:

-   **Home:** dataset statistics and quick filters.
-   **Search:** fuzzy lookup with nutrient badges and CSV export.
-   **Detail:** score gauge, nutrient charts, and ML driver tips.
-   **Compare:** side-by-side scoring deltas and driver highlights.
-   **Substitute:** substitution candidates with quantified improvements.

## Tests

Execute the full suite with:

```bash
make test
```

Schema, ML monotonicity, recommender constraints, and golden-score checks are included. Formatting may be enforced via `make format`.

## Design Notes & Limits

-   Scores rely solely on nutrient composition—no allergen or ingredient parsing.
-   Added sugar fallbacks estimate penalties using total sugar where necessary.
-   Branded food metadata quality varies; serving inference falls back to 100 g.
-   Recommender constraints are tuned for general guidance, not personalised diets.

## File Map

-   `src/` – Core ETL, scoring, ML, and recommender modules.
-   `app/` – Streamlit pages.
-   `configs/` – YAML configuration for paths, scoring, and recommendations.
-   `data/processed/` – Generated parquet datasets and indices.
-   `models/` – Persisted calibration and interpretability artifacts.
-   `docs/` – Generated documentation and audit outputs.
-   `tests/` – Pytest suites for schema, ML, recommender, and golden checks.
-   `scripts/` – Utility scripts (audit, evaluation, self-check).

## Troubleshooting

-   **Missing parquet files:** Run `make build` followed by `make score` and `make reco`.
-   **Model artifacts missing:** Execute `make ml`.
-   **Streamlit import errors:** Ensure all pipeline steps ran successfully and the virtual environment uses Python 3.13.1.
-   **Tests skipped:** They require generated datasets; run the Make sequence first.
