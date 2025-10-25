# Nutrition Scoring Overview

The rule-based score implemented in `src/scoring.py` follows the specification provided
in `configs/scoring.yml`. The pipeline assumes nutrients are available in
`data/processed/foods_nutrients.parquet` with both per-100g and per-serving fields.

## Sub-scores

- **Macros (weight 0.45)** – average of the Daily Value (DV) percentages for
  `protein_g_perserving` and `fiber_g_perserving`, capped at 100.
- **Micros (weight 0.35)** – DV percentage for `potassium_mg_perserving`, capped at 100.
- **Penalties (weight 0.20)** – average of the capped penalty metrics for added sugar,
  sodium, saturated fat, and energy density. Added sugar falls back to 75% of total sugar
  if the added sugar nutrient is missing.

The final score is:

```
score = macros - penalties + micros
score = clip(score, 0, 100)
```

## Daily values

The reference DV values (grams per serving unless noted) are defined in
`configs/scoring.yml`. They can be tuned without changing code.

## Grades

Letter grades are assigned using the following cutpoints:

| Grade | Score range |
| ----- | ----------- |
| A     | ≥ 85        |
| B     | 75–84       |
| C     | 65–74       |
| D     | 50–64       |
| E     | < 50        |

The scoring script stores the numeric score (`nutrition_score`) and the grade (`grade`)
back into `foods_nutrients.parquet` and writes a `score_sanity_sample.csv` report of top
and bottom items for quick inspection.
