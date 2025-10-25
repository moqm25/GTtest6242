# Model Card

## Overview

The nutrition scoring system combines a rule-based score with two lightweight machine
learning components:

- **Calibration** – Multinomial logistic regression on the rule-based score to
  generate grade probabilities and a combined `prob_AB` metric.
- **Interpretability** – Elastic Net regression on per-100g nutrient features to
  highlight the factors that most influence the rule-based score (positive and
  negative drivers).

## Data

- Source: USDA FoodData Central (Foundation, SR Legacy, Branded, WWEIA).
- Features: Per 100g nutrient amounts standardized via the canonical mapping defined in
  `src/utils_nutrients.py`.
- Labels: Rule-based grade for each food (A–E).
- Serving calculations: primary portion chosen from `food_portion.csv`, with fallbacks
  to 100 g if missing.

## Training

- Python 3.13, scikit-learn 1.4.
- 5-fold cross-validation for both logistic regression (log-loss) and Elastic Net (RMSE).
- Models stored in `models/calibration.joblib` and `models/elastic_net.joblib`.
- Coefficients exported to `models/coef.json`.

## Intended Use

- Interactive exploration via the Streamlit app.
- Generating grade probabilities and substitution recommendations in downstream tools.

## Limitations

- Relies solely on nutrient composition; does not consider additives, allergens, or
  overall dietary patterns.
- Elastic Net coefficients are on standardized features; raw magnitude comparisons
  should be interpreted qualitatively.
- Missing added-sugar values fall back to total sugar estimates, potentially inflating
  penalties for some items.

## Ethical & Practical Considerations

- Scores are not medical advice; they are intended for comparative guidance.
- Recommender respects sodium and calorie constraints but does not track user-specific
  dietary restrictions.
- Branded food data quality varies; some entries may lack portions or nutrients.
