# ---- standard header for app/*.py ----
from __future__ import annotations

from pathlib import Path
import sys

# Project root = repo folder (parent of /app)
ROOT = Path(__file__).resolve().parents[1]

# Ensure absolute imports work no matter how Streamlit is launched
if (p := str(ROOT)) not in sys.path:
    sys.path.insert(0, p)
SRC = ROOT / "src"
if (sp := str(SRC)) not in sys.path:
    sys.path.insert(0, sp)

# Convenience paths for configs/data/models (use these instead of relative strings)
CONFIGS = ROOT / "configs"
DATA_DIR = ROOT / "DATA"
MODELS_DIR = ROOT / "models"

# Optional: load .env if present (safe if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass
# ---- end header ----
# 

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import load_service_or_error, page_link_button, render_nav, safe_serving_value

FRIENDLY_LABELS = {
    "energy_kcal": "Calories",
    "protein_g": "Protein",
    "carbs_g": "Carbs",
    "fiber_g": "Fiber",
    "sugar_g": "Sugar",
    "added_sugar_g": "Added Sugar",
    "sodium_mg": "Sodium",
    "sat_fat_g": "Saturated Fat",
}

GRADE_STEPS = [
    (0, 50, "#f8b4b4"),
    (50, 65, "#ffd9a8"),
    (65, 75, "#fff1a8"),
    (75, 85, "#d3efb6"),
    (85, 100, "#9fd8a8"),
]


def _get_query_fdc_id(dataset: pd.DataFrame) -> int | None:
    try:
        params = st.query_params  # type: ignore[attr-defined]
    except AttributeError:
        params = st.experimental_get_query_params()  # type: ignore[attr-defined]
    raw = params.get("fdc_id") if params else None
    if not raw:
        return None
    if isinstance(raw, list):
        raw = raw[0]
    try:
        fdc_id = int(raw)
    except (TypeError, ValueError):
        return None
    if fdc_id in dataset["fdc_id"].values:
        return fdc_id
    return None


def _gauge(score: float, grade: str) -> go.Figure:
    indicator = go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Score ({grade})"},
        number={"suffix": ""},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#238636"},
            "steps": [{"range": step[:2], "color": step[2]} for step in GRADE_STEPS],
        },
    )
    fig = go.Figure(indicator)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _nutrient_chart(labels: List[str], values: List[float], title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#2c7fb8"))
    fig.update_layout(
        title=title,
        xaxis={"tickangle": -30},
        yaxis_title="Amount",
        height=360,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    return fig


def _prepare_nutrients(row: pd.Series, service, suffix: str) -> Dict[str, float]:
    cols = service.nutrient_cols(suffix)
    data: Dict[str, float] = {}
    for column in cols:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        base = column.replace("_per100g", "").replace("_perserving", "")
        label = FRIENDLY_LABELS.get(base, base.replace("_", " ").title())
        data[label] = float(value)
    return data


def _driver_text(drivers: Dict[str, List[str]]) -> str:
    if not drivers:
        return "No explanation available."
    positives = ", ".join(drivers.get("positive", [])) or "n/a"
    negatives = ", ".join(drivers.get("negative", [])) or "n/a"
    return f"**Top positives:** {positives}  \n**Watch outs:** {negatives}"


def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Food Detail")
    render_nav("Detail")
    service = load_service_or_error()
    dataset = service.dataset

    fdc_id = _get_query_fdc_id(dataset)
    if fdc_id is None:
        options = dataset[["fdc_id", "description"]].copy()
        options["label"] = options.apply(lambda r: f"{r['description']} ({r['fdc_id']})", axis=1)
        label = st.selectbox("Choose a food", options["label"])
        fdc_id = int(label.split("(")[-1].strip(")"))

    row = dataset.loc[dataset["fdc_id"] == fdc_id]
    if row.empty:
        st.error("Food not found in dataset.")
        return
    food = row.iloc[0]
    score_col = service.score_column
    score = food.get(score_col, np.nan)
    grade = food.get("grade", "—") or "—"

    col_left, col_right = st.columns([1.6, 1])
    with col_left:
        st.plotly_chart(
            _gauge(float(score) if pd.notna(score) else 0.0, str(grade)),
            use_container_width=True,
        )
        per_choice = st.radio(
            "Show nutrients",
            options=("per serving", "per 100g"),
            index=0,
            horizontal=True,
        )
        suffix = "perserving" if per_choice == "per serving" else "per100g"
        nutrients = _prepare_nutrients(food, service, suffix)
        if nutrients:
            st.plotly_chart(
                _nutrient_chart(
                    list(nutrients.keys()), list(nutrients.values()), per_choice.title()
                ),
                use_container_width=True,
            )
        else:
            st.info("Nutrient values unavailable for this view.")

    with col_right:
        st.subheader("At a glance")
        badges = service.format_badges(food)
        bcols = st.columns(2)
        bcols[0].metric("Grade", badges["grade"])
        bcols[1].metric("Score", badges["score"])
        st.metric("kcal / serving", badges["kcal_serv"])
        st.metric("Sodium / serving", badges["sodium_serv"])
        st.metric("Sugar / serving", badges["sugar_serv"])

        compare_ids = st.session_state.setdefault("compare_ids", set())
        if st.button("Add to Compare", disabled=len(compare_ids) >= 3 and fdc_id not in compare_ids):
            compare_ids.add(fdc_id)
        st.page_link("Compare.py", label="Open compare →")
        page_link_button("Substitute.py", label="Find substitutes →", params={"fdc_id": fdc_id})

        st.subheader("Grade probabilities")
        prediction = service.predict_probs(fdc_id)
        probabilities = prediction.get("probabilities") if prediction else {}
        if probabilities:
            prob_cols = [col for col in probabilities if col.startswith("prob_")]
            prob_cols.sort()
            for col in prob_cols:
                label = col.replace("prob_", "").upper()
                st.write(f"{label}: {probabilities[col]*100:.1f}%")
        else:
            st.info("Model probabilities unavailable.")

        st.subheader("Top drivers")
        drivers = prediction.get("drivers") if prediction else {}
        st.markdown(_driver_text(drivers))

    st.divider()
    serving_desc = food.get("serving_desc")
    serving_weight = safe_serving_value(food, "serving_gram_weight", unit="g", decimals=0)
    st.caption(
        f"Serving: {serving_desc or '—'} | Serving weight: {serving_weight}"
    )
    st.page_link(
        "Search.py",
        label="← Back to search results",
    )


if __name__ == "__main__":
    main()
