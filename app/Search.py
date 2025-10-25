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

import math
from typing import Any

import pandas as pd
import streamlit as st

from app.utils import (
    find_items,
    load_service_or_error,
    page_link_button,
    render_nav,
    safe_serving_value,
)


def _init_compare_bucket() -> set[int]:
    bucket = st.session_state.setdefault("compare_ids", set())
    if not isinstance(bucket, set):
        bucket = set(bucket)
        st.session_state["compare_ids"] = bucket
    return bucket


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    form_options = ["All"]
    if "form" in df.columns:
        form_options += sorted(x for x in df["form"].dropna().unique())
    group_options = ["All"]
    if "food_category" in df.columns:
        group_options += sorted(x for x in df["food_category"].dropna().unique())

    with st.sidebar:
        st.header("Filters")
        low_sodium = st.checkbox("Low sodium (≤140mg/100g)", key="filter_low_sodium")
        high_fiber = st.checkbox("High fiber (≥5g/100g)", key="filter_high_fiber")
        form_choice = st.selectbox("Form", options=form_options, key="filter_form")
        group_choice = st.selectbox("Food group", options=group_options, key="filter_group")

    filtered = df.copy()
    if low_sodium and "sodium_mg_per100g" in filtered.columns:
        filtered = filtered[filtered["sodium_mg_per100g"].le(140)]
    if high_fiber and "fiber_g_per100g" in filtered.columns:
        filtered = filtered[filtered["fiber_g_per100g"].ge(5)]
    if form_choice != "All" and "form" in filtered.columns:
        filtered = filtered[filtered["form"] == form_choice]
    if group_choice != "All" and "food_category" in filtered.columns:
        filtered = filtered[filtered["food_category"] == group_choice]
    return filtered


def _render_rows(df: pd.DataFrame, service) -> None:
    bucket = _init_compare_bucket()
    score_col = service.score_column
    page_size = 25
    total = len(df)
    pages = max(1, math.ceil(total / page_size))
    page = 0
    if pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
        ) - 1
    start = page * page_size
    end = start + page_size
    subset = df.iloc[start:end]

    for _, row in subset.iterrows():
        food_id = int(row["fdc_id"])
        cols = st.columns([4, 1.5, 1, 1, 1.3, 1.6, 1.5])
        name = row.get("description", "Unknown item")
        group = row.get("food_category", "—")
        grade = row.get("grade", "—")
        score_val = _format_entry(row.get(score_col))
        kcal = safe_serving_value(row, "energy_kcal_perserving", unit="kcal", decimals=0)
        sodium = safe_serving_value(row, "sodium_mg_perserving", unit="mg", decimals=0)
        sugar = safe_serving_value(row, "sugar_g_perserving", unit="g", decimals=1)
        cols[0].markdown(f"**{name}**  \n`FDC {food_id}`")
        cols[1].markdown(f"Group\n`{group}`")
        cols[2].markdown(f"Grade\n`{grade}`")
        cols[3].markdown(f"Score\n`{score_val}`")
        cols[4].markdown(f"kcal/serv\n`{kcal}`")
        cols[5].markdown(f"Sodium/serv\n`{sodium}`  \nSugar/serv\n`{sugar}`")

        with cols[6]:
            page_link_button("Detail.py", label="Detail →", params={"fdc_id": food_id})
            disabled = len(bucket) >= 3 and food_id not in bucket
            if st.button("Compare +", key=f"compare_add_{food_id}", disabled=disabled):
                bucket.add(food_id)
        st.divider()


def _format_entry(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Search Foods")
    render_nav("Search")
    service = load_service_or_error()
    dataset = service.dataset

    st.markdown(
        "Filter foods by name, brand, form, or nutrient shortcuts. "
        "Add interesting items to your compare basket or jump straight to the detail view."
    )

    query = st.text_input("Search foods or brands…", value="", max_chars=60)

    if len(query) >= 2:
        base = find_items(dataset, query, max_rows=500)
    else:
        base = dataset.nlargest(200, service.score_column)

    filtered = _apply_filters(base)
    filtered = filtered.head(500)
    filtered = filtered.sort_values(service.score_column, ascending=False)

    if filtered.empty:
        message = (
            "Type at least two characters to search." if len(query) < 2 else "No foods matched the filters."
        )
        st.info(message)
        return

    csv_bytes = filtered[
        ["fdc_id", "description", "food_category", service.score_column, "grade"]
    ].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="search_results.csv",
        mime="text/csv",
    )

    _render_rows(filtered, service)


if __name__ == "__main__":
    main()
