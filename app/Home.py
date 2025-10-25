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

import pandas as pd
import streamlit as st

from app.utils import load_service_or_error, render_nav, quick_nav_button

def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Nutrition Explorer")
    render_nav("Home")
    service = load_service_or_error()
    dataset = service.dataset
    score_col = service.score_column

    st.markdown(
        """
        Explore USDA FoodData Central foods, track nutrient strengths, and discover
        healthier swaps powered by our scoring model. Use the quick actions below to dive in.
        """
    )

    metrics = st.columns(3)
    with metrics[0]:
        st.metric("Foods ingested", f"{len(dataset):,}")
    with metrics[1]:
        avg = dataset[score_col].dropna().mean()
        st.metric("Average score", f"{avg:.1f}" if pd.notna(avg) else "—")
    with metrics[2]:
        grade_share = (
            dataset["grade"].isin(["A", "B"]).mean() * 100 if "grade" in dataset.columns else float("nan")
        )
        st.metric("Grade A/B share", f"{grade_share:.1f}%" if pd.notna(grade_share) else "—")

    st.divider()
    st.subheader("Quick actions")
    actions = st.columns(3)
    with actions[0]:
        quick_nav_button("🔎 Search foods", "Search.py")
    with actions[1]:
        quick_nav_button("📊 Compare items", "Compare.py")
    with actions[2]:
        quick_nav_button("🔁 Find substitutes", "Substitute.py")

    st.divider()
    st.subheader("Nutrient highlights")
    highlight_cols = st.columns(2)
    if "sodium_mg_per100g" in dataset.columns:
        low_sodium = dataset["sodium_mg_per100g"].le(140).sum()
        highlight_cols[0].metric("Low sodium (≤140mg/100g)", f"{low_sodium:,}")
    else:
        highlight_cols[0].metric("Low sodium (≤140mg/100g)", "—")
    if "fiber_g_per100g" in dataset.columns:
        high_fiber = dataset["fiber_g_per100g"].ge(5).sum()
        highlight_cols[1].metric("High fiber (≥5g/100g)", f"{high_fiber:,}")
    else:
        highlight_cols[1].metric("High fiber (≥5g/100g)", "—")

    st.caption(
        "Data sourced from USDA FoodData Central. Scores and grades come from the trained nutrition model."
    )


if __name__ == "__main__":
    main()
