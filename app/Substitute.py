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

from typing import Any, Dict

import yaml
import streamlit as st

from app.utils import load_service_or_error, page_link_button, render_nav
from src.reco import get_substitutions

DEFAULT_CONSTRAINTS = {
    "min_score_gain": 10,
    "max_kcal_delta": 100,
    "sodium_cap": 500,
    "form_match": True,
    "group_match": True,
}


def _load_reco_defaults() -> Dict[str, Any]:
    cfg_path = CONFIGS / "reco.yml"
    if not cfg_path.exists():
        return DEFAULT_CONSTRAINTS.copy()
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update({k: data.get(k, v) for k, v in DEFAULT_CONSTRAINTS.items()})
    return constraints


def _current_fdc_id(dataset) -> int | None:
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
        candidate = int(raw)
    except (TypeError, ValueError):
        return None
    if candidate in dataset["fdc_id"].values:
        return candidate
    return None


def _constraint_panel(defaults: Dict[str, Any]) -> Dict[str, Any]:
    with st.expander("Constraints", expanded=False):
        form_match = st.checkbox("Match form (e.g., solid vs beverage)", value=defaults["form_match"])
        group_match = st.checkbox("Match food group", value=defaults["group_match"])
        min_score_gain = st.number_input(
            "Minimum score gain",
            min_value=0,
            max_value=100,
            value=int(defaults["min_score_gain"]),
            step=1,
        )
        max_kcal_delta = st.number_input(
            "Max kcal delta (per serving)",
            min_value=0,
            value=int(defaults["max_kcal_delta"]),
            step=10,
        )
        sodium_cap = st.number_input(
            "Sodium cap (mg per serving)",
            min_value=0,
            value=int(defaults["sodium_cap"]),
            step=25,
        )
    return {
        "form_match": form_match,
        "group_match": group_match,
        "min_score_gain": min_score_gain,
        "max_kcal_delta": max_kcal_delta,
        "sodium_cap": sodium_cap,
    }


def _render_cards(subs, dataset, service) -> None:
    bucket = st.session_state.setdefault("compare_ids", set())
    for _, row in subs.iterrows():
        fdc_id = int(row["neighbor_id"])
        match = dataset.loc[dataset["fdc_id"] == fdc_id]
        if match.empty:
            continue
        food = match.iloc[0]
        badges = service.format_badges(food)
        st.subheader(food.get("description", "Unknown item"))
        st.caption(f"Group: {food.get('food_category', '—')}")
        cols = st.columns(3)
        cols[0].metric("Score", badges["score"])
        cols[1].metric("Grade", badges["grade"])
        cols[2].metric("kcal / serving", badges["kcal_serv"])
        st.write(f"Why: {row.get('why', '—')}")
        ccols = st.columns(3)
        with ccols[0]:
            page_link_button("Detail.py", label="View detail →", params={"fdc_id": fdc_id})
        with ccols[1]:
            disabled = len(bucket) >= 3 and fdc_id not in bucket
            if st.button("Compare +", key=f"compare_sub_{fdc_id}", disabled=disabled):
                bucket.add(fdc_id)
        with ccols[2]:
            page_link_button("Substitute.py", label="More swaps →", params={"fdc_id": fdc_id})
        st.divider()


def main() -> None:
    st.set_page_config(page_title="Nutrition Explorer", page_icon="🥗", layout="wide")
    st.title("Find Substitutes")
    render_nav("Substitute")
    service = load_service_or_error()
    dataset = service.dataset

    st.markdown(
        "Pick a baseline food, tweak the substitution constraints, and surface healthier neighbors "
        "from the recommendation index."
    )

    fdc_id = _current_fdc_id(dataset)
    options = dataset[["fdc_id", "description"]].copy()
    options["label"] = options.apply(lambda r: f"{r['description']} ({r['fdc_id']})", axis=1)
    default_index = 0
    if fdc_id is not None:
        try:
            default_index = options.index[options["fdc_id"] == fdc_id][0]
        except IndexError:
            default_index = 0
    choice = st.selectbox("Select a food", options["label"], index=default_index)
    fdc_id = int(choice.split("(")[-1].strip(")"))

    base_row = dataset.loc[dataset["fdc_id"] == fdc_id].iloc[0]
    badges = service.format_badges(base_row)
    st.markdown(
        f"**Selected:** {base_row['description']}  \n"
        f"Grade {badges['grade']} · Score {badges['score']} · kcal/serv {badges['kcal_serv']}"
    )

    defaults = _load_reco_defaults()
    constraints = _constraint_panel(defaults)

    if st.button("Find substitutes", type="primary"):
        subs = get_substitutions(
            fdc_id,
            k=5,
            constraints=constraints,
            foods=dataset,
            nn_index=service.neighbors if not service.neighbors.empty else None,
        )
        if subs.empty:
            st.warning("No substitutions met the current constraints. Try relaxing them.")
            return
        enriched = subs.merge(
            dataset,
            left_on="neighbor_id",
            right_on="fdc_id",
            how="left",
            suffixes=("", "_neighbor"),
        )
        _render_cards(enriched, dataset, service)
    else:
        st.info("Adjust constraints and click **Find substitutes** to see recommendations.")


if __name__ == "__main__":
    main()
