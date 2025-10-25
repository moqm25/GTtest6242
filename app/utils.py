"""Shared loaders and helpers for the Nutrition Explorer app."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "DATA" / "processed",
    ROOT / "data" / "processed",
]
MODELS_DIR = ROOT / "models"
NUTRIENT_BASES = [
    "energy_kcal",
    "protein_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "added_sugar_g",
    "sodium_mg",
    "sat_fat_g",
]
NAV_ITEMS = [
    ("Home", "Home.py"),
    ("Search", "Search.py"),
    # ("Detail", "Detail.py"),
    ("Compare", "Compare.py"),
    ("Substitute", "Substitute.py"),
]

if TYPE_CHECKING:  # pragma: no cover
    from src.service.predictor import PredictorService

LOGGER = logging.getLogger(__name__)
_PREDICTOR_FAILED = object()


def _resolve_processed_path(filename: str) -> Path:
    for base in DATA_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return candidate
    # fall back to first candidate even if missing to surface helpful error upstream
    return DATA_CANDIDATES[0] / filename


@st.cache_data(show_spinner=False)
def load_foods() -> pd.DataFrame:
    """Load the foods dataset with expected alias columns."""
    path = _resolve_processed_path("foods_nutrients.parquet")
    if not path.exists():
        raise FileNotFoundError(
            "foods_nutrients.parquet not found. Run make build to generate processed data."
        )
    try:
        df = pd.read_parquet(path).copy()
    except OSError as exc:
        raise RuntimeError(
            "Failed to load foods_nutrients.parquet. This is often caused by "
            "an incompatible pyarrow build; try updating pyarrow or rebuilding the dataset."
        ) from exc

    # Ensure both score columns exist for downstream references
    if "score" not in df.columns and "nutrition_score" in df.columns:
        df["score"] = df["nutrition_score"]
    if "nutrition_score" not in df.columns and "score" in df.columns:
        df["nutrition_score"] = df["score"]

    for base in NUTRIENT_BASES:
        per100_col = f"{base}_per100g"
        perserving_col = f"{base}_perserving"
        has_per100 = per100_col in df.columns
        has_perserving = perserving_col in df.columns
        if not has_per100 and has_perserving:
            df[per100_col] = pd.NA
        if not has_perserving and has_per100:
            df[perserving_col] = pd.NA

    # Guarantee optional text columns exist to avoid KeyErrors
    for column in ("brand_owner", "food_category", "wweia_category", "grade"):
        if column not in df.columns:
            df[column] = pd.NA

    return df


@st.cache_data(show_spinner=False)
def load_nn() -> pd.DataFrame:
    """Load nearest-neighbor dataframe if available."""
    path = _resolve_processed_path("nn_index.parquet")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted files
        LOGGER.warning("Failed to read nn_index.parquet: %s", exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_importances() -> Dict[str, Any]:
    """Load importance coefficients for tooltip/context if present."""
    path = MODELS_DIR / "coef.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Invalid coef.json: %s", exc)
        return {}


def _format_number(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    fmt = f"{{:.{decimals}f}}".format(number)
    return f"{fmt}{suffix}"


def safe_serving_value(row: pd.Series, column: str, unit: str = "", decimals: int = 0) -> str:
    value = row.get(column)
    suffix = f" {unit}" if unit else ""
    return _format_number(value, decimals=decimals, suffix=suffix)


def find_items(df: pd.DataFrame, q: str, max_rows: int = 500) -> pd.DataFrame:
    """Case-insensitive substring search over name and brand owner."""
    if not q:
        return df.head(0)
    needles = [token.strip() for token in q.lower().split() if token.strip()]
    if not needles:
        return df.head(0)

    def contains(series: pd.Series, needle: str) -> pd.Series:
        return series.fillna("").str.lower().str.contains(needle, na=False)

    mask = pd.Series(True, index=df.index)
    for needle in needles:
        name_match = contains(df["description"], needle)
        brand_match = contains(df["brand_owner"], needle)
        mask &= name_match | brand_match

    results = df.loc[mask].copy()
    if results.empty:
        # fallback to first token match to keep UX forgiving
        primary = needles[0]
        results = df[contains(df["description"], primary) | contains(df["brand_owner"], primary)].copy()
    return results.head(max_rows)


def link_to(page: str, **params: Any) -> str:
    """Return a URL pointing at a registered Streamlit page with optional query params."""
    return page_url(page, **params)


def _resolve_page(page: str) -> Path:
    return (APP_DIR / page).resolve()


def _find_page_info(page: str) -> Dict[str, Any]:
    ctx = get_script_run_ctx()
    if not ctx:
        return {}
    target = str(_resolve_page(page))
    for info in ctx.pages_manager.get_pages().values():
        if info.get("script_path") == target:
            return info
    return {}


def page_path(page: str) -> str:
    """Return the path portion for a registered page (e.g. '/detail')."""
    info = _find_page_info(page)
    url_path = info.get("url_pathname")
    if url_path is None:
        return ""
    return "/" if url_path == "" else f"/{url_path}"


def page_url(page: str, **params: Any) -> str:
    """Return a URL for a page, including optional query parameters."""
    base = page_path(page)
    query = {key: value for key, value in params.items() if value is not None}
    if not base:
        fallback = {"page": page}
        fallback.update(query)
        return "?" + urlencode(fallback, doseq=True)
    if not query:
        return base
    return f"{base}?{urlencode(query, doseq=True)}"


def page_link_button(
    page: str,
    label: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    disabled: bool = False,
    width: str = "content",
    button_type: str = "secondary",
    icon: Optional[str] = None,
) -> None:
    """Render a link-style button that navigates to another Streamlit page."""
    url = page_url(page, **(params or {}))
    st.link_button(
        label,
        url=url,
        disabled=disabled,
        width=width,
        type=button_type,
        icon=icon,
    )

def _switch_page_fallback(page: str) -> bool:
    """Attempt to switch to a page using several candidate script paths."""
    candidates = []
    if page.endswith(".py"):
        candidates.append(page)
        candidates.append(f"app/{page}")
    else:
        script = f"{page}.py"
        candidates.extend([script, f"app/{script}"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            st.switch_page(candidate)
            return True
        except Exception:
            continue
    return False


def quick_nav_button(label: str, page: str) -> None:
    """
    Render a navigation button that prefers Streamlit's native page_link
    but gracefully falls back to switch_page for direct script runs.
    """
    script = page if page.endswith(".py") else f"{page}.py"
    try:
        st.page_link(script, label=label, width="stretch")
        return
    except Exception:
        pass
    # Fallback for contexts where navigation APIs aren't initialised (e.g. direct script run)
    if st.button(label, width="stretch"):
        if not _switch_page_fallback(script):
            st.warning("Unable to navigate to the requested page. Use the sidebar navigation instead.")

def render_nav(active: str) -> None:
    """Display a top navigation bar using Streamlit page links."""
    cols = st.columns(len(NAV_ITEMS))
    for col, (label, page) in zip(cols, NAV_ITEMS):
        with col:
            nav_label = f"· {label} ·" if label == active else label
            page_info = _find_page_info(page)
            if page_info.get("url_pathname") is not None:
                st.page_link(
                    page,
                    label=nav_label,
                    disabled=label == active,
                    width="stretch",
                )
            else:
                page_link_button(
                    page,
                    nav_label,
                    disabled=label == active,
                    width="stretch",
                    button_type="primary" if label == active else "secondary",
                )


def load_service_or_error() -> "AppService":
    """Return the cached service or surface a user-friendly error."""
    try:
        return get_service()
    except Exception as exc:  # pragma: no cover - Streamlit should show message
        st.error(
            "We couldn't load the processed nutrition dataset. "
            "Please rerun the data pipeline (`make build`) or check your local PyArrow installation."
        )
        st.exception(exc)
        st.stop()

@dataclass(slots=True)
class AppService:
    dataset: pd.DataFrame
    neighbors: pd.DataFrame
    importances: Dict[str, Any]
    processed_dir: Path
    _predictor: object = field(init=False, default=None, repr=False)
    _score_column_cache: Optional[str] = field(init=False, default=None, repr=False)

    @property
    def score_column(self) -> str:
        value = self._score_column_cache
        if value is None:
            value = "score" if "score" in self.dataset.columns else "nutrition_score"
            object.__setattr__(self, "_score_column_cache", value)
        return value

    def nutrient_cols(self, per: str) -> List[str]:
        suffix = "_per100g" if per == "per100g" else "_perserving"
        columns: List[str] = []
        for base in NUTRIENT_BASES:
            column_name = f"{base}{suffix}"
            if column_name in self.dataset.columns:
                columns.append(column_name)
        return columns

    def predict_probs(self, fdc_id: int) -> Dict[str, Any]:
        predictor = self._get_predictor()
        if predictor is None:
            return {}
        try:
            return predictor.predict(fdc_id)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Prediction failed for %s: %s", fdc_id, exc)
            return {}

    def format_badges(self, row: pd.Series) -> Dict[str, str]:
        score = _format_number(row.get(self.score_column), decimals=1)
        grade = row.get("grade") if pd.notna(row.get("grade")) else "—"
        kcal = safe_serving_value(row, "energy_kcal_perserving", unit="kcal", decimals=0)
        sodium = safe_serving_value(row, "sodium_mg_perserving", unit="mg", decimals=0)
        sugar = safe_serving_value(row, "sugar_g_perserving", unit="g", decimals=1)
        return {
            "score": score,
            "grade": grade,
            "kcal_serv": kcal,
            "sodium_serv": sodium,
            "sugar_serv": sugar,
        }

    def _get_predictor(self) -> Optional["PredictorService"]:
        if self._predictor is _PREDICTOR_FAILED:
            return None
        if self._predictor is None:
            try:
                from src.service.predictor import PredictorService  # local import for small apps

                self._predictor = PredictorService(processed_dir=self.processed_dir)
            except Exception as exc:
                LOGGER.warning("PredictorService unavailable: %s", exc)
                self._predictor = _PREDICTOR_FAILED
        return None if self._predictor is _PREDICTOR_FAILED else self._predictor


@st.cache_resource(show_spinner=False)
def get_service() -> AppService:
    dataset = load_foods()
    neighbors = load_nn()
    processed_dir = _resolve_processed_path("foods_nutrients.parquet").parent
    return AppService(
        dataset=dataset,
        neighbors=neighbors,
        importances=load_importances(),
        processed_dir=processed_dir,
    )
