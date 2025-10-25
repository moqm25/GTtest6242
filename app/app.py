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

import streamlit as st


def main() -> None:
    """Configure multipage navigation and run the selected page."""
    pages = [
        st.Page("Home.py", title="Nutrition Explorer", icon="🥗", default=True),
        st.Page("Search.py", title="Search Foods", icon=":material/search:", url_path="search"),
        # st.Page("Detail.py", title="Food Detail", icon=":material/info:", url_path="detail"),
        st.Page("Compare.py", title="Compare Foods", icon=":material/bar_chart:", url_path="compare"),
        st.Page("Substitute.py", title="Find Substitutes", icon=":material/sync_alt:", url_path="substitute"),
    ]
    current_page = st.navigation(pages, position="hidden")
    current_page.run()


if __name__ == "__main__":
    main()
