from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .progress import ProgressLogger, progress_for_iterable, timed_step

ROOT = Path(__file__).resolve().parents[1]
CFG_RECO = ROOT / "configs" / "reco.yml"
DATA_PARQUET = ROOT / "data" / "processed" / "foods_nutrients.parquet"
NN_PARQUET = ROOT / "data" / "processed" / "nn_index.parquet"

# ---- utils -----------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Choose nutrient features for embeddings (per100g only).
    Drop columns that are entirely NaN to avoid degenerate stats.
    """
    candidates = [
        "protein_g_per100g", "fiber_g_per100g", "potassium_mg_per100g",
        "added_sugar_g_per100g", "sugar_g_per100g", "sodium_mg_per100g",
        "sat_fat_g_per100g", "total_fat_g_per100g", "carbs_g_per100g",
        "energy_kcal_per100g", "mono_fat_g_per100g", "poly_fat_g_per100g"
    ]
    present = [c for c in candidates if c in df.columns]
    # Drop features that are all NaN
    keep = [c for c in present if df[c].notna().any()]
    return keep

def _robust_scale(X: np.ndarray) -> np.ndarray:
    """
    Robust scaling: (x - median) / IQR, with NaNs -> 0.0 after scaling.
    """
    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = (q3 - q1)
    iqr[iqr == 0] = 1.0
    Xs = (X - med) / iqr
    return np.nan_to_num(Xs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def _eligible_rows(df: pd.DataFrame, feat_cols: List[str], min_feats: int) -> pd.Series:
    has_serv = df["serving_gram_weight"].notna()
    nonnull = df[feat_cols].notna().sum(axis=1) >= min_feats
    return has_serv & nonnull

def _maybe_limit(df: pd.DataFrame, max_items: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_items:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.values, size=max_items, replace=False)
    return df.loc[np.sort(idx)]

def _build_knn(emb: np.ndarray, n_neighbors: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit KNN and return (indices, distances). Distances are cosine if metric='cosine'.
    """
    n_neighbors = min(n_neighbors + 1, emb.shape[0])  # +1 to include self
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb, return_distance=True)
    # Drop self neighbor at position 0
    return idx[:, 1:], dist[:, 1:]

# ---- index builder ----------------------------------------------------------

def build_index(
    cfg: dict,
    df: pd.DataFrame,
    progress: Optional[ProgressLogger] = None,
) -> pd.DataFrame:
    feat_cols = _select_features(df)
    if not feat_cols:
        raise RuntimeError("No usable feature columns found for embeddings.")

    if progress:
        progress.step(f"Selected {len(feat_cols)} features")

    eligible = _eligible_rows(df, feat_cols, cfg.get("min_features_present", 6))
    work = df.loc[eligible, ["fdc_id", "form", "food_category", "score",
                              "kcal_per_serv", "sodium_mg_perserving"] + feat_cols].copy()

    # Optional narrowing: prefer non-branded rows first to reduce size/NaN rate, then sample
    if "data_type" in df.columns:
        foundation_mask = df["data_type"].isin(["SR Legacy", "Foundation"])
        pri = df.loc[eligible & foundation_mask, ["fdc_id"]].index
        sec = df.loc[eligible & ~foundation_mask, ["fdc_id"]].index
        # keep order: foundation first, then branded
        work = pd.concat([df.loc[pri, work.columns], df.loc[sec, work.columns]], axis=0)

    if progress:
        progress.step(f"Eligible items: {len(work):,}")

    work = _maybe_limit(work, cfg.get("max_items", 250000), cfg.get("random_seed", 13))
    if progress:
        progress.step(f"Working set size: {len(work):,}")

    # Build feature matrix
    X = work[feat_cols].to_numpy(dtype=float, copy=False)
    X = _robust_scale(X)
    if progress:
        progress.step("Features scaled")

    # cosine distance -> similarity = 1 - distance
    with timed_step("Fitting KNN"):
        idx, dist = _build_knn(X, cfg.get("n_neighbors", 50), cfg.get("metric", "cosine"))
    sim = 1.0 - dist

    # Flatten to long DataFrame
    rows = []
    fdc_ids = work["fdc_id"].to_numpy()
    base_scores = work["score"].to_numpy()
    base_kcal = work["kcal_per_serv"].to_numpy()
    base_sodium = work["sodium_mg_perserving"].to_numpy()

    for i in progress_for_iterable(range(idx.shape[0]), "Flatten neighbors", length=idx.shape[0]):
        src = fdc_ids[i]
        src_score = base_scores[i]
        for j, neighbor_pos in enumerate(idx[i]):
            nb = fdc_ids[neighbor_pos]
            rows.append((src, nb, float(sim[i, j]), float(src_score)))

    nn_df = pd.DataFrame(rows, columns=["fdc_id", "neighbor_id", "cos_sim", "base_score"])
    if progress:
        progress.step(f"Constructed neighbor rows: {len(nn_df):,}")
    return nn_df


def load_reco_config(path: Optional[Path] = None) -> Dict:
    cfg_path = Path(path) if path else CFG_RECO
    return _read_yaml(cfg_path)

# ---- public API: get_substitutions -----------------------------------------

def get_substitutions(
    food_id: int,
    k: int = 5,
    constraints: Optional[Dict] = None,
    foods: Optional[pd.DataFrame] = None,
    nn_index: Optional[pd.DataFrame] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Rank neighbors by 0.6*similarity + 0.4*(score_gain/100) and apply constraints.
    """
    if foods is None:
        source = Path(processed_dir) / "foods_nutrients.parquet" if processed_dir else DATA_PARQUET
        foods = pd.read_parquet(source)

    if nn_index is None:
        source_index = Path(processed_dir) / "nn_index.parquet" if processed_dir else NN_PARQUET
        nn_index = pd.read_parquet(source_index)

    cfg = _read_yaml(CFG_RECO)
    defaults = cfg.get("constraints", cfg)
    effective_constraints = dict(defaults or {})
    if constraints:
        effective_constraints.update(constraints)
    w_sim = cfg["objective_weights"]["similarity"]
    w_gain = cfg["objective_weights"]["score_gain"]

    base = foods.loc[foods["fdc_id"] == food_id]
    if base.empty:
        return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    base = base.iloc[0]

    if "fdc_id" in nn_index.columns:
        neighbor_rows = nn_index.loc[nn_index["fdc_id"] == food_id]
    else:
        try:
            neighbor_rows = nn_index.loc[[food_id]].reset_index()
        except KeyError:
            return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
    cand = neighbor_rows.merge(
        foods[["fdc_id", "score", "grade", "form", "food_category",
               "kcal_per_serv", "sodium_mg_perserving"]],
        left_on="neighbor_id", right_on="fdc_id", how="left", suffixes=("","_nb")
    )

    # Constraints
    if effective_constraints.get("form_match", True) and pd.notna(base["form"]):
        cand = cand[cand["form"] == base["form"]]
    if effective_constraints.get("group_match", True) and pd.notna(base["food_category"]):
        cand = cand[cand["food_category"] == base["food_category"]]
    if "max_kcal_delta" in effective_constraints and pd.notna(base["kcal_per_serv"]):
        cand = cand[
            cand["kcal_per_serv"].sub(base["kcal_per_serv"]).abs() <= effective_constraints["max_kcal_delta"]
        ]
    if "sodium_cap" in effective_constraints:
        cand = cand[cand["sodium_mg_perserving"] <= effective_constraints["sodium_cap"]]

    cand["score_gain"] = cand["score"].sub(base["score"])
    cand = cand[cand["score_gain"] >= effective_constraints.get("min_score_gain", 10)]
    cand["score_gain"] = cand["score_gain"].fillna(0)

    cand["obj"] = w_sim * cand["cos_sim"] + w_gain * (cand["score_gain"] / 100.0)
    cand = cand.sort_values("obj", ascending=False)

    # build why string
    def fmt(r):
        parts = []
        parts.append(f"score +{int(round(r['score_gain']))}")
        if pd.notna(r["sodium_mg_perserving"]):
            parts.append(f"sodium {int(round(r['sodium_mg_perserving']))}mg")
        if pd.notna(r["kcal_per_serv"]):
            parts.append(f"kcal {int(round(r['kcal_per_serv']))}")
        return ", ".join(parts)

    out = cand.head(k).copy()
    if out.empty:
        fallback = foods[foods["score"] > base["score"]].copy()
        if effective_constraints.get("form_match", True) and pd.notna(base["form"]):
            fallback = fallback[fallback["form"] == base["form"]]
        if effective_constraints.get("group_match", True) and pd.notna(base["food_category"]):
            fallback = fallback[fallback["food_category"] == base["food_category"]]
        min_gain = effective_constraints.get("min_score_gain", 10)
        fallback = fallback[fallback["score"] >= base["score"] + min_gain]
        fallback = fallback.nlargest(k, "score")
        if fallback.empty:
            fallback = foods[foods["score"] >= base["score"] + min_gain]
            fallback = fallback.nlargest(k, "score")
            if fallback.empty:
                return pd.DataFrame(columns=["neighbor_id", "score", "grade", "why"])
        fallback = fallback.assign(neighbor_id=fallback["fdc_id"])
        fallback["why"] = [
            f"score +{int(max(0, round(row['score'] - base['score'])))}"
            for _, row in fallback.iterrows()
        ]
        fallback["score_gain"] = fallback["score"] - base["score"]
        return fallback[["neighbor_id", "score", "score_gain", "grade", "why"]]
    out["why"] = [fmt(row) for _, row in out.iterrows()]
    return out[["neighbor_id", "score", "score_gain", "grade", "why"]]

# ---- CLI --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build nutrient neighbor index.")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of the nn index.")
    args = p.parse_args()

    cfg = _read_yaml(CFG_RECO)
    progress = ProgressLogger("Rebuild neighbor index")
    progress.step("Loaded configuration")
    print(f"[reco] config: {json.dumps(cfg, indent=2)}")

    if not DATA_PARQUET.exists():
        raise SystemExit(f"Missing dataset: {DATA_PARQUET}")

    progress.step("Loading foods dataset")
    with timed_step("Loading foods dataset"):
        df = pd.read_parquet(DATA_PARQUET)
    # Sanity: required columns for constraints/scoring
    req = {"fdc_id","score","form","food_category","kcal_per_serv","sodium_mg_perserving"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")

    print(f"[reco] dataset rows: {len(df):,}")
    progress.step("Building neighbor index")
    with timed_step("Building neighbor index"):
        nn_df = build_index(cfg, df, progress=progress)
    print(f"[reco] built neighbor table with {len(nn_df):,} rows; writing {NN_PARQUET} ...")
    progress.step("Writing parquet")
    with timed_step("Writing nn_index.parquet"):
        NN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        nn_df.to_parquet(NN_PARQUET, index=False)
    progress.done("Neighbor index ready")
    print("[reco] done.")

if __name__ == "__main__":
    main()
