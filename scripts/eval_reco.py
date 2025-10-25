import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

foods = pd.read_parquet(ROOT/"data/processed/foods_nutrients.parquet")
df = foods[["fdc_id","score","grade"]].copy()
cand = df[df["grade"].isin(["D","E"])].copy()
if len(cand) > 500:
    cand = cand.nsmallest(500, "score")
nn = pd.read_parquet(ROOT/"data/processed/nn_index.parquet")
subset = nn[nn["fdc_id"].isin(cand["fdc_id"])]
neighbor_scores = df.rename(columns={"fdc_id": "neighbor_id", "score": "neighbor_score"})
subset = subset.merge(neighbor_scores[["neighbor_id", "neighbor_score"]], on="neighbor_id", how="left")
subset["score_gain"] = subset["neighbor_score"] - subset["base_score"]
valid = subset[subset["score_gain"] >= 10]
covered = set(valid["fdc_id"])
cov = len(covered) / max(len(cand), 1)
print(f"Coverage on D/E items: {cov:.1%} ({len(covered)}/{len(cand)})")
