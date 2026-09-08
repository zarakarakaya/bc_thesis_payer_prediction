import pandas as pd
import json
from pathlib import Path

BEST_DIR = Path("results") / "best"

with open(BEST_DIR / "feature_columns.json") as f:
    feature_columns = json.load(f)

df = pd.read_parquet("data/data_test_23.parquet")  # your 9k input
feature_df = df.drop(columns=["player_id"], errors="ignore")
encoded = pd.get_dummies(feature_df, columns=["network_name"])

missing_in_input = set(feature_columns) - set(encoded.columns)
extra_in_input = set(encoded.columns) - set(feature_columns)

print("Columns training expects but inference doesn't have (will be silently zero-filled):")
print(missing_in_input)
print("\nColumns inference has that training doesn't (will be silently dropped):")
print(extra_in_input)

print("\nTop-SHAP feature summaries in the new 9k batch:")
for feat in ["gem_spend", "gem_earn", "crates_claimed", "jobs_completed"]:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(df[feat].describe())
    else:
        print(f"\n{feat}: NOT PRESENT in input at all")