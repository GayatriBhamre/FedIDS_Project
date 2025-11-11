"""
Standardize numeric CSV columns (z-score) excluding target column.
Usage: python tools/standardize_csv.py data/datasetA.csv label
"""
import sys, pandas as pd
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/standardize_csv.py <csv> <target_col>"); exit(0)
    path, target = sys.argv[1], sys.argv[2]
    df = pd.read_csv(path)
    feats = df.drop(columns=[target], errors="ignore").select_dtypes(include=["number"]).columns.tolist()
    for c in feats:
        mu, sd = df[c].mean(), df[c].std(ddof=0)
        if sd == 0 or pd.isna(sd): continue
        df[c] = (df[c] - mu) / sd
    df.to_csv(path, index=False)
    print("Standardized", path)
