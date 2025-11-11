import json, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
with open("configs/config.json","r", encoding="utf-8") as f:
    cfg = json.load(f)
dfs = [pd.read_csv(p) for p in cfg["data_paths"]]
parts = []
for df in dfs:
    n = len(df)
    parts.append(df.tail(int(n*(1-cfg["train_split"]))))
test_df = pd.concat(parts, ignore_index=True)
X = test_df.drop(columns=[cfg["target_col"]]).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
y_raw = test_df[cfg["target_col"]].astype(str).values
le = LabelEncoder(); le.fit(cfg["class_names"])
y = le.transform(y_raw)
weights = np.load("artifacts/global_final.npz")
W, b = weights["W"], weights["b"]
logits = X @ W + b
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)
y_pred = probs.argmax(axis=1)
print("[Eval] Combined holdout accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred, target_names=cfg["class_names"], zero_division=0))
