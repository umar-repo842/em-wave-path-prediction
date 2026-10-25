import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("path_coordinates.csv").sort_values(["point_id", "frame_id"])
K = 5  # use the last 5 positions

X, Y, F = [], [], []
for pid, g in df.groupby("point_id"):
    g = g[g.frame_id.diff().fillna(1) == 1]      # keep unbroken stretches
    xy = g[["x", "y"]].values
    for t in range(K, len(xy)):
        window = xy[t-K:t]
        X.append(np.diff(window, axis=0).flatten())   # last 4 moves (dx, dy) = 8 numbers
        Y.append(xy[t] - xy[t-1])                     # the next move
        F.append(g.frame_id.values[t])
X, Y, F = np.array(X), np.array(Y), np.array(F)
last = X[:, 6:8]                                      # most recent move

c1, c2 = np.quantile(F, [0.6, 0.8])
tr, va, te = F < c1, (F >= c1) & (F < c2), F >= c2

ridge = Ridge(alpha=1.0).fit(X[tr], Y[tr])
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[tr], Y[tr] - last[tr])

for name, m in [("val", va), ("test", te)]:
    print(name,
          "| baseline", round(mean_absolute_error(Y[m], last[m]), 3),
          "| ridge", round(mean_absolute_error(Y[m], ridge.predict(X[m])), 3),
          "| RF-correction", round(mean_absolute_error(Y[m], last[m] + rf.predict(X[m])), 3))

joblib.dump({"model": ridge, "test_start": float(c2)}, "path_predictor.pkl")
import os
cols = [f"move{i}_{a}" for i in range(1, 5) for a in ("dx", "dy")]   # the 4 past movements
for name, m in [("train", tr), ("val", va), ("test", te)]:
    os.makedirs(f"{name}_data", exist_ok=True)
    out = pd.DataFrame(X[m], columns=cols)
    out["target_dx"] = Y[m][:, 0]     # the next movement the model must predict
    out["target_dy"] = Y[m][:, 1]
    out["frame_id"] = F[m]
    out.to_csv(f"{name}_data/{name}.csv", index=False)