import pandas as pd, numpy as np, joblib

bundle = joblib.load("path_predictor.pkl")
model, test_start = bundle["model"], bundle["test_start"]
df = pd.read_csv("path_coordinates.csv")
test = df[df.frame_id >= test_start]

for H in [1, 3, 5, 10, 15]:
    em, en = [], []
    for pid, g in test.groupby("point_id"):
        g = g.sort_values("frame_id")
        run = (g.frame_id.diff() != 1).cumsum()
        for _, gr in g.groupby(run):
            xy = gr[["x", "y"]].values
            if len(xy) < 5 + H:
                continue
            real = xy[:5 + H]
            pred = list(real[:5])
            for _ in range(H):
                steps = np.diff(np.array(pred[-5:]), axis=0).flatten()
                pred.append(pred[-1] + model.predict([steps])[0])
            naive = real[4] + H * (real[4] - real[3])
            em.append(np.linalg.norm(pred[-1] - real[-1]))
            en.append(np.linalg.norm(naive - real[-1]))
    print(f"H={H:2d} | paths={len(em)} | model error={np.mean(em):.2f} px | naive error={np.mean(en):.2f} px")