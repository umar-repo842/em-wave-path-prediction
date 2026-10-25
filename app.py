import streamlit as st, pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt

st.title("🔥 Predicting Paths in a Fire's Heat Haze")
st.write("We track tiny points in a Schlieren video of a wood fire, then ask a model to "
         "predict where a point will travel, given only its first 5 positions.")

bundle = joblib.load("path_predictor.pkl")
model, test_start = bundle["model"], bundle["test_start"]
df = pd.read_csv("path_coordinates.csv")
test = df[df.frame_id >= test_start]          # frames the model never trained on

H = st.sidebar.slider("How many frames ahead to predict", 5, 30, 15)

@st.cache_data
def get_runs(H):
    runs = []
    for pid, g in test.groupby("point_id"):
        g = g.sort_values("frame_id")
        run = (g.frame_id.diff() != 1).cumsum()
        for _, gr in g.groupby(run):
            if len(gr) >= 5 + H:
                runs.append((int(pid), int(gr.frame_id.iloc[0]), gr[["x", "y"]].values[:5 + H]))
    return runs

runs = get_runs(H)
if not runs:
    st.warning("No long enough paths. Lower the slider.")
    st.stop()

i = st.sidebar.slider("Pick a path", 0, len(runs) - 1, 0)
pid, start_frame, real = runs[i]

pred = list(real[:5])
for _ in range(H):
    steps = np.diff(np.array(pred[-5:]), axis=0).flatten()
    pred.append(pred[-1] + model.predict([steps])[0])
pred = np.array(pred)

v = real[4] - real[3]
naive = np.array([real[4] + k * v for k in range(0, H + 1)])

fig, ax = plt.subplots()
ax.plot(real[:, 0], real[:, 1], "g-o", ms=3, label="Real path")
ax.plot(pred[4:, 0], pred[4:, 1], "r--o", ms=3, label="Model prediction")
ax.plot(naive[:, 0], naive[:, 1], "b:", label="Naive (straight line)")
ax.scatter(real[:5, 0], real[:5, 1], c="k", zorder=5, label="Positions given to model")
ax.invert_yaxis(); ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)"); ax.legend()
st.pyplot(fig)

err_m = np.linalg.norm(pred[-1] - real[-1])
err_n = np.linalg.norm(naive[-1] - real[-1])
c1, c2 = st.columns(2)
c1.metric("Model error at the end", f"{err_m:.1f} px")
c2.metric("Naive error at the end", f"{err_n:.1f} px")
st.caption(f"Point #{pid}, starting at frame {start_frame}. Errors grow the further ahead we predict. "
           "Turbulence is hard to forecast.")