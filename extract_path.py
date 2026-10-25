import cv2, numpy as np, pandas as pd

START, END = 11100, 16000
cap = cv2.VideoCapture("wood_fire_thermal.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

ret, frame = cap.read()
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def new_points(gray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=15)

pts = new_points(prev)
ids = np.arange(len(pts))
next_id = len(pts)
rows = [(START, int(i), float(p[0][0]), float(p[0][1])) for i, p in zip(ids, pts)]

frame_id = START
while frame_id < END:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=2)
    ok = status.flatten() == 1
    pts, ids = new[ok], ids[ok]

    for i, p in zip(ids, pts):
        rows.append((frame_id, int(i), float(p[0][0]), float(p[0][1])))

    # top up tracked points when too many are lost
    if len(pts) < 30:
        extra = new_points(gray)
        if extra is not None:
            extra = extra[:50 - len(pts)]
            pts = np.vstack([pts, extra]) if len(pts) else extra
            ids = np.concatenate([ids, np.arange(next_id, next_id + len(extra))])
            next_id += len(extra)
    prev = gray

pd.DataFrame(rows, columns=["frame_id", "point_id", "x", "y"]).to_csv("path_coordinates.csv", index=False)
print("Saved", len(rows), "coordinate rows")