# EM Wave Path Prediction from Wood-Burning Data

A proof-of-concept machine learning pipeline. It tracks points in a Schlieren video of a wood fire (heat haze) and predicts where those points move next. The paths are a visual analogy for how radio signals bend in hot turbulent air. No real radio data was used.

Course: Fundamentals of AI/ML (PHC-351), IIT Roorkee. Author: Umar Shahid.

## Files
* `extract_path.py`: tracks points in the video and saves their x, y coordinates.
* `path_coordinates.csv`: the dataset (frame, point ID, x, y).
* `train_model.py`: trains and compares the naive baseline, Ridge regression and Random Forest, using a time-ordered train/validation/test split.
* `evaluate.py`: measures error for predicting 1 to 15 frames ahead.
* `app.py`: Streamlit dashboard that plots real vs predicted paths.
* `path_predictor.pkl`: the trained model.

## How to run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

To redo everything, put the video in this folder as `wood_fire_thermal.mp4` and run `extract_path.py`, then `train_model.py`, then `evaluate.py`.

## Results (test period, unseen later frames)
| Frames ahead | Model error | Naive error |
|---|---|---|
| 1 | 0.39 px | 0.55 px |
| 5 | 1.44 px | 2.15 px |
| 15 | 5.99 px | 8.05 px |

## Limitations
One video, one fire type, no radio measurements. Do not use for real rescue decisions.

## Links
* Dataset: (add link)
* Report: (add link)
* Video: (add link)
* Live dashboard: (add link)
