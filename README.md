# EM Wave Path Prediction

Physics-informed machine learning pipeline predicting electromagnetic signal deviation through turbulent thermal plumes using a Random Forest Regressor.

## Project Structure
* `clean_em_wave_dataset.csv`: Cleaned dataset isolating the active thermal plume.
* `train_data/` & `test_data/`: Explicitly separated 80/20 data splits.
* `generate_dataset.py`: OpenCV script for extracting optical features from video.
* `train_model.py`: Scikit-Learn training script.
* `app.py`: Streamlit web dashboard for live inference.
* `em_wave_predictor.pkl`: Trained model file.

## How to Run the Live Demo
1. Install dependencies: `pip install pandas scikit-learn streamlit joblib opencv-python`
2. Run the dashboard: `streamlit run app.py`
