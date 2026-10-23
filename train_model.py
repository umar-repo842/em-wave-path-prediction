import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the explicitly separated datasets
train_df = pd.read_csv("train_data/train.csv")
test_df = pd.read_csv("test_data/test.csv")

# Define Features and Target
features = ['mean_brightness', 'thermal_shimmer_blur', 'refraction_edge_density']
X_train, y_train = train_df[features], train_df['target_em_deviation']
X_test, y_test = test_df[features], test_df['target_em_deviation']

# Train and Evaluate
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.6f}")
print(f"Accuracy (R2 Score): {r2_score(y_test, predictions):.4f}")

joblib.dump(model, "em_wave_predictor.pkl")