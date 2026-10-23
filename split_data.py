import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset
df = pd.read_csv("clean_em_wave_dataset.csv")

# 2. Split the data (80% training, 20% testing)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Create the required folders
os.makedirs("train_data", exist_ok=True)
os.makedirs("test_data", exist_ok=True)

# 4. Save the datasets
train_df.to_csv("train_data/train.csv", index=False)
test_df.to_csv("test_data/test.csv", index=False)

print(f"Saved {len(train_df)} rows to train_data/train.csv")
print(f"Saved {len(test_df)} rows to test_data/test.csv")