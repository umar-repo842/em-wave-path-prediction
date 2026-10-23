import pandas as pd

# Load the raw 17,598-row dataset
df = pd.read_csv("em_wave_wood_burning_dataset.csv")

# Filter out the intro footage (keep frames 11100 to 16000)
clean_df = df[(df['frame_id'] >= 11100) & (df['frame_id'] <= 16000)]

# Save the cleaned dataset
clean_df.to_csv("clean_em_wave_dataset.csv", index=False)

print(f"Removed intro footage. Cleaned dataset now has {len(clean_df)} valid thermal frames.")