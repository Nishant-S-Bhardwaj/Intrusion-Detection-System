import pandas as pd

# Load the full dataset
full_path = r"N:\DATASET\bot_iot\10-best features\UNSW_2018_IoT_Botnet_Final_10_Best.csv"
df_full = pd.read_csv(full_path, delimiter=';')
print(f"Full dataset size: {len(df_full)} rows")

# Sample one third of the data (as in your script)
df_sampled = df_full.sample(frac=1/3, random_state=42)
print(f"Sampled dataset size (used in model.py): {len(df_sampled)} rows")