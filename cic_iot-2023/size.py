import pandas as pd
import glob

# Pattern for your files (Merged01.csv, Merged02.csv, ...)
DATASET_GLOB = 'Merged*.csv'

# Load and merge up to 10 files
all_files = sorted(glob.glob(DATASET_GLOB))[:10]
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# Print the number of files and total rows
print(f"Merged {len(df_list)} files, total rows: {df.shape}")