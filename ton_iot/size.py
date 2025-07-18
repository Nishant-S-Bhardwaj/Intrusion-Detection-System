import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Pattern for your files (Network_dataset_1.csv, Network_dataset_2.csv, ...)
DATASET_GLOB = 'Network_dataset_*.csv'

# Load and merge all files
all_files = sorted(glob.glob(DATASET_GLOB))
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# Try to find the class/label column
possible_label_cols = [col for col in df.columns if col.lower() in ['label', 'class', 'attack_cat', 'target']]
if possible_label_cols:
    label_col = possible_label_cols[0]
else:
    raise ValueError('No class/label column found! Please check your dataset.')

# Plot class distribution
plt.figure(figsize=(10, 6))
class_counts = df[label_col].value_counts().sort_values(ascending=False)
class_counts.plot(kind='bar')
plt.title('Class Distribution in TON_IoT Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

print(f"Class distribution plot saved as class_distribution.png.\nClass counts:\n{class_counts}") 