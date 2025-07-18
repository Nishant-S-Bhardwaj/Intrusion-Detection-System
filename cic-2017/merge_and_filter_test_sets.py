import pandas as pd

# Paths to the test datasets
path1 = r'N:\DATASET\cic-2017\preprocessing\final_test_set.csv'
path2 = r'N:\DATASET\cic-2017\preprocessing2.ipynb\final_test_set.csv'

# Load both test datasets
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# Merge the datasets (concatenate rows)
merged = pd.concat([df1, df2], ignore_index=True)

# List of 32 features to keep (from your training set)
features = [
    'Bwd Header Length', 'PSH Flag Count', 'Packet Length Variance', 'Init_Win_bytes_forward',
    'ACK Flag Count', 'Active Min', 'Flow IAT Min', 'act_data_pkt_fwd', 'Flow Duration',
    'Bwd IAT Max', 'Destination Port', 'Bwd IAT Mean', 'Fwd IAT Min', 'Init_Win_bytes_backward',
    'Fwd Packet Length Min', 'Subflow Bwd Bytes', 'Fwd Header Length.1', 'Bwd IAT Total',
    'Fwd Packet Length Max', 'Min Packet Length', 'Fwd Packet Length Mean', 'Flow Bytes/s',
    'FIN Flag Count', 'Bwd Packets/s', 'Fwd IAT Mean', 'Bwd IAT Min', 'Flow IAT Mean',
    'Bwd IAT Std', 'Bwd Packet Length Min', 'Down/Up Ratio', 'min_seg_size_forward', 'Label'
]

# Keep only the 32 features (and Label if present)
filtered = merged[[col for col in features if col in merged.columns]]

# Save the merged and filtered test set
filtered.to_csv(r'N:\DATASET\cic-2017\preprocessing\merged_test_set.csv', index=False)

print('Merged and filtered test set saved to preprocessing/merged_test_set.csv')
